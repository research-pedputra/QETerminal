"""SFTP owner thread plus an independent PTY pump, so file IO cannot block typing."""
import base64
import hashlib
import posixpath
import queue
import socket
import stat
import threading
import uuid
import time
from pathlib import Path
from PySide6.QtCore import QThread, Signal
import paramiko

EDITOR_LIMIT = 20 * 1024 * 1024
STRUCTURE_LIMIT = 256 * 1024 * 1024


def fingerprint(key):
    return 'SHA256:' + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_remote(sftp, path, limit=EDITOR_LIMIT):
    info = sftp.stat(path)
    if info.st_size > limit:
        raise ValueError(f'File exceeds the {limit // 1024 // 1024} MiB limit.')
    with sftp.open(path, 'rb') as f:
        data = f.read(limit+1)
    if len(data) > limit: raise ValueError('File grew beyond the size limit while reading.')
    return data


def save_remote(sftp, path, data, expected, limit=EDITOR_LIMIT):
    """Optimistic conflict check, sibling temp upload, POSIX atomic replacement.

    Refuse a non-atomic overwrite on servers without posix-rename support.
    A server-side concurrent writer can still race the last check and rename.
    """
    path = sftp.normalize(path)  # Edit a symlink's target without replacing the symlink.
    if digest(read_remote(sftp, path, limit)) != expected:
        raise ValueError('Remote file changed since it was opened. Reload it or save a local copy before overwriting.')
    permissions = stat.S_IMODE(sftp.stat(path).st_mode)
    temporary = posixpath.join(posixpath.dirname(path), '.' + posixpath.basename(path) + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with sftp.open(temporary, 'wx') as f:
            f.write(data)
            f.flush()
        sftp.chmod(temporary, permissions)
        if digest(read_remote(sftp, path, limit)) != expected:
            raise ValueError('Remote file changed during upload; nothing was overwritten.')
        try:
            sftp.posix_rename(temporary, path)
        except OSError as exc:
            raise OSError('This server could not atomically replace the file. Save a local copy instead. ' + str(exc)) from exc
    finally:
        try: sftp.remove(temporary)
        except OSError: pass
    return digest(data)


def transfer_items(sftp, direction, sources, destination, stopping, progress=None):
    """Copy selected files/trees without overwriting existing destinations."""
    completed, errors = [], []
    def check():
        if stopping.is_set(): raise OSError('Transfer canceled: SSH disconnected.')
    def safe_name(name):
        if (not name or name in ('.', '..') or any(c in name for c in '/\\:<>"|?*')
                or any(ord(c) < 32 for c in name) or name.endswith((' ', '.'))
                or name.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *('COM'+str(i) for i in range(1, 10)), *('LPT'+str(i) for i in range(1, 10))}):
            raise ValueError('Unsupported filename: ' + name)
        return name
    def copy(source, target):
        check()
        local = Path(source if direction == 'upload' else target)
        remote = target if direction == 'upload' else source
        mode = local.lstat().st_mode if direction == 'upload' else sftp.lstat(remote).st_mode
        if stat.S_ISLNK(mode) or (direction == 'upload' and getattr(local, 'is_junction', lambda: False)()):
            raise ValueError('Symbolic links are not copied: ' + str(source))
        if stat.S_ISDIR(mode):
            # Exclusive directory creation also prevents merging into symlinks.
            if direction == 'upload':
                sftp.mkdir(remote)
                for child in local.iterdir(): copy(str(child), posixpath.join(remote, child.name))
            else:
                local.mkdir()
                for child in sftp.listdir_attr(remote):
                    name = safe_name(child.filename)
                    copy(posixpath.join(remote, name), str(local/name))
        elif stat.S_ISREG(mode):
            created = False
            total = local.stat().st_size if direction == 'upload' else sftp.stat(remote).st_size
            copied = 0
            if progress: progress(str(source), 0, total)
            try:
                if direction == 'upload':
                    with local.open('rb') as reader, sftp.open(remote, 'wx') as writer:
                        created = True
                        while True:
                            check(); chunk = reader.read(1024*1024)
                            if not chunk: break
                            writer.write(chunk); copied += len(chunk)
                            if progress: progress(str(source), copied, total)
                else:
                    with sftp.open(remote, 'rb') as reader, local.open('xb') as writer:
                        created = True
                        while True:
                            check(); chunk = reader.read(1024*1024)
                            if not chunk: break
                            writer.write(chunk); copied += len(chunk)
                            if progress: progress(str(source), copied, total)
            except Exception:
                if created:
                    try:
                        if direction == 'upload': sftp.remove(remote)
                        else: local.unlink()
                    except OSError: pass
                raise
        else: raise ValueError('Not a regular file or folder: ' + str(source))
    for source in sources:
        try:
            check()
            name = Path(source).name if direction == 'upload' else safe_name(posixpath.basename(source.rstrip('/')))
            target = posixpath.join(destination, name) if direction == 'upload' else str(Path(destination)/name)
            copy(source, target); completed.append(source)
        except Exception as exc:
            errors.append(f'{source}: {exc}')
            if stopping.is_set(): break
    return dict(completed=completed, errors=errors)


class AskHostKey(paramiko.MissingHostKeyPolicy):
    def __init__(self, worker): self.worker = worker

    def missing_host_key(self, client, hostname, key):
        worker = self.worker
        worker.trust_answer = False; worker.trust_event.clear()
        worker.host_key.emit(hostname, key.get_name(), fingerprint(key))
        while not worker.trust_event.wait(.1):
            if worker.stopping.is_set(): raise paramiko.SSHException('Connection canceled.')
        if not worker.trust_answer: raise paramiko.SSHException('Host key was not trusted.')
        client.get_host_keys().add(hostname, key.get_name(), key)
        worker.known_hosts.parent.mkdir(parents=True, exist_ok=True)
        client.save_host_keys(str(worker.known_hosts))


class InteractiveTransport(paramiko.Transport):
    """Route PAM/MFA prompts to the GUI, never to stdin or password guessing."""
    def __init__(self, *args, prompt_handler, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_handler = prompt_handler
        self.publickey_attempted = False

    def auth_publickey(self, username, key, event=None):
        self.publickey_attempted = True
        return super().auth_publickey(username, key, event=event)

    def auth_password(self, username, password, event=None, fallback=True):
        try:
            return super().auth_password(username, password, event=event, fallback=False)
        except paramiko.BadAuthenticationType as exc:
            if 'keyboard-interactive' not in exc.allowed_types: raise
            return self.auth_interactive(username, self.prompt_handler)

    def auth_interactive_dumb(self, username, handler=None, submethods=''):
        return self.auth_interactive(username, self.prompt_handler, submethods)


class SSHSession(QThread):
    connected = Signal(str)
    disconnected = Signal(str)
    host_key = Signal(str, str, str)
    result = Signal(str, str, object)
    failed = Signal(str, str, str)
    transfer_progress = Signal(str, str, int, int)
    output = Signal(bytes)
    auth_prompt = Signal(str, str, object)

    def __init__(self, config, known_hosts, parent=None):
        super().__init__(parent)
        self.config = dict(config)
        self.known_hosts = Path(known_hosts)
        self.session_id = uuid.uuid4().hex
        self.jobs = queue.Queue(); self.keys = queue.Queue()
        self.stopping = threading.Event(); self.trust_event = threading.Event()
        self.trust_answer = False
        self.auth_event = threading.Event(); self.auth_answers = None
        self.interactive_attempted = False
        self.client = None; self.channel = None
        self.pending_resize = (100, 24)
        self.output_lock = threading.Lock()
        self.output_buffer = bytearray()
        self.pump = None
        self.pump_error = None

    def request(self, operation, *args):
        token = uuid.uuid4().hex
        self.jobs.put((token, operation, args)); return token

    def send(self, data): self.keys.put(data)

    def resize_terminal(self, columns, rows): self.pending_resize = (columns, rows)

    def trust(self, answer):
        self.trust_answer = answer; self.trust_event.set()

    def answer_auth(self, answers):
        self.auth_answers = answers; self.auth_event.set()

    def interactive_auth(self, title, instructions, prompts):
        self.interactive_attempted = True
        if not prompts: return []
        self.auth_answers = None; self.auth_event.clear()
        self.auth_prompt.emit(title, instructions, prompts)
        deadline = time.monotonic() + 180
        while not self.auth_event.wait(.1):
            if self.stopping.is_set() or time.monotonic() > deadline:
                raise paramiko.AuthenticationException('Login canceled or prompt timed out.')
        answers = self.auth_answers; self.auth_answers = None
        if self.stopping.is_set() or answers is None:
            raise paramiko.AuthenticationException('Login canceled.')
        return answers

    def stop(self):
        self.stopping.set(); self.trust_event.set(); self.auth_event.set()
        # Closing the socket interrupts blocked network IO; no GUI-thread join.
        if self.client is not None: self.client.close()

    def take_output(self, limit=16384):
        with self.output_lock:
            result = bytes(self.output_buffer[:limit])
            del self.output_buffer[:limit]
            return result

    def pump_terminal(self):
        """Bound reads per turn; prioritize keys even when a remote job floods stdout."""
        last_size = None
        pending = b''
        try:
            while not self.stopping.wait(.005):
                channel = self.channel
                if channel.closed:
                    self.pump_error = 'Remote shell exited.'; break
                if self.pending_resize != last_size:
                    last_size = self.pending_resize
                    channel.resize_pty(width=last_size[0], height=last_size[1])
                for _ in range(32):
                    if not pending:
                        try: pending = self.keys.get_nowait()
                        except queue.Empty: break
                    if not channel.send_ready(): break
                    try:
                        sent = channel.send(pending)
                        if sent == 0: raise OSError('Remote shell closed.')
                        pending = pending[sent:]
                    except socket.timeout: break
                with self.output_lock:
                    capacity = 2*1024*1024-len(self.output_buffer)
                if capacity > 0 and channel.recv_ready():
                    chunk = channel.recv(min(capacity, 32768))
                    if chunk:
                        with self.output_lock: self.output_buffer.extend(chunk)
                        self.output.emit(chunk)  # Optional observers; GUI uses bounded pull instead.
                if channel.exit_status_ready() and not channel.recv_ready():
                    self.pump_error = 'Remote shell exited.'; break
        except Exception as exc:
            if not self.stopping.is_set(): self.pump_error = 'Terminal: ' + str(exc)

    def run(self):
        message = 'Disconnected.'
        try:
            config = self.config
            self.client = paramiko.SSHClient()
            self.client.load_system_host_keys()
            if self.known_hosts.exists(): self.client.load_host_keys(str(self.known_hosts))
            self.client.set_missing_host_key_policy(AskHostKey(self))
            try:
                self.client.connect(hostname=config['host'], port=config['port'], username=config['user'],
                                password=config.get('password') or None,
                                key_filename=config.get('key') or None,
                                passphrase=config.get('passphrase') or None,
                                look_for_keys=True, allow_agent=True, timeout=10, banner_timeout=10, auth_timeout=180,
                                transport_factory=lambda *a, **kw: InteractiveTransport(*a, prompt_handler=self.interactive_auth, **kw))
            except paramiko.SSHException as exc:
                transport = self.client.get_transport()
                # connect() does not try interactive auth when no password/key is supplied.
                # Never repeat an interactive attempt or bypass host-key verification.
                eligible = (isinstance(exc, paramiko.AuthenticationException)
                            or str(exc) == 'No authentication methods available'
                            or getattr(transport, 'publickey_attempted', False))
                if isinstance(exc, paramiko.BadAuthenticationType) and 'keyboard-interactive' not in exc.allowed_types: eligible = False
                if not eligible or self.interactive_attempted or self.stopping.is_set() or not transport or not transport.is_active(): raise
                transport.auth_interactive(config['user'], self.interactive_auth)
            if not self.client.get_transport().is_authenticated():
                raise paramiko.AuthenticationException('Additional authentication is required by the server.')
            config.pop('password', None); config.pop('passphrase', None)
            if self.stopping.is_set(): return
            self.client.get_transport().set_keepalive(30)
            sftp = self.client.open_sftp(); sftp.get_channel().settimeout(15)
            home = sftp.normalize(config.get('directory') or '.')
            self.channel = self.client.invoke_shell(term='xterm', width=100, height=24)
            self.channel.settimeout(.1)
            self.connected.emit(home)
            self.pump = threading.Thread(target=self.pump_terminal, daemon=True, name='SSH-PTY')
            self.pump.start()
            while not self.stopping.is_set():
                if not self.client.get_transport().is_active(): break
                if self.pump_error:
                    message = self.pump_error; break
                try: token, operation, args = self.jobs.get(timeout=.02)
                except queue.Empty: continue
                try:
                    if operation == 'list':
                        path = sftp.normalize(args[0])
                        rows = []
                        for info in sftp.listdir_attr(path):
                            if info.filename in ('.', '..'): continue
                            is_dir = stat.S_ISDIR(info.st_mode)
                            if stat.S_ISLNK(info.st_mode):
                                try: is_dir = stat.S_ISDIR(sftp.stat(posixpath.join(path, info.filename)).st_mode)
                                except OSError: pass
                            rows.append((info.filename, is_dir, info.st_size))
                        value = (path, sorted(rows, key=lambda r: (not r[1], r[0].lower())))
                    elif operation == 'mkdir':
                        sftp.mkdir(args[0]); value = args[0]
                    elif operation in ('read', 'geometry'):
                        value = read_remote(sftp, args[0], STRUCTURE_LIMIT if operation == 'geometry' else EDITOR_LIMIT)
                    elif operation == 'save_cube':
                        value = save_remote(sftp, *args, limit=STRUCTURE_LIMIT)
                    elif operation == 'save':
                        value = save_remote(sftp, *args)
                    elif operation in ('upload', 'download'):
                        value = transfer_items(sftp, operation, args[0], args[1], self.stopping,
                            lambda name, copied, total: self.transfer_progress.emit(token, name, copied, total))
                    elif operation == 'paste_files':
                        from file_clipboard import paste_files
                        value = paste_files(sftp, *args)
                    else: raise ValueError('Unknown SSH operation.')
                    self.result.emit(token, operation, value)
                except Exception as exc:
                    self.failed.emit(token, operation, str(exc))
        except Exception as exc:
            if not self.stopping.is_set(): message = str(exc)
        finally:
            self.auth_answers = None
            self.config.pop('password', None); self.config.pop('passphrase', None)
            if self.client is not None: self.client.close()
            if self.pump is not None: self.pump.join(timeout=2)
            self.disconnected.emit(message)
