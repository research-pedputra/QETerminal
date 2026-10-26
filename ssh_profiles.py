"""Named SSH connection settings, with credentials protected by Windows DPAPI."""
import json
import base64
import ctypes
import os
from ctypes import wintypes
from PySide6.QtCore import QSettings


def _protect(value):
    if not value: return ''
    if os.name != 'nt': raise OSError('Saving SSH passwords requires Windows credential protection.')
    class Blob(ctypes.Structure):
        _fields_ = [('cbData', wintypes.DWORD), ('pbData', ctypes.POINTER(ctypes.c_byte))]
    raw = value.encode('utf-8'); source_buf = ctypes.create_string_buffer(raw)
    source = Blob(len(raw), ctypes.cast(source_buf, ctypes.POINTER(ctypes.c_byte)))
    output = Blob()
    if not ctypes.windll.crypt32.CryptProtectData(ctypes.byref(source), 'QEterminal SSH', None, None, None, 0, ctypes.byref(output)):
        raise OSError('Windows could not protect the SSH credential.')
    try: return base64.b64encode(ctypes.string_at(output.pbData, output.cbData)).decode('ascii')
    finally: ctypes.windll.kernel32.LocalFree(output.pbData)


def _unprotect(value):
    if not value: return ''
    if os.name != 'nt': return ''
    class Blob(ctypes.Structure):
        _fields_ = [('cbData', wintypes.DWORD), ('pbData', ctypes.POINTER(ctypes.c_byte))]
    raw = base64.b64decode(value); input_buf = ctypes.create_string_buffer(raw)
    source = Blob(len(raw), ctypes.cast(input_buf, ctypes.POINTER(ctypes.c_byte))); output = Blob(); description = wintypes.LPWSTR()
    if not ctypes.windll.crypt32.CryptUnprotectData(ctypes.byref(source), ctypes.byref(description), None, None, None, 0, ctypes.byref(output)):
        return ''
    try: return ctypes.string_at(output.pbData, output.cbData).decode('utf-8')
    finally:
        ctypes.windll.kernel32.LocalFree(output.pbData)
        if description: ctypes.windll.kernel32.LocalFree(description)


class SSHProfiles:
    fields = ('host', 'user', 'port', 'key', 'directory', 'password', 'passphrase')

    def __init__(self, settings=None):
        self.settings = settings if settings is not None else QSettings('QEterminal', 'SSH')

    def all(self):
        try:
            data = json.loads(self.settings.value('profiles', '{}'))
            result = {}
            for name, config in data.items():
                if not isinstance(config, dict): continue
                cleaned = {key: value for key, value in config.items() if key in self.fields and key not in ('password', 'passphrase')}
                for key in ('password', 'passphrase'): cleaned[key] = _unprotect(config.get(key, ''))
                result[name] = cleaned
            return result
        except (ValueError, TypeError, AttributeError): return {}

    def last(self):
        return self.settings.value('last_profile', '')

    def save(self, name, config):
        profiles = self.all()
        profiles[name] = {key: config[key] for key in self.fields if key in config and key not in ('password', 'passphrase')}
        profiles[name]['password'] = _protect(config.get('password', ''))
        profiles[name]['passphrase'] = _protect(config.get('passphrase', ''))
        self.settings.setValue('profiles', json.dumps(profiles))
        self.settings.setValue('last_profile', name)
        self.settings.sync()
        if self.settings.status() != QSettings.Status.NoError:
            raise OSError('Could not save SSH profile settings.')

    def delete(self, name):
        profiles = self.all(); profiles.pop(name, None)
        self.settings.setValue('profiles', json.dumps(profiles))
        if self.last() == name: self.settings.remove('last_profile')
        self.settings.sync()
