"""Exclusive, streaming file copy/move between local and SFTP locations."""
import os
import posixpath
import shutil
import stat
from pathlib import Path


def paste_files(sftp, sources, source_remote, destination, target_remote, cut=False):
    completed, errors = [], []
    for source in sources:
        name = posixpath.basename(source) if source_remote else Path(source).name
        if not target_remote and (Path(name).name != name or Path(name).is_absolute() or any(c in name for c in '\\:<>"|?*')):
            errors.append(f'{source}: filename cannot be represented safely on this local filesystem.')
            continue
        target = posixpath.join(destination, name) if target_remote else str(Path(destination) / name)
        created = False
        try:
            before = sftp.lstat(source) if source_remote else os.lstat(source)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError('Select regular files for copy/cut; use Upload/Download for folders.')
            if source_remote == target_remote:
                same = (sftp.normalize(source) == posixpath.join(sftp.normalize(destination), posixpath.basename(source))
                        if source_remote else Path(source).resolve() == Path(target).resolve())
                if same: raise ValueError('Source and destination are the same file.')
            if cut and source_remote and target_remote:
                try: sftp.lstat(target)
                except FileNotFoundError: pass
                else: raise FileExistsError('Destination already exists: ' + target)
                sftp.rename(source, target)  # Standard SFTP rename refuses existing targets.
            else:
                reader = sftp.open(source, 'rb') if source_remote else open(source, 'rb')
                with reader:
                    writer = sftp.open(target, 'wx') if target_remote else open(target, 'xb')
                    with writer:
                        created = True
                        shutil.copyfileobj(reader, writer, 1024 * 1024)
                after = sftp.stat(source) if source_remote else os.stat(source)
                copied = sftp.stat(target) if target_remote else os.stat(target)
                if (before.st_size, before.st_mtime) != (after.st_size, after.st_mtime) or copied.st_size != after.st_size:
                    raise OSError('Source changed during transfer; source retained.')
                if cut:
                    if source_remote: sftp.remove(source)
                    else: os.unlink(source)
            completed.append(source)
        except Exception as exc:
            # A completed destination is retained if source removal fails, avoiding data loss.
            errors.append(f'{source}: {exc}' + (' (destination may contain a copy)' if created else ''))
    return {'completed': completed, 'errors': errors}
