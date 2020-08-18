"""
Functions for reorganizing a file's imports nicely
"""
import distutils.sysconfig as sysconfig
import os
import platform


def get_drive(drive_letter):
    if platform.system().lower() == "linux":
        return "/" + drive_letter.lower()
    elif platform.system().lower() == "windows":
        return drive_letter.upper() + ":\\"


d_drive = get_drive("d")


def convert_dir_to_local(dir_to_convert):
    dirs = []
    t = dir_to_convert.replace("\\", "/")
    while True:
        t, d = os.path.split(t)
        if len(t) == 0:
            if d[0] == ".":
                # relative directory
                dir_out = os.path.join(d, *dirs)
                break
            else:
                # absolute windows directory
                # get rid of semicolon and convert to local
                dir_out = os.path.join(get_drive(d.replace(":", "")), *dirs)
                break
        elif t == "/":
            # absolute linux directory
            dir_out = os.path.join(get_drive(d), *dirs)
            break

        dirs = [d] + dirs
    return dir_out

