import os
import subprocess
import psutil

BOX_DATA_DIR = os.path.join(
    'C:\\',
    'Users',
    'cartemic',
    'Box'
)
BOX_EXECUTABLE = 'Box.exe'
BOX_EXECUTABLE_DIR = os.path.join(
    'C:\\',
    'Program Files',
    'Box',
    'Box'
)


def please():
    box_path = os.path.join(
        BOX_EXECUTABLE_DIR,
        BOX_EXECUTABLE
    )
    # make sure it isn't already open
    processes = set()
    for pid in psutil.pids():
        try:
            processes.add(psutil.Process(pid).name())
        except psutil.NoSuchProcess:
            pass

    if processes.intersection({BOX_EXECUTABLE}):
        print('\033[94mBox was already running :)\033[0m')
    else:
        subprocess.Popen(box_path)
        print('\033[92mBox is now running :)\033[0m')


if __name__ == '__main__':
    please()
