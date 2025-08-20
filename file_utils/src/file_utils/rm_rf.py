from pathlib import Path


def rm_rf(path: Path) -> None:
    for thing in path.iterdir():
        if thing.is_file():
            thing.unlink()
        else:
            rm_rf(thing)
            thing.rmdir()
