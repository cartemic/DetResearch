import shutil
from pathlib import Path

if __name__ == "__main__":
    import cantera as ct

    ct_data_dir = Path(ct.get_data_directories()[-1])
    my_mech_dir = Path(__file__).parents[1] / "mechanisms"
    for file in my_mech_dir.rglob("*.yaml"):
        shutil.copy(file, ct_data_dir / file.name)
        print(f"Copied {file.name} to {ct_data_dir}")
