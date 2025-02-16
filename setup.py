from setuptools import find_packages, setup

setup(
    name="funcs",
    package=find_packages(),
    install_requires=[
        "pandas",
        "nptdms",
        "uncertainties",
        "numpy",
        "scipy",
        "pint",
        "matplotlib",
        "seaborn",
        "scikit-image",
        "tables",
        "PyQt5",
    ],
)
