from setuptools import setup, find_packages

setup(
    name='funcs',
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
        "PyQt5"
    ])
