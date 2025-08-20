====================
Installation and use
====================

The functions for detonation data analysis and simulation contained within this repository are in the ``funcs`` module. This is intended to be a work in progress, as new functionality may need to be added on the fly. Therefore, the module is installed in development mode.

To install the module:

1. Download the repository from github

2. In a console, navigate to the top level module directory

3. Activate your desired conda environment, e.g. ``$ source activate research``

   a. If you don't already have a conda environment for this, you may find it convenient to install from the existing yaml file ``$ conda env create -f research_pls.yaml``. The environment name defaults to ``research``, as indicated in the yaml file. Once the environment is installed, activate it before proceeding to step 4.

4. Install the module using ``$ conda develop .``

Now you should be able to ``import funcs`` as well as the sub-modules from within your research environment. Neato.


