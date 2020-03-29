# Composer
Composer is a deep learning enabled music generator module built in [Python](https://www.python.org/) and using [TensorFlow](https://www.tensorflow.org/). The goal of this project is to investigate how music can be generated and enhanced using machine learning techniques (such as deep learning via neural networks). It is a sandbox for generative MIDI neural-network models. What it is NOT is a fully-featured music generator.

## Installation
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. Composer uses [https://docs.conda.io/en/latest/](https://docs.conda.io/en/latest/), an open-source package management system, for virtual environment management. 

### Mac OS X or Linux
Run the ``install_composer.sh`` script located in the ``scripts`` folder. This will automatically create and setup the conda environment. After the script is complete, open a new terminal window and activate the ``composer`` environment by running ``source activate composer``.

### Windows
**Prerequisites:**
* A functioning conda installation (We recommend [Miniconda](https://docs.conda.io/en/latest/miniconda.html), a minimal installer for conda).

Run the ``install_composer.bat`` file located in the ``scripts`` folder. This will automatically create and setup the conda environment. After the script is complete, open a conda prompt (i.e. a command prompt window with conda available) and activate the ``composer`` environment by running ``conda activate composer``. You will have to activate this environment every time you open a new command propt window that will be used to run Composer.
