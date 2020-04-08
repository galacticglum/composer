# Composer
Composer is a deep learning enabled music generator module built in [Python](https://www.python.org/) and using [TensorFlow](https://www.tensorflow.org/). The goal of this project is to investigate how music can be generated and enhanced using machine learning techniques (such as deep learning via neural networks). It is a sandbox for generative MIDI neural-network models. What it is NOT is a fully-featured music generator.

## Installation
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. Composer uses [conda](https://docs.conda.io/en/latest/), an open-source package management system, for virtual environment management. 

**Prerequisites:**
* A functioning conda installation (We recommend [Miniconda](https://docs.conda.io/en/latest/miniconda.html), a minimal installer for conda).

**Note:** The ``environment.yml`` file defines all the Python requirements—these are packages thatt will automatically be installed. This includes ``tensorflow-gpu`` which is what we recommend you use (as it will be tremendously faster than running on CPU); however, if you cannot run ``tensorflow-gpu``, remember to change to change this ``tensorflow``. Leave the version number the same.

### Automated Install
We provide an automated installation script for Composer located in the ``scripts`` directory. If you are running on Mac OS X or Linux, this script is called ``install_composer.sh``. If you are on Windows, it is called ``install_composer.bat``. 

From the root project directory (this is important!), open a new terminal window and run the respective script for your platform:

* **Windows**: ``scripts/install_composer.bat``.
* **Mac OS X or Linux**: ``source scripts/install_composer.sh``.

After the script is complete, open a terminal window and activate the ``composer`` environment by running ``conda activate composer``. You will have to activate this environment every time you open a new terminal window that will be used to run Composer.

### Manual Install
To manually install Composer, open a terminal window and navigate to the root project directory.

First clone create a conda environment from the ``environment.yml`` file:
```
conda env create -f environment.yml -q
```
Next, execute the setup command:
```
pip install -e .
``` 

You can now run the Composer CLI in the conda environment as you would any other command. Also, any changes made to the CLI will be automatically reflected (due to the ``-e`` argument when running the setup command).
