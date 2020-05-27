# Composer
*Not to be confused with the PHP dependency management package.*

Composer is a deep learning enabled music generator module built in [Python](https://www.python.org/) and using [TensorFlow](https://www.tensorflow.org/). The goal of this project is to investigate how music can be generated and enhanced using machine learning techniques (such as deep learning via neural networks). It is a sandbox for generative MIDI neural-network models. What it is NOT is a fully-featured music generator.

## Results
 <table style="width:100%">
  <tr>
    <td><img src="https://puu.sh/FPtrz/2c19e95e51.jpg" width="128" />
</td>
    <td>
     The Transformer model was used to create <a href="https://soundcloud.com/galacticglum/sets/ramblings-of-a-transformer">the ramblings of a transformer</a>.
    </td>
  </tr>
</table> 

## Installation
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. Composer uses [conda](https://docs.conda.io/en/latest/), an open-source package management system, for virtual environment management. 

**Prerequisites:**
* A functioning conda installation (We recommend [Miniconda](https://docs.conda.io/en/latest/miniconda.html), a minimal installer for conda).

**Note:** The ``environment.yml`` file defines all the Python requirements—these are packages that will automatically be installed. This includes ``tensorflow-gpu`` which is what we recommend you use (as it will be tremendously faster than running on CPU); however, if you cannot run ``tensorflow-gpu``, remember to change to change this ``tensorflow``. Leave the version number the same.

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

## Command line tools usage
See docstring for basic information about each command or refer to this guide for usage information.

**composer**

The root command for all functions related to this project.

Usage:
```
composer [OPTIONS] COMMAND [ARGS]...
```

Options:
* ``-v``, ``--verbosity``: either ``CRITICAL``, ``ERROR``, ``WARNING``, ``INFO``, or ``DEBUG``; indicates the logger verbosity level.
* ``--seed``: an integer indicating the seed of the random engine.
* ``--help``: show help information.

**composer evaluate**

Evaluate the specified model.

Usage:
```
composer evaluate [OPTIONS] MODEL_TYPE DATASET_PATH RESTOREDIR
```

Options:
* ``--use-generator / --no-use-generator``: indicates whether the dataset should be loaded in chunks during processing (rather than into memory all at once). Defaults to False.
* ``--max-files``: the maximum number of files to load. Defaults to None, which means that ALL files will be loaded.
* ``--help``: show help instructions.

**composer export-dataset**

Exports a processed dataset input pipeline as a TFRecord file for fast loading times when training.

Note that the ``PREPROCESSED_PATH`` argument refers to the path containing the preprocessed '.data' files. For example, this could be "dataset_parent/train", "dataset_parent/test", or simply "path/to/files".

Usage:
```
composer export-dataset [OPTIONS] MODEL_TYPE PREPROCESSED_PATH OUTPUT_PATH
```

Options:
* ``-c``, ``--config``: the path to the model configuration file. If unspecified, uses the default config for the model.
* ``--use-generator / --no-use-generator``: indicates whether the dataset should be loaded in chunks during processing (rather than into memory all at once). Defaults to False.
* ``--max-files``: the maximum number of files to load. Defaults to None, which means that ALL files will be loaded.
* ``--help``: show help instructions.

**composer generate**

Generate a MIDI file.

Usage:
```
composer generate [OPTIONS] MODEL_TYPE RESTOREDIR OUTPUT_FILEPATH
```

Options:
* ``-p``, ``--prompt``: The path of the MIDI file to prompt the network with. Defaults to None, meaning a random prompt will be created.
* ``--prompt-length``: number of events to take from the start of the prompt. Defaults to 10.
* ``-l``, ``--length``: The length of the generated event sequence. Defaults to 1024.
* ``--temperature``: a float which dictates how random the result is. Low temperature yields more predictable output. On the other hand, high temperature yields very random ("surprising") outputs. Defaults to 1.0.
* ``--help``: show help instructions.

**composer make-config**

Creates a configuration file from the default configuration.

Usage:
```
composer make-config [OPTIONS] FILEPATH
```

Options:
* ``--help``: show help instructions.

**composer preprocess**

Preprocesses a raw dataset so that it can be used by specified model type.

Usage:
```
composer preprocess [OPTIONS] MODEL_TYPE DATASET_PATH OUTPUT_DIRECTORY
```

Options:
* ``-w``, ``--num-workers``: the number of worker threads to spawn. Defaults to 16.
* ``-c``, ``--config``: The path to the model configuration file. If unspecified, uses the default config for the model.
* ``-spe``, ``--sustain-period-encode-mode``: a member of the ``composer.dataset.sequence.NoteSequence.SustainPeriodEncodeMode`` enum.  The way in which sustain periods should be encoded. Defaults to EXTEND. Refer to NoteSequence.to_event_sequence documentation for more details on this parameter.
* ``--transform / --no-transform``: indicates whether the dataset should be transformed. If true, a percentage of the dataset is duplicated and pitch shifted and/or time-stretched. Defaults to True. Note that transforming a single sample produces many new samples: one for each pitch in the pitch shift range, and a timestretched one (uniformly sampled from the time stretch range).
* ``--transform-percent``:  the ratio of the dataset (from 0 to 1) that should be transformed. Defaults to 100% of the dataset.
* ``--split / --no-split``: indicates whether the dataset should be split into train and test sets. Defaults to True.
* `` --test-percent``: the percentage of the dataset that is allocated to testing. Defaults to 30%.
* ``--metadata / --no-metadata``: indicates whether to output metadata. Defaults to True.
* ``--help``: show help instructions.

**composer summary**

Prints a summary of the model.

Usage:
```
composer summary [OPTIONS] MODEL_TYPE
```

Options:
* ``-c``, ``--config``: the path to the model configuration file. If unspecified, uses the default config for the model.
* ``--help``: show help instructions.

**composer synthesize**

Synthesize the specified MIDI file using a soundfont.

Usage:
```
composer synthesize [OPTIONS] MIDI_FILEPATH
```

Options:
* ``--sf-path``: the filepath of the soundfont to use. If not specified, uses the default soundfont.
* ``--sf-save-path``: the path to save the default soundfont to.
* ``--chunk-size``: the number of bytes to download in a single chunk. Defaults to 32768.
* ``--help``: show help instructions.

**composer train**

Trains the specified model.

Usage:
```
composer train [OPTIONS] MODEL_TYPE DATASET_PATH
 ```
 
Options:
* `` --logdir``: the root log directory. Defaults to './output/logdir'.
* `` --restoredir``: the directory of the model to continue training.
* ``-c``, ``--config``: the path to the model configuration file. If unspecified, uses the default config for the model. If a restoredir is specified, the configuration file in the restoredir is used instead (and this value is ignored).
 * ``-e``, ``--epochs``: the number of epochs to train for. Defaults to 10.
 * ``--use-generator / --no-use-generator``: indicates whether the dataset should be oaded in chunks during processing (rather than into memory all at once). Defaults to False.
 * ``--max-files``: the maximum number of files to load. Defaults to None, which means that ALL files will be loaded.
 * ``--save-freq-mode``: a member of ``composer.models.ModelSaveFrequencyMode``. The units of the save frequency. Defaults to ``GLOBAL_STEP``.
 * ``--save-freq``: the frequency at which to save the model (in the units specified by the save frequency mode). Defaults to every 500 global steps.
* ``--max-checkpoints``: the maximum number of checkpoints to keep. Defaults to 3.
* ``--show-progress-bar / --no-show-progress-bar``: indicates whether a progress bar will be shown to indicate epoch status. Defaults to True. 
 * ``--help``: show help instructions.

**composer visualize-training**

Visualize how the model will train. This displays the input and expected
output (features and labels) for each step given the dataset.

Usage:
```
composer visualize-training [OPTIONS] MODEL_TYPE DATASET_PATH
```

Options:
*  ``-c``, ``--config``: the path to the model configuration file. If unspecified, uses the default config for the model.
*  ``--steps``: the number of steps to visualize. Defaults to 5.
* ``--decode-events / -no-decode--events``: indicates whether the events should be decoded or displayed as their raw values (i.e. as a one-hot vector or integer id).
 * ``--help``: show help instructions.

## References
1. Transformer-decoder block implementation based on the open-source [OpenAI GPT-2 code](https://github.com/openai/gpt-2/blob/master/src/model.py).
2. Default "Yamaha-C5-Salamander-JNv5.1" sounddfont from https://sites.google.com/site/soundfonts4u/.
