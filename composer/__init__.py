from enum import Enum, unique

@unique
class ModelSaveFrequencyMode(Enum):
    '''
    Indicates the units of the model save frequency.

    :cvar EPOCH:
        The model save frequency is in epochs.
    :cvar GLOBAL_STEP:
        The model save frequency is in global steps.

    '''

    EPOCH = 'epoch'
    GLOBAL_STEP = 'step'

from composer.cli import cli