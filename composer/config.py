'''
A configuration file manager.

'''

import yaml

class Dotdict(dict):
    '''
    A dictionary that supports dot notation as well as dictionary access notation.

    '''

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, data=None):
        '''
        Initialize an instance of :class:`Dotdict`.

        '''

        data = dict() if not data else data
        for key, value in data.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)

            self[key] = value    

class ConfigInstance(Dotdict):
    '''
    A loaded configuration.

    :ivar filepath:
        The path of the original config file. 
    
    '''

    def __init__(self, filepath, data):
        '''
        Initialize an instance of :class:`ConfigInstance`.

        :param filepath:
            The path of the original config file.
        :param data:
            A ``dict`` containing the configuration values.

        '''

        self.filepath = filepath
        super().__init__(data)

def get(filepath):
    '''
    Gets a configuration.

    :param filepath:
        The path to the configuration file.
    :returns:
        An instance of :class:`ConfigInstance`.

    '''

    with open(filepath) as file:
        data = yaml.safe_load_all(file)
        data_dict = dict()
        for x in data:
            for k, v in x.items():
                data_dict[k] = v
        
        return ConfigInstance(filepath, data_dict)