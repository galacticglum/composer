'''
Click (CLI) related helper functionality.

'''

import re
import click
from enum import EnumMeta

class EnumType(click.Choice):
    '''
    Enum type for Click options/arguments.

    :ivar enum:
        The :class:`Type` object representing the type of the enum to display.
    :ivar casesensitive:
        Indicates whether the enum values are case-sensitive. Defaults to `True`.

    '''

    def __init__(self, enum, casesensitive=True):
        '''
        Initializes an instance of :class:`EnumType`.

        :param enum:
            The :class:`Type` object representing the type of the enum to display.
        :ivar casesensitive:
            Indicates whether the enum values are case-sensitive. Defaults to `True`.
        
        '''

        if isinstance(enum, tuple):
            choices = (i.name for i in enum)
        elif isinstance(enum, EnumMeta):
            choices = enum.__members__
            self._EnumType__enum = enum
        else:
            raise TypeError('`enum` must be `tuple` or `Enum`')

        if not casesensitive:
            choices = (i.lower() for i in choices)

        self.enum = enum
        self.casesensitive = casesensitive
        super().__init__(list(sorted(set(choices))))
    
    def convert(self, value, param, ctx):
        '''
        Converts this :class:`EnumType` to the underlying enum value.

        :param value:
            The input value of the choice.
        :param param:
            Parameters passed to the convert function.
        :param ctx:
            The click CLI context.

        '''

        def _compare_name(name, value, casesensitive):
            return name == value if casesensitive else name.lower() == value.lower()

        if not self.casesensitive:
            value = value.lower()

        value = super().convert(value, param, ctx)
        return next(i for i in self._EnumType__enum if _compare_name(i.name, value, self.casesensitive))

    def get_metavar(self, param):
        '''
        Gets metavar automatically from enum name.

        '''
        
        word = self.enum.__name__
        word = re.sub(r"([A-Z]+)([A-Z][a-z])", r'\1_\2', word)
        word = re.sub(r"([a-z\d])([A-Z])", r'\1_\2', word)
        word = word.replace("-", "_").lower().split("_")

        if word[-1] == 'enum':
            word.pop()

        return '_'.join(word).upper()