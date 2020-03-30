import re
import copy
import logging
import colorama

def colourize_string(string, colour):
    return '{colour_begin}{string}{colour_end}'.format(
        colour_begin=colour,
        string=string,
        colour_end=colorama.Style.RESET_ALL
    )

_LOG_COLORS = {
    logging.FATAL: colorama.Fore.LIGHTRED_EX,
    logging.ERROR: colorama.Fore.RED,
    logging.WARNING: colorama.Fore.YELLOW,
    logging.DEBUG: colorama.Fore.LIGHTWHITE_EX
}

_LOG_LEVEL_FORMATS = {
    logging.INFO: '%(message)s'
}

_LOGGER_FORMAT = '%(levelname)s: %(message)s'

def init():
    class ColourizedLoggerFormat(logging.Formatter):
        def format(self, record, *args, **kwargs):
            # if the corresponding logger has children, they may receive modified
            # record, so we want to keep it intact
            new_record = copy.copy(record)
            if new_record.levelno in _LOG_COLORS:
                # we want levelname to be in different color, so let's modify it
                new_record.levelname = "{color_begin}{level}{color_end}".format(
                    level=new_record.levelname,
                    color_begin=_LOG_COLORS[new_record.levelno],
                    color_end=colorama.Style.RESET_ALL,
                )

            original_format = self._style._fmt
            self._style._fmt = _LOG_LEVEL_FORMATS.get(record.levelno, original_format)

            # now we can let standart formatting take care of the rest
            result = super(ColourizedLoggerFormat, self).format(new_record, *args, **kwargs)

            self._style._fmt = original_format
            return result

    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(ColourizedLoggerFormat(_LOGGER_FORMAT))
    logger.addHandler(handler)