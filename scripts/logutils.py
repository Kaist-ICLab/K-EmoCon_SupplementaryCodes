import pytz
import logging
from datetime import datetime


def init_logger(tz, name='default'):
    assert tz in pytz.all_timezones_set, f'{tz} is not a valid pytz timezone'

    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
    )

    # set converter for time format
    def ltnow(*args, tz=tz):
        '''Returns a formatted string of local time now given a string for timezone.'''
        return pytz.utc.localize(datetime.utcnow()).astimezone(pytz.timezone(tz)).timetuple()

    formatter.converter = ltnow

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger
