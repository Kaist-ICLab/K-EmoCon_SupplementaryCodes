import time
import pytz
import logging
from datetime import datetime


def ltnow(*args, tz='Asia/Seoul'):
    '''Returns a formatted string of local time now given a string for timezone.'''
    return pytz.utc.localize(datetime.utcnow()).astimezone(pytz.timezone(tz)).timetuple()


def init_logger(name='default'):
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
    formatter.converter = ltnow

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger
