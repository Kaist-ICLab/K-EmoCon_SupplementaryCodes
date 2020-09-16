import time
import pytz
from datetime import datetime


def ltnow(*args, tz='Asia/Seoul'):
    '''Returns a formatted string of local time now given a string for timezone.'''
    return pytz.utc.localize(datetime.utcnow()).astimezone(pytz.timezone(tz)).timetuple()
