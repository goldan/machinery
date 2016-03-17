"""Common utility functions and constants used throughout the project."""
import csv
from datetime import datetime


CSV_OPTIONS = {'delimiter': ',', 'quotechar': '"', 'quoting': csv.QUOTE_MINIMAL}


def writerow(writer, row):
    """Write a row using csv writer.

    Takes care of encoding.

    Args:
        row: list of values to write.
    """
    writer.writerow([unicode(value).encode('utf8') for value in row])


def roundto(value, digits=3):
    """Round value to specified number of digits.

    Args:
        value: value to round.
        digits: number of digits to round to.

    Returns:
        rounded value.
    """
    return round(value, digits)


def pretty_date(time=False):
    """
    Get pretty representation of a date/time.

    Get a datetime object or a int() Epoch timestamp and return
    a pretty string like 'an hour ago', 'Yesterday', '3 months ago',
    'just now', etc.
    Copied from http://stackoverflow.com/a/1551394/304209.

    Args:
        time: datetime object or epoch timestamp.

    Returns:
        date in human format, like 'an hour ago'.
    """
    now = datetime.now()
    if isinstance(time, int):
        diff = now - datetime.fromtimestamp(time)
    elif isinstance(time, datetime):
        diff = now - time
    elif not time:
        diff = now - now
    second_diff = diff.seconds
    day_diff = diff.days

    if day_diff < 0:
        return ''

    if day_diff == 0:
        if second_diff < 10:
            return "just now"
        if second_diff < 60:
            return str(second_diff) + " seconds ago"
        if second_diff < 120:
            return "a minute ago"
        if second_diff < 3600:
            return str(second_diff / 60) + " minutes ago"
        if second_diff < 7200:
            return "an hour ago"
        if second_diff < 86400:
            return str(second_diff / 3600) + " hours ago"
    if day_diff == 1:
        return "Yesterday"
    if day_diff < 7:
        return str(day_diff) + " days ago"
    if day_diff < 31:
        return str(day_diff / 7) + " weeks ago"
    if day_diff < 365:
        return str(day_diff / 30) + " months ago"
    return str(day_diff / 365) + " years ago"


def flatten_list(lst):
    """Transform a 1-level nested list into a flat one.

    Args:
        lst: a nested list to transform.

    Returns:
        a flat list.

    >>> flatten_list([[1, 2],[3, 4]])
    >>> [1, 2, 3, 4]
    """
    return [item for sublist in lst for item in sublist]
