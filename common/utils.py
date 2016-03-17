"""Common utility functions and constants used throughout the project."""
import collections
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


def parse_classification_report(clfreport):
    """
    Parse a sklearn classification report into a dict keyed by class name
    and containing a tuple (precision, recall, fscore, support) for each class.
    Taken from https://gist.github.com/julienr/6b9b9a03bd8224db7b4f
    """
    lines = clfreport.split('\n')
    # Remove empty lines
    lines = filter(lambda l: not len(l.strip()) == 0, lines)

    # Starts with a header, then score for each class and finally an average
    header = lines[0]
    cls_lines = lines[1:-1]
    avg_line = lines[-1]

    assert header.split() == ['precision', 'recall', 'f1-score', 'support']
    assert avg_line.split()[0] == 'avg'

    # We cannot simply use split because class names can have spaces. So instead
    # figure the width of the class field by looking at the indentation of the
    # precision header
    cls_field_width = len(header) - len(header.lstrip())
    # Now, collect all the class names and score in a dict
    def parse_line(l):
        """Parse a line of classification_report"""
        cls_name = l[:cls_field_width].strip()
        precision, recall, fscore, support = l[cls_field_width:].split()
        precision = float(precision)
        recall = float(recall)
        fscore = float(fscore)
        support = int(support)
        return (cls_name, precision, recall, fscore, support)

    data = collections.OrderedDict()
    for l in cls_lines:
        ret = parse_line(l)
        cls_name = ret[0]
        scores = ret[1:]
        data[cls_name] = scores

    # average
    data['avg'] = parse_line(avg_line)[1:]

    return data