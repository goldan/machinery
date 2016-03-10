"""Common utility functions and constants used throughout the project."""
import csv

CSV_OPTIONS = {'delimiter': ',', 'quotechar': '"', 'quoting': csv.QUOTE_MINIMAL}


def writerow(writer, row):
    """Write a row using csv writer.

    Takes care of encoding.

    Args:
        row: list of values to write.
    """
    writer.writerow([unicode(value).encode('utf8') for value in row])
