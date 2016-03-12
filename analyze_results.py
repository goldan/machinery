# -*- coding: utf-8 -*-
u"""Print experiments results stored in MongoDB.

Usage:
    analyze_results.py <db_name>

Options:
 -h --help              Show this screen.
 --version              Show Version.
 <db_name>              Mongo database name where the results are stored.
"""
import sys

from docopt import docopt
from featureforge.experimentation.stats_manager import StatsManager


def analyze_results():
    """Print experiments results stored in MongoDB."""
    options = docopt(__doc__)
    manager = StatsManager(100000, options["<db_name>"])
    for experiment in manager.iter_results():
        print experiment['classifier']['name']
        print experiment['results']['accuracy']
        print experiment['results']['report']
        print experiment['results']['confusion_matrix']


if __name__ == "__main__":
    try:
        analyze_results()
    except KeyboardInterrupt:
        pass
    sys.exit(0)
