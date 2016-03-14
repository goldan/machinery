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
from tabulate import tabulate


def analyze_results():
    """Print experiments results stored in MongoDB."""
    options = docopt(__doc__)
    manager = StatsManager(100000, options["<db_name>"])
    headers = ['Classifier', 'scaling', 'accuracy']
    results = [(
        exp['classifier']['name'],
        exp['features']['scaling'],
        str(exp['results']['accuracy']*100) + '%') for exp in manager.iter_results()]
    print tabulate(results, headers)


if __name__ == "__main__":
    try:
        analyze_results()
    except KeyboardInterrupt:
        pass
    sys.exit(0)
