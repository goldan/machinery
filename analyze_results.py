# -*- coding: utf-8 -*-
u"""Print experiments results stored in MongoDB.

Usage:
    analyze_results.py <db_name>
    analyze_results.py diff <db_name> <id1> <id2>

Options:
 -h --help              Show this screen.
 --version              Show Version.
 <db_name>              Mongo database name where the results are stored.
 default action: print table with results.
 diff: print diff between two experiments with <id1> and <id2>.
"""
import sys

from deepdiff import DeepDiff
from docopt import docopt
from featureforge.experimentation.stats_manager import StatsManager
from prettytable import PrettyTable
from tabulate import tabulate
from machinery.common.utils import pretty_date


def get_experiments(db_name, ids=None, from_time=100000):
    """Get list of experiments from Mongo database, all or by ids.

    When search by ids, prefixes are allowed, so it finds
    all experiments which ids start from ids specified.

    Args:
        db_name: database name to connect to.
        ids: (optional) iterable of ids to filter results by.
        from_time: range of seconds (from now to the past),
            to filter experiments by.

    Returns:
        list of experiments dicts.
    """
    manager = StatsManager(from_time, db_name)
    return [exp for exp in manager.iter_results() if not ids or
            any(str(exp['_id']).startswith(pk) for pk in ids)]


def analyze_results(db_name, experiment_ids=None):
    """Print table with experiment results stored in MongoDB.

    Experiments are sorted by accuracy score reversed.

    Args:
        db_name: database name to connect to.
        experiment_ids: (optional) iterable of ids to filter results by.
    """
    experiments = get_experiments(db_name, experiment_ids)
    headers = ['ID', 'Classifier', 'Scaling', 'Grid size', 'Accuracy', 'Random state', 'Booked at']

    results = sorted([(
        exp['_id'],
        exp['classifier']['name'].split('.')[-1],
        exp['features']['scaling'],
        exp['results']['grid_size'],
        str(exp['results']['accuracy']*100) + '%',
        exp['random_state'],
        pretty_date(exp['booked_at']))
                      for exp in experiments], key=lambda row: row[4], reverse=True)
    print tabulate(results, headers)


def prepare_diff_key(key):
    """Extract deepest dict key from a DeepDiff key.

    >>> prepare_diff_key("['classifier']['config']['n_neighbors']")
    'n_neighbors'

    Args:
        key: DeepDiff (nested) key of changed dictionary value.

    Returns:
        the deepest key of the diff dictionary key.
    """
    return key.split('[')[-1].strip("]'")


def prepare_diff_value(key, value):
    """Convert DeepDiff value to a format suitable for output.

    Args:
        key: DeepDiff (nested) key of changed dictionary value.
        value: value of the key (old or new version).

    Returns:
        value converted for suitable representation, e.g. truncated.
    """
    if prepare_diff_key(key) == 'git_info':
        value = value[:70]
    return value


def diff_experiments(db_name, id1, id2):
    """Print diff between experiment results stored in MongoDB.

    First, print the analysis table of the two experiments.
    Then print the diff between all the config options and results.
    Note that id prefixes are allowed, so id search is done by prefixes.

    Args:
        db_name: database name to connect to.
        id1: id of the first experiment to compare.
        id2: id of the second experiment to compare.

    Raises:
        Exception if more or less than 2 experiments were found by ids.
    """
    analyze_results(db_name, (id1, id2))
    experiments = get_experiments(db_name, (id1, id2))
    if len(experiments) < 2:
        raise Exception("No experiments with given ids")
    if len(experiments) > 2:
        raise Exception("Too many experiments with given ids")
    diff = DeepDiff(*experiments)
    table = PrettyTable()
    table.field_names = ["Parameter", id1, id2]
    # we make a set to eliminate duplicates in diff, e.g. 'random_state' occurs twice in keys
    # sort by key name
    rows = sorted(list(set([(
        prepare_diff_key(key),
        prepare_diff_value(key, values['oldvalue']),
        prepare_diff_value(key, values['newvalue'])) for key, values in
                            diff['values_changed'].items()])), key=lambda row: row[0])
    for row in rows:
        table.add_row(row)
    table.align = 'l'
    print table


def main():
    """Select the action from command line and execute the function."""
    options = docopt(__doc__)
    if options["diff"]:
        diff_experiments(options["<db_name>"], options["<id1>"], options["<id2>"])
    else:
        analyze_results(options["<db_name>"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    sys.exit(0)
