# -*- coding: utf-8 -*-
u"""Print experiments results stored in MongoDB.

Usage:
    analyze_results.py list <db_name> [--sort=(score|name|state|date) -r]
    analyze_results.py diff <db_name> <id1> <id2>
    analyze_results.py show <db_name> <experiment_id> [<key>]
    analyze_results.py features <features.csv> <output_file>

Commands:
    list: print table with experiments.
    diff: print diff between two experiments with <id1> and <id2>.
    show: print experiment with <experiment_id> dict <key> value.
    features: print features value counts to <output_file>.

Options:
    -h --help                       Show this screen.
    --Version                       Show Version.
    <db_name>                       Mongo database name where the results are stored.
    --sort=(score|name|state|date)  Sort table by respective column.
    -r                              Reverse sort order
                                    (by default, some sort values (score) have ascending order,
                                    some (name) have descending)
"""
import sys
from collections import namedtuple

import pandas
from deepdiff import DeepDiff
from docopt import docopt
from featureforge.experimentation.stats_manager import StatsManager
from prettytable import PrettyTable
from tabulate import tabulate
from machinery.common.utils import pretty_date, parse_classification_report


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


def analyze_results(db_name, experiment_ids=None, sort_by='score', reverse=False):
    """Print table with experiment results stored in MongoDB.

    Experiments are sorted by accuracy score reversed.

    Args:
        db_name: database name to connect to.
        experiment_ids: (optional) iterable of ids to filter results by.
        sort_by: sort by column name.
        reverse: if True, reverse default sort order.
    """
    if not sort_by:  # it can be None if not specified in CLI
        sort_by = 'score'
    experiments = get_experiments(db_name, experiment_ids)
    headers = ['ID', 'Classifier', 'Scaling', 'Grid size', 'Accuracy', 'Random state', 'Booked at']

    sort_key = {  # sort key row index, default reverse value
        'score': (4, True),
        'name': (1, False),
        'state': (5, False),
        'date': (6, True)
    }
    Row = namedtuple("TableRow", "id, name, scaling, grid_size, score, random_state, booked_at")
    rows = [Row(exp['_id'],
                exp['classifier']['name'].split('.')[-1],
                exp['features']['scaling'],
                exp['results']['grid_size'],
                exp['results']['accuracy'],
                exp['random_state'],
                exp['booked_at'])
            for exp in experiments]
    # first sort, then replace with human-readable values
    do_reverse = sort_key[sort_by][1]  # default reverse value
    if reverse:  # if reverse is True, reverse that default reverse value
        do_reverse = not do_reverse
    rows.sort(key=lambda row: (row[sort_key[sort_by][0]], row[4]), reverse=do_reverse)
    rows = [row._replace(score=str(row.score*100)+'%', booked_at=pretty_date(row.booked_at))
            for row in rows]
    print tabulate(rows, headers)


def update_experiment(manager, experiment_id, key, value):
    """Update experiment record in Mongo database.

    Args:
        manager: StatsManager connected to the database.
        experiment_id: id of the experiment to update.
        key: nested key in the record to update, in dot-notation.
        value: value to set for the key.
    """
    query = {u'_id': experiment_id}
    update = {'$set': {key: value}}
    manager.data.find_and_modify(query, update)


def recreate_metrics(db_name):
    """Recreate and store metrics for experiments from classification reports.

    For every experiment in the database, take its classification report,
    parse it and store individual metrics back in the database.
    Used for experiments for which metrics like precision/recall/f1-score and support
    was not stored, but the classification reports were.
    Note that values are always updated, even if they are actually in the database.

    Args:
        db_name: database name to connect to.
    """
    experiments = get_experiments(db_name)
    manager = StatsManager(100000, db_name)
    for experiment in experiments:
        eid = experiment['_id']
        print eid
        report = parse_classification_report(experiment['results']['report'])
        precisions = [scores[0] for scores in report.values()[:-1]]
        recalls = [scores[1] for scores in report.values()[:-1]]
        fscores = [scores[2] for scores in report.values()[:-1]]
        supports = [scores[3] for scores in report.values()[:-1]]
        update_experiment(manager, eid, 'results.precisions', precisions)
        update_experiment(manager, eid, 'results.recalls', recalls)
        update_experiment(manager, eid, 'results.f1-scores', fscores)
        update_experiment(manager, eid, 'results.supports', supports)
        precision, recall, fscore, support = report.values()[-1]
        update_experiment(manager, eid, 'results.precision', precision)
        update_experiment(manager, eid, 'results.recall', recall)
        update_experiment(manager, eid, 'results.f1-score', fscore)
        update_experiment(manager, eid, 'results.support', support)


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
    analyze_results(db_name, experiment_ids=(id1, id2))
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
                            diff['values_changed'].items() +
                            diff.get('type_changes', {}).items()])), key=lambda row: row[0])
    for row in rows:
        table.add_row(row)
    table.align = 'l'
    print table


def analyze_features(features_filename, output_filename):
    """Print features value counts to a file.

    Args:
        features_filename: name of file containing feature values.
        output_filename: name of file to output result to.
    """
    with open(output_filename, 'w') as fout:
        data = pandas.read_csv(features_filename)
        for name in data.keys():
            fout.write("\n\n%s:\n" % name)
            fout.write(unicode(data[name].value_counts()).encode('utf-8'))


def truncate_lines(value, limit=100):
    """Truncate each line in value by <limit> chars.

    It's useful before printing long lines in console, not to mess up the output,
    if one line starts to take multiple lines.
    But note that if value is e.g. a confusion matrix, it takes multiple lines,
    but all of them are short. So we just make sure that every line in output
    is no longer than limit.

    Args:
        value: value to truncate.
        limit: number of chars to truncate each line in value by.

    Returns:
        truncated value, with newline characters preserved.
    """
    return "\n".join(line[:limit] for line in str(value).split("\n"))


def print_value(value):
    """Smart tabular print of values, which can be dictionaries.

    If value is not a dictionary, just print it in a table.
    If it is a dictionary, print every key in it on a separate line of a table,
    printint nested keys too.

    Args:
        value: value to print.
    """
    def _print_value(table, value, prefix=""):
        """Add row of value to a table, or call the function recursively.

        If value is a dict, call the function for every dict key.

        Args:
            table: PrettyTable instance to output values.
            value: value to print.
            prefix: key of the value, to print in the first column of the table.
        """
        if isinstance(value, dict):
            for key, subvalue in value.items():
                _print_value(table, subvalue, (prefix+"." if prefix else "")+key)
        else:
            table.add_row([prefix, truncate_lines(value)])
    table = PrettyTable()
    table.field_names = ["Key", "Value"]
    _print_value(table, value)
    table.align = 'l'
    print table


def show_experiment(db_name, experiment_id, key):
    """Print experiment dict values by key (dotted notation).

    Args:
        db_name: database name to connect to.
        experiment_id: id of experiment to print values for.
        key: key of experiment dict in database to print contains of.
            Key can be nested, subkeys are separated by ".".
    """
    key = key or "results"  # default key
    experiment = get_experiments(db_name, [experiment_id])[0]
    value = experiment
    for subkey in key.split("."):
        value = value[subkey]
    print_value(value)


def main():
    """Select the action from command line and execute the function."""
    options = docopt(__doc__)
    if options["list"]:
        analyze_results(options["<db_name>"], sort_by=options["--sort"],
                        reverse=options['-r'])
    elif options["diff"]:
        diff_experiments(options["<db_name>"], options["<id1>"], options["<id2>"])
    elif options["features"]:
        analyze_features(options["<features.csv>"], options["<output_file>"])
    elif options["show"]:
        show_experiment(options["<db_name>"], options["<experiment_id>"], options["<key>"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    sys.exit(0)
