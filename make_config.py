# -*- coding: utf-8 -*-
u"""Make json config file with experiment configuration.

Usage:
    make_config.py <x_train.csv> <y_train.csv> <x_test.csv> <y_test.csv> <config.json> [--classifier=<classifier> -n --random-state=<random_state> --grid-scoring=<grid_scoring> --scaling=<scaling>]

Arguments:
 <x_train.csv>          Name of csv file with feature values for the training set.
 <y_train.csv>          Name of csv file with target classes values for the training set.
 <x_test.csv>           Name of csv file with feature values for the test set.
 <y_test.csv>           Name of csv file with target classes values for the test set.
 <config.json>          Name of json file to write configuration to.

Options:
 -h --help                      Show this screen.
 --version                      Show Version.
 --classifier=<classifier>      Classifier name to use.
 -n                             Disable grid hyperparameter search.
 --random-state=<random_state>  Use preselected random state.
 --grid-scoring=<grid_scoring>  Grid hyperparameter set evaluation method. Examples: f1_weighted, accuracy, cohen_kappa.
 --scaling=<scaling>            Use specific value for feature scaling (instead of setting both True and False).
"""
import json
import random
import sys
from collections import OrderedDict
from hashlib import sha256
from itertools import product
from numpy import linspace
import pandas

from docopt import docopt


def classifiers_config(random_state, classifier_name=None, skip_grid=False):
    """Get classifiers config dict with parameter grid.

    Classifier config dict values can be iterables, meaning
    that a grid search should be performed.
    If they are not iterables (e.g. random_state is int),
    the values are just to be passed to a classifier constructor.

    Args:
        random_state: random_state config option to include to some classifiers.
        classifier_name: if specified, use only this classifier.
        skip_grid: if True, do not add grid parameters to search.

    Returns:
        dict {<classifier_name>: {'init': classifier init options,
                                  'grid': dict of classifier parameters grid}}.
    """
    classifiers = {
        'tree.DecisionTreeClassifier': {  # attributes: feature_importances_
            'init': {
                'random_state': random_state
            },
            'grid': {
                'criterion': ('gini', 'entropy'),
                'splitter': ('best', 'random'),
                'max_features': ('auto', 'sqrt', 'log2', None),
                'class_weight': ('balanced', None)
            }
        },
        'neighbors.KNeighborsClassifier': {  # no attributes
            'grid': {
                'n_neighbors': range(1, 101, 3),
                'weights': ('uniform', 'distance'),
                'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
                'p': range(1, 11)
            },
            'scaling': False,
        },
        'svm.LinearSVC': {  # attributes: coef_
            'init': {
                'random_state': random_state
            },
            'grid': {
                'C': range(11),
                'loss': ('hinge', 'squared_hinge'),
                'penalty': ('l1', 'l2'),
                'dual': (True, False),
                'multi_class': ('ovr', 'crammer_singer'),
                'class_weight': ('balanced', None)
            }
        },
        'svm.SVC': {  # no attributes
            'init': {
                'random_state': random_state
            },
            'grid': {
                'C': range(11),
                # exclude linear kernel, because we have LinearSVC, and
                # for SVC linear kernel is very slow
                'kernel': ('poly', 'rbf', 'sigmoid', 'precomputed'),
                'shrinking': (True, False),
                'class_weight': ('balanced', None),
                'decision_function_shape': ('ovo', 'ovr')
            }
        },
        'ensemble.RandomForestClassifier': {  # attributes: feature_importances_
            'init': {
                'random_state': random_state
            },
            'grid': {
                'n_estimators': range(2, 50, 4),
                'criterion': ('gini', 'entropy'),
                'max_features': ('auto', 'sqrt', 'log2', None),
                'bootstrap': (True, False),
                'oob_score': (True, False),
                'class_weight': ('balanced', 'balanced_subsample', None),
            }
        },
        'ensemble.ExtraTreesClassifier': {  # attributes: feature_importances_
            'init': {
                'random_state': random_state
            },
            'grid': {
                'n_estimators': range(2, 50, 4),
                'criterion': ('gini', 'entropy'),
                'max_features': ('auto', 'sqrt', 'log2', None),
                'bootstrap': (True, False),
                'oob_score': (True, False),
                'class_weight': ('balanced', 'balanced_subsample', None),
            }
        },
        'ensemble.AdaBoostClassifier': {  # attributes: feature_importances_
            'init': {
                'random_state': random_state
            },
            'grid': {
                'n_estimators': range(10, 101, 10),
                'learning_rate': [1] + list(linspace(0.1, 10, 21)),
                'algorithm': ('SAMME', 'SAMME.R'),
            }
        },
        'ensemble.GradientBoostingClassifier': {  # attributes: feature_importances_
            'init': {
                'random_state': random_state
            },
            'grid': {
                'loss': ('deviance', 'exponential'),
                'learning_rate': [0.01, 0.1, 0.2, 0.5],
                # 'learning_rate': [0.1] + list(linspace(0.01, 0.5, 5)),
                'n_estimators': [100, 300, 500],
                # 'n_estimators': range(50, 501, 50),
                'max_depth': [1, 3, 5, 7, 10],
                # 'max_depth': range(1, 11),
                'subsample': [0.1, 0.5, 1],
                # 'subsample': list(linspace(0.1, 1, 6)),
                'max_features': ('auto', 'sqrt', 'log2', None),
            }
        },
        'naive_bayes.GaussianNB': {},  # no attributes
        'discriminant_analysis.LinearDiscriminantAnalysis': {  # attributes: coef_
            'grid': {
                'solver': ('svd', 'lsqr', 'eigen'),
                'shrinkage': (None, 'auto'),
            }
        },
        'discriminant_analysis.QuadraticDiscriminantAnalysis': {},  # no attributes
    }
    if skip_grid:
        for name, config in classifiers.items():
            if 'grid' in config:
                config['grid'] = {}
    if classifier_name:
        classifiers = {name: config for name, config in classifiers.items()
                       if name == classifier_name}
    return classifiers


def get_file_hash(filename):
    """Calculate sha256 hash for a file by given filename."""
    return sha256(open(filename, 'rb').read()).hexdigest()


def make_config(options):
    """Make json configuration file for classification.

    Most parameters are the same and are taken from the command-line arguments,
        e.g. filenames of features, classes and configuration file.
    Some parameters are hard-coded (e.g. class names).
    And some are hard-coded and are cycled by (scaling, classifier names).
    So that every combination of parameters is written into config file.

    Args:
        options: CLI options dict.
    """
    random_state = int(options["--random-state"] or random.randint(0, 1000000))
    skip_grid = options["-n"]
    grid_scoring = options["--grid-scoring"]
    if grid_scoring:
        grid_scoring_options = (grid_scoring,)
    elif skip_grid:
        grid_scoring_options = ("f1_weighted",)
    else:
        grid_scoring_options = ("f1_weighted", "cohen_kappa", "accuracy")
    scaling = options["--scaling"]
    feature_scaling_options = (bool(scaling),) if scaling is not None else (False, True)

    class_names = ('0 Insuff', '1 Junior', '2 Exp-ed', '3 Expert')
    x_train = pandas.read_csv(options["<x_train.csv>"])
    y_train = pandas.read_csv(options["<y_train.csv>"])
    y_test = pandas.read_csv(options["<y_test.csv>"])
    features = x_train.columns

    # class_code: number of examples per class, e.g. {0: 10, 1: 20, 2: 35, 3: 12}
    train_classes_counts_raw = dict(y_train[y_train.columns[0]].value_counts())
    # class name: number of examples
    train_classes_counts = [(name, train_classes_counts_raw[i])
                            for i, name in enumerate(class_names)]
    test_classes_counts_raw = dict(y_test[y_test.columns[0]].value_counts())
    test_classes_counts = [(name, test_classes_counts_raw[i]) for i, name in enumerate(class_names)]

    classifiers = classifiers_config(random_state, options["--classifier"], skip_grid)
    config = []
    for classifier, scaling, grid_scoring in product(
            classifiers, feature_scaling_options, grid_scoring_options):
        if not classifiers[classifier].get('grid') \
                and grid_scoring != grid_scoring_options[0]:
            # if the classifier has no grid, we don't make configs
            # for non-first grid_scoring_options, because they will produce the same results,
            # because if there is no grid, grid scoring option does not make difference.
            continue
        classifier_scaling = classifiers[classifier].get('scaling')
        if classifier_scaling is not None and scaling != classifier_scaling:
            # if there is specific classifier scaling setting, use it always
            # and skip other options
            continue
        dct = OrderedDict()
        dct["classifier"] = OrderedDict((
            ("name", classifier),
            ("grid_scoring", grid_scoring),
            ("config", classifiers[classifier])))
        dct["classes"] = OrderedDict((
            ("train", OrderedDict((
                ("names", train_classes_counts),
                ("total", sum(dict(train_classes_counts).values())),
                ("filename", options[u"<y_train.csv>"]),
                ("filehash", get_file_hash(options["<y_train.csv>"]))))),
            ("test", OrderedDict((
                ("names", test_classes_counts),
                ("total", sum(dict(test_classes_counts).values())),
                ("filename", options[u"<y_test.csv>"]),
                ("filehash", get_file_hash(options["<y_test.csv>"])))))))
        dct["features"] = OrderedDict((
            ("scaling", scaling),
            ("count", len(features)),
            ("train_filename", options["<x_train.csv>"]),
            ("train_filehash", get_file_hash(options["<x_train.csv>"])),
            ("test_filename", options["<x_test.csv>"]),
            ("test_filehash", get_file_hash(options["<x_test.csv>"])),
            ("names", sorted(features))))
        dct["verbose"] = False
        dct["random_state"] = random_state
        config.append(dct)

    with open(options["<config.json>"], "w") as fout:
        json.dump(config, fout, indent=4)


if __name__ == "__main__":
    try:
        make_config(docopt(__doc__))
    except KeyboardInterrupt:
        pass
    sys.exit(0)
