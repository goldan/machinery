# -*- coding: utf-8 -*-
u"""Make json config file with experiment configuration.

Usage:
    make_config.py <features.csv> <classes.csv> <config.json>

Options:
 -h --help              Show this screen.
 --version              Show Version.
 <features.csv>         Name of csv file with feature values.
 <classes.csv>          Name of csv file with labeled classes.
 <config.json>          Name of json file to write configuration to.
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


def classifiers_config(random_state):
    """Get classifiers config dict with parameter grid.

    Classifier config dict values can be iterables, meaning
    that a grid search should be performed.
    If they are not iterables (e.g. random_state is int),
    the values are just to be passed to a classifier constructor.

    Args:
        random_state: random_state config option to include to some classifiers.

    Returns:
        dict {<classifier_name>: {'init': classifier init options,
                                  'grid': dict of classifier parameters grid}}.
    """
    return {
        'tree.DecisionTreeClassifier': {
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
        'neighbors.KNeighborsClassifier': {
            'grid': {
                'n_neighbors': range(1, 101, 3),
                'weights': ('uniform', 'distance'),
                'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
                'p': range(1, 11)
            }
        },
        'svm.LinearSVC': {
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
        'svm.SVC': {
            'init': {
                'random_state': random_state
            },
            'grid': {
                'C': range(11),
                'kernel': ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'),
                'shrinking': (True, False),
                'class_weight': ('balanced', None),
                'decision_function_shape': ('ovo', 'ovr')
            }
        },
        'ensemble.RandomForestClassifier': {
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
        'ensemble.ExtraTreesClassifier': {
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
        'ensemble.AdaBoostClassifier': {
            'init': {
                'random_state': random_state
            },
            'grid': {
                'n_estimators': range(10, 101, 10),
                'learning_rate': [1] + list(linspace(0.1, 10, 21)),
                'algorithm': ('SAMME', 'SAMME.R'),
            }
        },
        'ensemble.GradientBoostingClassifier': {
            'init': {
                'random_state': random_state
            },
            'grid': {
                'loss': ('deviance', 'exponential'),
                'learning_rate': [0.1] + list(linspace(0.01, 3, 21)),
                'n_estimators': range(50, 301, 25),
                'max_depth': range(1, 11),
                'subsample': [1] + list(linspace(0.1, 3, 21)),
                'max_features': ('auto', 'sqrt', 'log2', None),
            }
        },
        'naive_bayes.GaussianNB': {},
        'discriminant_analysis.LinearDiscriminantAnalysis': {
            'grid': {
                'solver': ('svd', 'lsqr', 'eigen'),
                'shrinkage': (None, 'auto'),
            }
        },
        'discriminant_analysis.QuadraticDiscriminantAnalysis': {},
    }


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
    random_state = random.randint(0, 1000000)
    class_names = ('0 Insuff', '1 Junior', '2 Exp-ed', '3 Expert')
    feature_scaling_options = (False, True)
    data = pandas.read_csv(options["<features.csv>"])
    classes = pandas.read_csv(options["<classes.csv>"])
    # class_code: number of examples per class, e.g. {0: 10, 1: 20, 2: 35, 3: 12}
    classes_counts_raw = dict(classes[classes.columns[0]].value_counts())
    # class name: number of examples
    classes_counts = [(name, classes_counts_raw[i]) for i, name in enumerate(class_names)]
    classifiers = classifiers_config(random_state)
    config = []
    for classifier, scaling in product(classifiers, feature_scaling_options):
        dct = OrderedDict()
        dct["classifier"] = OrderedDict((
            ("name", classifier),
            ("config", classifiers[classifier])))
        dct["classes"] = OrderedDict((
            ("names", classes_counts),
            ("total", sum(classes_counts_raw.values())),
            ("filename", options[u"<classes.csv>"]),
            ("filehash", sha256(open(options[u"<classes.csv>"], 'rb').read()).hexdigest())))
        dct["features"] = OrderedDict((
            ("scaling", scaling),
            ("count", len(data.columns)),
            ("filename", options["<features.csv>"]),
            ("filehash", sha256(open(options["<features.csv>"], 'rb').read()).hexdigest()),
            ("names", sorted(data.columns))))
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
