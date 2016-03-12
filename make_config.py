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
from itertools import product
import pandas

from docopt import docopt


def make_config():
    """Make json configuration file for classification.

    Most parameters are the same and are taken from the command-line arguments,
        e.g. filenames of features, classes and configuration file.
    Some parameters are hard-coded (e.g. class names).
    And some are hard-coded and are cycled by (scaling, classifier names).
    So that every combination of parameters is written into config file.
    """
    options = docopt(__doc__)
    random_state = random.randint(0, 1000000)
    class_names = ('0 Insuff', '1 Junior', '2 Exp-ed', '3 Expert')
    classifiers = ['tree.DecisionTreeClassifier', 'svm.LinearSVC']
    feature_scaling_options = (False, True)
    verbose = False
    data = pandas.read_csv(options["<features.csv>"])
    config = []
    for classifier, scaling in product(classifiers, feature_scaling_options):
        dct = OrderedDict()
        dct["classifier"] = OrderedDict((
            ("name", classifier),
            ("config", {
                "random_state": random_state
            })))
        dct["classes"] = OrderedDict((
            ("filename", options[u"<classes.csv>"]),
            ("names", class_names)))
        dct["features"] = OrderedDict((
            ("scaling", scaling),
            ("count", len(data.columns)),
            ("filename", options["<features.csv>"]),
            ("names", sorted(data.columns))))
        dct["verbose"] = verbose
        config.append(dct)

    with open(options["<config.json>"], "w") as fout:
        json.dump(config, fout, indent=4)


if __name__ == "__main__":
    try:
        make_config()
    except KeyboardInterrupt:
        pass
    sys.exit(0)
