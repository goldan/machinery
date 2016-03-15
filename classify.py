"""Run experiments using a json configuration file and store results in MongoDB."""
import os
import sys
from collections import OrderedDict
from importlib import import_module
from multiprocessing import cpu_count
from time import time

import numpy
import pandas
from featureforge.experimentation import runner
from sklearn import cross_validation, metrics
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.utils.extmath import density

from machinery.common.utils import roundto
from matplotlib import pyplot


def train_and_evaluate_classifier(options):
    """Train and evaluate classifier based on configuration options.

    Args:
        options: configuration dictionary.

    Returns:
        results dict with classification evaluation results.
    """
    X_train, X_test, y_train, y_test, feature_names = prepare_data(
        options['features']['train_filename'], options['classes']['train']['filename'],
        options['features']['test_filename'], options['classes']['test']['filename'],
        options['features']['scaling'], options['random_state'])

    classifier = get_classifier(options['classifier']['name'],
                                options['classifier']['config'].get('init', {}))

    grid = options['classifier']['config'].get('grid', {})
    classifier, grid_size, train_time = train_classifier(
        classifier, grid, X_train, y_train,
        options['random_state'], options['verbose'])

    y_predicted, test_time = predict(classifier, X_test, options['verbose'])

    results = OrderedDict()
    results["train_time"] = roundto(train_time)
    results["test_time"] = roundto(test_time)
    results["grid_size"] = grid_size

    evaluate(classifier, y_test, y_predicted, feature_names,
             dict(options['classes']['train']['names']).keys(), results, options['verbose'])
    return results


def prepare_data(x_train_filename, y_train_filename, x_test_filename, y_test_filename,
        features_scaling, random_state):
    """Load training and test sets for classification.

    Load features and target classes from csv files.
    If features_scaling is True, scale features to zero mean and standard deviation.
    Split data to train and test sets (test is 25%), using random_state.

    Args:
        features_filename: name of file with feature values.
        features_scaling: if True, scale features before using.
        classes_filename: name of file with assigned classes.
        random_state: random seed value to use for splitting the data
            into training and test sets.

    Returns:
        tuple (X_train, X_test, y_train, y_test, feature_names)
            X_train: matrix of training data, training examples x feature values.
            X_test: matrix of test data, test examples x feature values.
            y_train: vector of right answers (classes) for training set.
            y_test: vector of right answers (classes) for test set.
            feature_names: list of names of features.
    """
    X_train = pandas.read_csv(x_train_filename)
    y_train = pandas.read_csv(y_train_filename)
    X_test = pandas.read_csv(x_test_filename)
    y_test = pandas.read_csv(y_test_filename)
    feature_names = X_train.columns
    if features_scaling:
        X_train = scale(X_train)
        X_test = scale(X_test)
    return X_train, X_test, y_train, y_test, feature_names


def get_classifier(name, config_dict):
    """Get classifier instance a configuration dict.

    Args:
        name: module and class name of the classifier, in dot notation.
        config_dict: configuration dict used in classifier instance constructor.

    Returns:
        classifier instance.
    """
    module_name = '.'.join(['sklearn'] + name.split('.')[:-1])
    class_name = name.split('.')[-1]
    module = import_module(module_name)
    classifier_class = getattr(module, class_name)
    return classifier_class(**config_dict)


def train_classifier(classifier, grid, X_train, y_train, random_state, verbose):
    """Train classifier using training data.

    If grid is provided, use GridSearchCV to find the best hyperparameters
    of the classifier. Use KFold to split training data for grid searching.
    Training data is shuffled before folding into 10 folds.
    Then grid search is run against each possible set of parameter values
    on these folds, and mean score is calculated for the folds, and
    the best parameter set is taken.
    When the best estimator is selected, it is refit on the full training
    set (unfolded), so that it's best trained before applying for test set.

    Args:
        classifier: classifier instance.
        grid: dict of grid params to select the best combination.
        X_train: matrix of training data, (training examples x feature values).
        y_train: vector of right answers (classes) for training set.
        random_state: random seed value to use for splitting the data
            into training and test sets in model selection folding.
        verbose: if True, print information about training.

    Returns:
        tuple (best_classifier, train_time):
        best_classifier: classifier trained with best grid params subset
            on the whole training set.
        train_time: time that training took.
    """
    if verbose:
        print '_' * 80
        print "Training: "
        print classifier
    start = time()

    if grid:
        cross_validator = KFold(
            y_train.size, n_folds=10, shuffle=True, random_state=random_state)
        grid_searcher = GridSearchCV(
            classifier, grid, scoring='accuracy', cv=cross_validator,
            verbose=True, n_jobs=cpu_count())
        # it automatically refits the best classifier
        # to the full train set. So it's ready to be used to predict.
        best_classifier = grid_searcher.fit(X_train, y_train).best_estimator_
        grid_size = len(grid_searcher.grid_scores_)
    else:
        classifier.fit(X_train, y_train)
        best_classifier = classifier
        grid_size = 1

    train_time = time() - start
    if verbose:
        print "train time: %0.3fs" % train_time

    return best_classifier, grid_size, train_time


def predict(classifier, X_test, verbose):
    """Use trained classifier to predict classes for test data.

    Args:
        classifier: classifier instance.
        X_test: matrix of test data, test examples x feature values.
        verbose: if True, print information about prediction.

    Returns:
        y_predicted: vector of predicted answers (classes) for test set.
        test_time: time that prediction took.
    """
    start = time()
    y_predicted = classifier.predict(X_test)
    test_time = time() - start
    if verbose:
        print "test time:  %0.3fs" % test_time
    return y_predicted, test_time


def evaluate(classifier, y_test, y_predicted, feature_names, class_names, results, verbose):
    """Evaluate classifier performance.

    Metrics calculated:
        * accuracy
        * dimensionality
        * density
        * important features
        * classification report (precision/recall/f-score)
        * confusion matrix
        * normalized confusion matrix.

    Args:
        classifier: classifier instance.
        y_test: vector of right answers (classes) for test set.
        y_predicted: vector of predicted answers (classes) for test set.
        feature_names: list of names of features.
        class_names: list of target classes names.
        results: dict to store evaluation metrics in.
        verbose: if True, print information about evaluation.
    """
    score = metrics.accuracy_score(y_test, y_predicted)
    if verbose:
        print "accuracy:   %0.3f" % score
    results['accuracy'] = roundto(score, 4)

    if hasattr(classifier, 'coef_'):
        results['dimensionality'] = classifier.coef_.shape[1]
        results['density'] = roundto(density(classifier.coef_))
        if verbose:
            print "dimensionality: %d" % results['dimensionality']
            print "density: %f" % results['density']
            print ""

    if hasattr(classifier, "feature_importances_"):
        important_features = sorted(zip(feature_names, classifier.feature_importances_),
                                    key=lambda (name, value): value, reverse=True)
        results['important_features'] = [(name, roundto(value)) for
                                         name, value in important_features if value >= 0.001]
        results['important_features_count'] = len(results['important_features'])
        if verbose:
            print "\n".join("%s: %.2f" % (name, value) for name, value in important_features)

    results['report'] = metrics.classification_report(y_test, y_predicted, target_names=class_names)
    if verbose:
        print "classification report:"
        print results['report']

    get_confusion_matrix(y_test, y_predicted, class_names, results, verbose)
    if verbose:
        pyplot.show()

    results['config'] = classifier.get_params()


def get_confusion_matrix(y_test, y_predicted, class_names, results, verbose):
    """Calculate confusion and normalized confusion matrices for predicted values.

    Normalized confusion matrix is normalized by row (i.e by the number of samples in each class).

    Args:
        y_test: vector of right answers (classes) for test set.
        y_predicted: vector of predicted answers (classes) for test set.
        class_names: list of target classes names.
        results: dict to store matrices in.
        verbose: if True, print matrix to screen and plot.
    """
    confusion_matrix = metrics.confusion_matrix(y_test, y_predicted)
    results['confusion_matrix'] = str(confusion_matrix)
    normalized_confusion_matrix = (confusion_matrix.astype('float') /
                                   confusion_matrix.sum(axis=1)[
                                       :, numpy.newaxis]).round(2)
    results['normalized_confusion_matrix'] = str(normalized_confusion_matrix)

    if verbose:
        print "confusion matrix:"
        numpy.set_printoptions(precision=2)
        print 'Confusion matrix'
        print results['confusion_matrix']
        pyplot.figure()
        plot_confusion_matrix(confusion_matrix, class_names)
        print 'Normalized confusion matrix'
        print normalized_confusion_matrix
        pyplot.figure()
        plot_confusion_matrix(normalized_confusion_matrix, class_names,
                              title='Normalized confusion matrix')
        print ""


def plot_confusion_matrix(confusion_matrix, class_names,
                          title='Confusion matrix', cmap=pyplot.cm.Blues):
    """Plot confusion matrix.

    Args:
        confusion_matrix: confusion matrix object.
        class_names: list of target classes names.
        title: plot title.
        cmap: color map for plotting.
    """
    pyplot.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    pyplot.title(title)
    pyplot.colorbar()
    tick_marks = numpy.arange(len(class_names))
    pyplot.xticks(tick_marks, class_names, rotation=45)
    pyplot.yticks(tick_marks, class_names)
    pyplot.tight_layout()
    pyplot.ylabel('True label')
    pyplot.xlabel('Predicted label')


if __name__ == "__main__":
    try:
        runner.main(train_and_evaluate_classifier,
                    use_git_info_from_path=os.path.dirname(__file__) or '.')
    except KeyboardInterrupt:
        pass
    sys.exit(0)
