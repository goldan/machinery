"""Run experiments using a json configuration file and store results in MongoDB."""
import sys
from importlib import import_module
from time import time

import numpy
import pandas
from featureforge.experimentation import runner
from sklearn import cross_validation, metrics
from sklearn.preprocessing import scale
from sklearn.utils.extmath import density

from matplotlib import pyplot
from machinery.common.utils import roundto


def prepare_data(options):
    """Load training and test sets for classification.

    Args:
        options: configuration dictionary. Used keys:
            ['features']['filename']: name of file with feature values.
            ['features']['scaling']: if True, scale features before using.
            ['classes']['filename']: name of file with assigned classes.
            ['classifier']['config']['random_state']: random seed value to use for
                splitting the data into training and test sets.

    Returns:
        tuple (X_train, X_test, y_train, y_test, feature_names)
            X_train: matrix of training data, training examples x feature values.
            X_test: matrix of test data, test examples x feature values.
            y_train: vector of right answers (classes) for training set.
            y_test: vector of right answers (classes) for test set.
            feature_names: list of names of features.
    """
    data = pandas.read_csv(options['features']['filename'])
    classes = pandas.read_csv(options['classes']['filename'])
    X = numpy.array(data)
    if options['features']['scaling']:
        X = scale(X)
    y = numpy.array(classes)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, random_state=options['classifier']['config']['random_state'])
    return X_train, X_test, y_train, y_test, data.columns


def get_classifier(options):
    """Get classifier instance from a configuration dict.

    Args:
        options: ['classifier'] configuration subdictionary. Used keys:
            ['name']: module and class name of the classifier, in dot notation.
            ['config']: configuration dict used in classifier instance constructor.

    Returns:
        classifier instance.
    """
    module_name = '.'.join(['sklearn'] + options['name'].split('.')[:-1])
    class_name = options['name'].split('.')[-1]
    module = import_module(module_name)
    classifier_class = getattr(module, class_name)
    classifier = classifier_class(**options['config'])
    return classifier


def train_classifier(classifier, X_train, y_train, results, verbose):
    """Train classifier using training data.

    Args:
        classifier: classifier instance.
        X_train: matrix of training data, training examples x feature values.
        y_train: vector of right answers (classes) for training set.
        results: dict to store training time in.
        verbose: if True, print information about training.
    """
    if verbose:
        print '_' * 80
        print "Training: "
        print classifier
    start = time()
    classifier.fit(X_train, y_train)
    train_time = time() - start
    if verbose:
        print "train time: %0.3fs" % train_time
    results["train_time"] = roundto(train_time)


def predict(classifier, X_test, results, verbose):
    """Use trained classifier to predict classes for test data.

    Args:
        classifier: classifier instance.
        X_test: matrix of test data, test examples x feature values.
        results: dict to store test time in.
        verbose: if True, print information about prediction.

    Returns:
        y_predicted: vector of predicted answers (classes) for test set.
    """
    start = time()
    y_predicted = classifier.predict(X_test)
    test_time = time() - start
    if verbose:
        print "test time:  %0.3fs" % test_time
    results["test_time"] = roundto(test_time)
    return y_predicted


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
                                       :, numpy.newaxis])
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


def train_and_evaluate_classifier(options):
    """Train and evaluate classifier based on configuration options.

    Args:
        options: configuration dictionary.

    Returns:
        results dict with classification evaluation results.
    """
    X_train, X_test, y_train, y_test, feature_names = prepare_data(options)
    classifier = get_classifier(options['classifier'])
    results = {}
    train_classifier(classifier, X_train, y_train, results, options['verbose'])
    y_predicted = predict(classifier, X_test, results, options['verbose'])
    evaluate(classifier, y_test, y_predicted, feature_names,
             options['classes']['names'], results, options['verbose'])
    return results


if __name__ == "__main__":
    try:
        runner.main(train_and_evaluate_classifier)
    except KeyboardInterrupt:
        pass
    sys.exit(0)
