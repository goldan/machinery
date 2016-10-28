"""Run experiments using a json configuration file and store results in MongoDB."""
import os
import pickle
import sys
from collections import OrderedDict
from importlib import import_module
from multiprocessing import cpu_count
from time import time

import numpy
import pandas
from featureforge.experimentation import runner
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import density

from machinery.common.utils import roundto


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
        options['features']['scaling'])

    classifier = get_classifier(options['classifier']['name'],
                                options['classifier']['config'].get('init', {}),
                                options['verbose'])

    grid = options['classifier']['config'].get('grid', {})
    classifier, grid_size, train_time = train_classifier(
        classifier, grid, X_train, y_train,
        options['classifier']['grid_scoring'],
        options['random_state'], options['verbose'])

    classifier_filename = options['classifier'].get('filename')
    if classifier_filename:
        with open(classifier_filename, 'wb') as f:
            pickle.dump(classifier, f)

    y_predicted, test_time = predict(classifier, X_test, options['verbose'])

    results = OrderedDict()
    results["train_time"] = roundto(train_time)
    results["test_time"] = roundto(test_time)
    results["grid_size"] = grid_size

    classes_names = [name for name, _ in options['classes']['train']['names']]
    evaluate(classifier, y_test, y_predicted, feature_names,
             classes_names, results, options['verbose'])
    return results


def prepare_data(x_train_filename, y_train_filename, x_test_filename, y_test_filename,
                 features_scaling):
    """Load training and test sets for classification.

    Load features and target classes from csv files.
    If features_scaling is True, scale features to zero mean and standard deviation.

    Args:
        features_filename: name of file with feature values.
        features_scaling: if True, scale features before using.
        classes_filename: name of file with assigned classes.

    Returns:
        tuple (X_train, X_test, y_train, y_test, feature_names)
            X_train: matrix of training data, training examples x feature values.
            X_test: matrix of test data, test examples x feature values.
            y_train: vector of right answers (classes) for training set.
            y_test: vector of right answers (classes) for test set.
            feature_names: list of names of features.
    """
    df = pandas.read_csv(x_train_filename)
    feature_names = df.columns
    X_train = df.as_matrix()
    y_train = pandas.read_csv(y_train_filename, header=None).ix[:, 0]
    X_test = pandas.read_csv(x_test_filename).as_matrix()
    y_test = pandas.read_csv(y_test_filename, header=None).ix[:, 0]
    if features_scaling:
        # setup scaler only on training set, and then
        # apply both to training and test sets
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, feature_names


def get_classifier(name, config_dict, verbose):
    """Get classifier instance a configuration dict.

    Args:
        name: module and class name of the classifier, in dot notation.
        config_dict: configuration dict used in classifier instance constructor.
        verbose: if True, init classifier with verbose=True.

    Returns:
        classifier instance.
    """
    module_name = '.'.join(['sklearn'] + name.split('.')[:-1])
    class_name = name.split('.')[-1]
    module = import_module(module_name)
    classifier_class = getattr(module, class_name)
    if verbose:
        config_dict['verbose'] = True
    return classifier_class(**config_dict)


def train_classifier(classifier, grid, X_train, y_train, grid_scoring, random_state, verbose):
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
        grid_scoring: metrics to optimize in grid search. See
            http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
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
        if grid_scoring == "cohen_kappa":
            grid_scoring = metrics.make_scorer(metrics.cohen_kappa_score)
        if 'kernel' in grid:
            # svm.SVC does not like unicode value of kernel, requires str
            grid['kernel'] = [str(elem) for elem in grid['kernel']]
        grid_searcher = GridSearchCV(
            classifier, grid, scoring=grid_scoring, cv=cross_validator,
            verbose=True, n_jobs=cpu_count(), error_score=0)
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
        * cohen kappa coefficient
        * precision, recall and f1-score for each class
        * average precision, recall and f1-score (average weighted by support)
        * classes support
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

    kappa_score = metrics.cohen_kappa_score(y_test, y_predicted)
    results['cohen_kappa'] = roundto(kappa_score, 4)

    precisions, recalls, fscores, supports = metrics.precision_recall_fscore_support(
        y_test, y_predicted)
    results['precisions'] = [roundto(prec, 4) for prec in precisions]
    results['recalls'] = [roundto(rec, 4) for rec in recalls]
    results['f1-scores'] = [roundto(fsco, 4) for fsco in fscores]
    results['supports'] = list(supports)
    results['support'] = sum(supports)

    # we choose weighted average as a most reasonable way to average
    # it is used by sklearn by default in classification report
    # classes are counted according to their support
    # see http://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel
    average = 'weighted' if len(class_names) > 2 else 'binary'
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
        y_test, y_predicted, average=average)
    results['precision'] = roundto(precision, 4)
    results['recall'] = roundto(recall, 4)
    results['f1-score'] = roundto(fscore, 4)

    if hasattr(classifier, 'coef_'):
        results['dimensionality'] = classifier.coef_.shape[1]
        results['density'] = roundto(density(classifier.coef_))
        if verbose:
            print "dimensionality: %d" % results['dimensionality']
            print "density: %f" % results['density']
            print ""
        # coef_ is a matrix (classes x features), values may be negative,
        # so we take abs of it and sum by all classes
        feature_weights = abs(classifier.coef_).sum(axis=0)
        set_important_features(feature_names, feature_weights, results, verbose)

    if hasattr(classifier, "feature_importances_"):
        set_important_features(feature_names, classifier.feature_importances_, results, verbose)

    results['report'] = metrics.classification_report(y_test, y_predicted, target_names=class_names)
    if verbose:
        print "classification report:"
        print results['report']

    get_confusion_matrix(y_test, y_predicted, class_names, results, verbose)
    if verbose:
        from matplotlib import pyplot
        pyplot.show()

    results['config'] = classifier.get_params()


def set_important_features(feature_names, feature_weights, results, verbose):
    """Calculate important features and store them (and their count) in results dict.

    A feature is considered important, if its weight is higher than a threshold (0.001).

    Args:
        feature_names: list of feature names.
        feature_weights: list of feature weights.
        results: dict to store results in.
        verbose: if True, print output to console.
    """
    important_features = sorted(zip(feature_names, feature_weights),
                                key=lambda (name, value): value, reverse=True)
    results['important_features'] = [(name, roundto(value)) for
                                     name, value in important_features if value >= 0.001]
    results['important_features_count'] = len(results['important_features'])
    if verbose:
        print "\n".join("%s: %.2f" % (name, value) for name, value in important_features)


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
        from matplotlib import pyplot
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
                          title='Confusion matrix', cmap=None):
    """Plot confusion matrix.

    Args:
        confusion_matrix: confusion matrix object.
        class_names: list of target classes names.
        title: plot title.
        cmap: color map for plotting.
    """
    from matplotlib import pyplot
    if not cmap:
        cmap = pyplot.cm.Blues
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
