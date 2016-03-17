"""Feature classes for Candidate experience classification."""
import copy
import csv
import pickle
import string
import sys
from argparse import ArgumentParser
from collections import OrderedDict, Counter
from itertools import product
from types import MethodType

from featureforge.vectorizer import Vectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from machinery.common.atlas import (GENDER_MALE, PROFILE_TYPES_BY_TYPE_ID,
                                    Candidate, CandidateExperienceClass,
                                    Organization, sourcerer_get_by_id)
from machinery.common.features import (AttributeBool, AttributeInt,
                                       AttributeLen, DictKeyBool, DictKeyInt,
                                       DictKeyLen, ListFeature, SetFeature,
                                       SubAttributeSet,
                                       make_list_of_values_features)
from machinery.common.utils import CSV_OPTIONS, writerow, flatten_list


class Gender(AttributeBool):
    """Gender feature. 1 for male, 0 for women (no sexism)."""

    attribute_name = "gender"
    default = True

    def _postprocess(self, gender):
        """Convert gender code to binary format."""
        return gender == GENDER_MALE


class SkillsFeature(ListFeature):
    """Feature for representing profile skill points."""

    _name = 'skills'

    def _preprocess(self, profile):
        """Get profile skill points for further processing.

        Args:
            profile to evaluate.

        Returns:
            list of skill points for the profile.
        """
        return [skill.points for skill in getattr(profile, "skills", [])
                if hasattr(skill, "points")]


class FootprintsFeature(ListFeature):
    """Feature for representing profile digital footprint scores."""

    _name = 'footprints'

    def _preprocess(self, profile):
        """Get profile digital footprint scores for further processing.

        Args:
            profile to evaluate.

        Returns:
            list of footprint scores of the profile.
        """
        return [footprint.value for footprint in
                getattr(profile, "digital_footprint_scores", []) or []
                if hasattr(footprint, "value")]


class ProfileOrganizationNames(SetFeature):
    """Bag-of-words feature representing all organizations in a profile.

    It's not to be used by itself, but only to calculate popular organizations
    for a candidate.
    """

    @staticmethod
    def normalize_name(name):
        """Normalize organization name from dirty data.

        What it does:
        * lowercase
        * replace commas with spaces
        * remove all punctuation from both sides of the name.
        * find name as a synonym in the collection (case-insensitive).
            If found, replace with the lowercased main name.

        Args:
            name: name to normalize.

        Returns:
            normalized organization name.
        """
        new_name = name.lower().replace(",", " ").replace("  ", " ")
        organizations = Organization.objects.filter(synonyms__iexact=new_name)
        if organizations:
            new_name = organizations[0].name.lower()
        exclude = "".join(string.punctuation)
        new_name = new_name.strip(exclude)
        return new_name

    def _postprocess(self, profile):
        """Get set of all organization names in the profile.

        Args:
            profile to evaluate.

        Returns:
            set of normalized organization names in the profile.organizations.
        """
        names = set()
        for organization in getattr(profile, "organizations", []) or []:
            if organization.name:
                names.add(self.normalize_name(organization.name))
        return names


class ProfileJobTitlesWords(SetFeature):
    """Bag-of-words feature representing all words in job titles in a profile.

    It's not to be used by itself, but only to calculate popular job titles
    for a candidate.
    """

    @staticmethod
    def normalize_title(title):
        """Normalize job title.

        What it does:
        * lowercase
        * tokenize
        * remove punctuation from sides of every token
        * remove english stopwords
        * lemmatize using WordNet

        It's made a staticmethod to enable usage on cls objects,
            e.g. in get_counter.

        Returns:
            set of tokens in the normalized title.
        """
        tokens = nltk.word_tokenize(title.lower())
        exclude = "".join(string.punctuation)
        stop = stopwords.words('english')
        lemmatizer = WordNetLemmatizer()
        token_set = set([])
        for token in tokens:
            tok = token.strip(exclude)
            if tok and tok not in stop:
                token_set.add(lemmatizer.lemmatize(tok))
        return token_set

    def _postprocess(self, profile):
        """Get set of all words in job titles in the profile.

        Args:
            profile to evaluate.

        Returns:
            set of normalized words occuring in job titles in the profile.organizations.
        """
        titles_words = set()
        for organization in getattr(profile, "organizations", []) or []:
            if getattr(organization, "title", None):
                titles_words |= self.normalize_title(organization.title)
        return titles_words


class OrganizationNames(SetFeature):
    """Bag-of-words feature representing popular organizations in candidate profiles.

    Attributes:
        organizations: set of popular organization names to search for in the profiles.
    """

    def __init__(self, org_popularity_file, org_popularity_threshold=100):
        """Store organizations to search for.

        Args:
            org_popularity_file: file descriptor with organization names popularity table.
            org_popularity_threshold: number of times organization needs to appear in jobs
                to be included as a separate feature
        """
        super(OrganizationNames, self).__init__()
        self.load_organizations(org_popularity_file, org_popularity_threshold)

    def load_organizations(self, org_popularity_file, threshold):
        """Load organization names to search for.

        Args:
            org_popularity_file: file descriptor with organization names popularity table.
            threshold: number of times organization needs to appear in jobs
                to be included as a separate feature.
        """
        reader = csv.reader(org_popularity_file, **CSV_OPTIONS)
        self.organizations = set([row[0].decode('utf8') for row in reader if
                                  int(row[1]) > threshold])

    def _process(self, candidate):
        """Get set of given organizations names occuring in the candidate profiles.

        Args:
            candidate to evaluate.

        Returns:
            set of organization names (among self.organizations)
            which occur in the candidate profiles.
        """
        profiles = dict((profile.type_id, profile)
                        for profile in getattr(candidate, "profiles", []))
        org_names = set()
        for type_id in PROFILE_TYPES_BY_TYPE_ID:
            profile = profiles.get(type_id)
            feature = ProfileOrganizationNames()
            org_names |= feature._evaluate(profile)
        return org_names & self.organizations


class JobTitlesWords(SetFeature):
    """Bag-of-words feature representing popular words in job titles in candidate profiles.

    Attributes:
        job_titles_words: set of popular job title words to search for in the profiles.
    """

    def __init__(self, words_popularity_file, words_popularity_threshold=100):
        """Store job title words to search for.

        Args:
            words_popularity_file: file descriptor with job title words popularity table.
            words_popularity_threshold: number of times a job title word must appear in jobs
                to be included as a separate feature.
        """
        super(JobTitlesWords, self).__init__()
        self.load_job_title_words(words_popularity_file, words_popularity_threshold)

    def load_job_title_words(self, words_popularity_file, threshold):
        """Load job title words to search for.

        Args:
            words_popularity_file: file descriptor with job title words popularity table.
            threshold: number of times a word needs to appear in jobs
                to be included as a separate feature.
        """
        reader = csv.reader(words_popularity_file, **CSV_OPTIONS)
        self.job_titles_words = set([row[0].decode('utf8') for row in reader if
                                     int(row[1]) > threshold])

    def _process(self, candidate):
        """Get set of given job titles words occuring in the candidate profiles.

        Args:
            candidate to evaluate.

        Returns:
            set of job titles words (among self.job_titles_words)
            which occur in the candidate profiles.
        """
        profiles = dict((profile.type_id, profile)
                        for profile in getattr(candidate, "profiles", []))
        titles_words = set()
        for type_id in PROFILE_TYPES_BY_TYPE_ID:
            profile = profiles.get(type_id)
            feature = ProfileJobTitlesWords()
            titles_words |= feature._evaluate(profile)
        return titles_words & self.job_titles_words


def make_candidate_feature(profile_feature, profile_type_id):
    """Make a candidate feature out of the profile feature.

    Given specific profile type_id and a profile feature,
    make a candidate feature which calls a profile feature for a given profile type.
    All profile_feature's attribute and methods are copied to the new feature.

    Args:
        profile_feature: feature expecting a CandidateProfile as a data point.
        profile_type_id: type_id of the profile to make feature for.

    Returns:
        copy of profile_feature with name and _postprocess method updated.
    """
    feature = copy.copy(profile_feature)
    feature._name = "%s %s" % (PROFILE_TYPES_BY_TYPE_ID[profile_type_id].display_name,
                               profile_feature.name)

    def _evaluate(self, candidate, profile_feature=profile_feature, type_id=profile_type_id):
        """Evaluate the candidate feature using underlying profile feature.

        Args:
            candidate: data point to evaluate.
            profile_feature: profile feature used to evaluate the profile.
            type_id: type id of the profile to be evaluated.

        Returns:
            evaluation of the feature for the given profile.
        """
        if not getattr(candidate, "profiles", None):
            return profile_feature.default
        profiles = dict((profile.type_id, profile) for profile in candidate.profiles
                        if hasattr(profile, "type_id"))
        profile = profiles.get(type_id)
        return profile_feature._evaluate(profile)

    feature._evaluate = MethodType(_evaluate, feature)
    return feature


def make_candidate_features(profile_features):
    """Make candidate features for each of the profile features and profile types.

    Args:
        profile_features: list of features expecting CandidateProfile objects as data points.
            Note that those features can operate on any profile type.

    Returns:
        list of candidate features for every pair of profile feature and profile type.
    """
    return [make_candidate_feature(profile_feature, type_id)
            for type_id, profile_feature in product(PROFILE_TYPES_BY_TYPE_ID, profile_features)]


def profile_aggregate_feature(profile_feature, aggregating_func, name_suffix,
                              exclude_default=False):
    """Create a candidate-level aggregating feature from profile-level feature.

    Take a feature which operates on a profile and make a feature which takes a candidate,
    evaluates the feature on every of his profile and combines the results using
    the aggregating function.

    Args:
        profile_feature: feature operating on a single profile.
        aggregating_func: function taking profile feature values and outputs a single value.
        name_suffix: suffix to be added to the name of the underlying feature, to
            form the name of the resulting feature.
        exclude_default: if True, filter the list of values of the feature on profiles,
            excluding values equal to feature.default value. So only non-default
            values are passed to the aggregating function
            If there are none of them, just return default value without calling the
            aggregating function.

    Returns:
        candidate feature, aggregating profile feature values on all candidate profiles.
    """
    feature = copy.copy(profile_feature)
    feature._name = "%s %s" % (profile_feature.name, name_suffix)

    def _evaluate(self, candidate, profile_feature=profile_feature):
        """Evaluate the candidate feature using underlying profile feature.

        Args:
            candidate: data point to evaluate.
            profile_feature: profile feature used to evaluate the profile.

        Returns:
            evaluation of the feature for the given profile.
        """
        if not getattr(candidate, "profiles", None):
            return profile_feature.default
        values = [profile_feature._evaluate(profile) for profile in candidate.profiles]
        value = profile_feature.default
        if exclude_default:
            # do aggregation only on non-default values, if any
            # if they all default, just return the default value
            values = [value for value in values if value != profile_feature.default]
        if values:
            value = aggregating_func(values)
        return value

    feature._evaluate = MethodType(_evaluate, feature)
    return feature


def profile_most_common_features(profile_features):
    """Make candidate features outputting most common profile values.

    Take a list of profile_features operating on profiles and
    create a list of candidate-level features. Each candidate-level feature
    outputs the most common value of the corresponding profile feature
    among all candidate profiles.

    Args:
        profile_features: list of profile features used to evaluate profiles.

    Returns:
        list of candidate-level features, each of which outputs most common
        value of corresponding profile feature among all candidate profiles.
    """
    def take_most_common(values):
        """Take an iterable of values and return the most popular one."""
        counter = Counter(values)
        return counter.most_common()[0][0]

    return [profile_aggregate_feature(
        feature, take_most_common, "most common", True) for feature in profile_features]


def profile_any_exists_features(profile_features):
    """Make candidate features outputting OR on profile feature values.

    Take a list of profile_features operating on profiles and
    create a list of candidate-level features. Each candidate-level feature
    outputs boolean OR on values of the corresponding profile feature
    among all candidate profiles.

    Args:
        profile_features: list of profile features used to evaluate profiles.

    Returns:
        list of candidate-level features, each of which outputs the boolean OR on
        values of corresponding profile feature among all candidate profiles.
    """
    return [profile_aggregate_feature(
        feature, any, "exists") for feature in profile_features]


def profile_sum_features(profile_features):
    """Make candidate features outputting sum of profile feature values.

    Take a list of profile_features operating on profiles and
    create a list of candidate-level features. Each candidate-level feature
    outputs sum of values of the corresponding profile feature
    among all candidate profiles.

    Args:
        profile_features: list of profile features used to evaluate profiles.

    Returns:
        list of candidate-level features, each of which outputs sum of
        values of corresponding profile feature among all candidate profiles.
    """
    return [profile_aggregate_feature(
        feature, sum, "sum") for feature in profile_features]


def profile_bag_of_words_features(profile_features):
    """Make candidate features outputting combined bag of words on profile feature values.

    Take a list of profile_features operating on profiles and
    create a list of candidate-level features. Each candidate-level feature
    outputs combined set of values of the corresponding profile feature
    among all candidate profiles.

    Args:
        profile_features: list of profile features used to evaluate profiles.

    Returns:
        list of candidate-level features, each of which outputs set of all
        values of corresponding profile feature among all candidate profiles.
    """
    def combine_bags_of_words(values):
        """Take a list of sets of values and return a set of all of them."""
        return set(flatten_list([list(value) for value in values]))

    return [profile_aggregate_feature(
        feature, combine_bags_of_words, "set") for feature in profile_features]


def candidate_sourcerer_feature(sourcerer_feature, sourcerer_cache):
    """Make candidate feature from sourcerer result feature.

    Take a feature operating on a sourcerer result (a dict)
    and convert it to a feature operating on a candidate.

    Args:
        sourcerer_feature: feature operating on a sourcerer response.
        sourcerer_cache: dict {candidate_id: sourcerer} used not to
            hit sourcerer multiple times for a candidate.

    Returns:
        candidate feature which outputs sourcerer_feature on candidate.
    """
    feature = copy.copy(sourcerer_feature)

    def _evaluate(self, candidate, sourcerer_feature=sourcerer_feature, cache=sourcerer_cache):
        """Evaluate the candidate feature using underlying sourcerer feature.

        Args:
            candidate: data point to evaluate.
            sourcerer_feature: sourcerer feature used to evaluate the sourcerer response.
            cache: dict {candidate_id: sourcerer} used not to hit sourcerer
                multiple times for a candidate.

        Returns:
            evaluation of the feature for the given candidate.
        """
        if not getattr(candidate, "id", None):
            return sourcerer_feature.default
        sourcerer_candidate = cache.get(candidate.id)
        if not sourcerer_candidate:
            sourcerer_candidate = sourcerer_get_by_id(candidate.id)
            cache[candidate.id] = sourcerer_candidate
        return sourcerer_feature._evaluate(sourcerer_candidate)

    feature._evaluate = MethodType(_evaluate, feature)
    return feature


def candidate_sourcerer_features(sourcerer_features):
    """Make candidate features from sourcerer result features.

    Take features operating on a sourcerer result (a dict)
    and convert them to features operating on a candidate.

    Args:
        sourcerer_features: list of features operating on a sourcerer response.

    Returns:
        list of candidate features which outputs sourcerer_features values on candidate.
    """
    sourcerer_cache = {}
    return [candidate_sourcerer_feature(feature, sourcerer_cache)
            for feature in sourcerer_features]


def get_features(org_popularity_file, job_words_popularity_file):
    """Get a list of all features for candidates.

    It includes:
        * candidate sourcerer features (years_experience)
        * profile aggregating features (age, organization names, job titles words)
        * profile-specific features (url)

    Args:
        org_popularity_file: file descriptor with organization names popularity table.
        job_words_popularity_file: file descriptor with job title words popularity table.

    Returns:
        tuple of all candidates feature instances.
    """
    candidates_base_features = candidate_sourcerer_features((
        DictKeyInt("years_experience"),
        DictKeyBool("first_name"),
        DictKeyBool("last_name"),
        DictKeyBool("full_name"),
        DictKeyBool("photo"),
        DictKeyBool("location_display"),
        DictKeyLen("orgs")))

    pr_most_common_features = profile_most_common_features((
        AttributeInt("age"),
        Gender()))

    pr_any_exists_features = profile_any_exists_features((
        AttributeBool("bio"),
        AttributeBool("websites")))

    pr_bag_of_words_features = profile_bag_of_words_features((
        SubAttributeSet("location", "country"),
        SubAttributeSet("location", "state"),
        SubAttributeSet("location", "city")))

    pr_sum_features = profile_sum_features((
        AttributeLen("chats"),
        AttributeInt("followers"),
        AttributeInt("following")))

    profile_specific_features = [
        AttributeBool("url"),
        AttributeInt("followers"),
        AttributeInt("following")]
    profile_features = (profile_specific_features +
                        make_list_of_values_features(SkillsFeature) +
                        make_list_of_values_features(FootprintsFeature))
    candidate_profile_features = make_candidate_features(profile_features)

    candidate_aggregated_features = [
        OrganizationNames(org_popularity_file),
        JobTitlesWords(job_words_popularity_file)]

    features = (
        candidates_base_features +
        pr_most_common_features +
        pr_any_exists_features +
        pr_bag_of_words_features +
        pr_sum_features +
        candidate_profile_features +
        candidate_aggregated_features)

    return features


def export_features(candidates_train, candidates_test,
                    features_train_filename, features_test_filename,
                    popular_organizations_filename, job_title_words_filename,
                    verbose):
    """Export feature values for candidates to a csv or binary file.

    Take candidates classified by humans.

    Args:
        candidates_train: list of candidates in training set to export features for.
        candidates_test: list of candidates in test set to export features for.
        features_train_filename: name of file to export train feature values to.
        features_test_filename: name of file to export test feature values to.
        popular_organizations_filename: name of file containing
            organizations popularity table.
        job_title_words_filename: name of file containing
            job titles popularity table.
        verbose: if set, export human-readable csv; otherwise export binary pickled sparse matrix
            which is faster to process by classification algorithms.
    """
    with open(features_train_filename, "wb") as ftrain, \
            open(features_test_filename, "wb") as ftest, \
            open(popular_organizations_filename, "rb") as popular_organizations_file, \
            open(job_title_words_filename, "rb") as popular_job_title_words_file:
        features = get_features(popular_organizations_file, popular_job_title_words_file)
        train_writer = csv.writer(ftrain, **CSV_OPTIONS)
        test_writer = csv.writer(ftest, **CSV_OPTIONS)
        i = 0
        vectorizer = Vectorizer(features, sparse=not verbose)
        # note that we fit the vectorizer both for train and test candidates,
        # which means that we take all possible values for bag-of-words features,
        # both from training and test candidates.
        # it's ok, because we just construct all possible values (from popular lists).
        vectorizer.fit(list(candidates_train)+list(candidates_test))
        values_train = vectorizer.transform(candidates_train)
        values_test = vectorizer.transform(candidates_test)
        if verbose:
            headers = []
            for i in range(values_train[0].shape[0]):
                feature, info = vectorizer.column_to_feature(i)
                header = feature.name + ((" " + info) if info else "")
                headers.append(header)
            writerow(train_writer, headers)
            writerow(test_writer, headers)
            for row in values_train:
                writerow(train_writer, row)
                i += 1
                print "\r%d" % i,
            for row in values_test:
                writerow(test_writer, row)
                i += 1
                print "\r%d" % i,
        else:
            pickle.dump(values_train, ftrain)
            pickle.dump(values_test, ftest)


def export_classes(candidates, filename):
    """Export experience classes codes for candidates to a csv file.

    Take candidates classified by humans.

    Args:
        candidates: list of candidates to export features for.
        filename: name of file to export classes to.
    """
    with open(filename, "wb") as fout:
        writer = csv.writer(fout, **CSV_OPTIONS)
        writerow(writer, ["Experience class"])
        for candidate in candidates:
            experience = OrderedDict(CandidateExperienceClass.CLASS_CHOICES).keys().index(
                candidate.experience.code)
            writerow(writer, [experience])


def main():
    """Function that is executed when a file is called from CLI.

    CLI arguments:
        popular_organizations_filename: name of file containing
            organizations popularity table.
        popular_job_title_words_filename: name of file containing
            job titles popularity table.
        x_train_filename: name of file to export training set feature values to.
        y_train_filename: name of file to export training set classes to.
        x_test_filename: name of file to export test set feature values to.
        y_test_filename: name of file to export test set classes to.
        limit: (optional) number of candidates to export features for.
        verbose: if set, export human-readable csv; otherwise export binary pickled sparse matrix
            which is faster to process by classification algorithms.
    """
    parser = ArgumentParser()
    parser.add_argument('--org-names', dest='popular_organizations_filename', type=str)
    parser.add_argument('--job-titles', dest='popular_job_title_words_filename', type=str)
    parser.add_argument('--x-train', dest='x_train_filename', type=str)
    parser.add_argument('--y-train', dest='y_train_filename', type=str)
    parser.add_argument('--x-test', dest='x_test_filename', type=str)
    parser.add_argument('--y-test', dest='y_test_filename', type=str)
    parser.add_argument('--limit', dest='limit', type=int)
    parser.add_argument('--verbose', dest='verbose', type=bool)
    options = parser.parse_args()
    candidates = Candidate.objects.filter(experience__exists=True,
                                          experience__classifier_category='H')
    if options.limit:
        candidates = candidates[:options.limit]
    candidates_test = candidates.filter(experience__test_set=True)
    test_ids = candidates_test.values_list('id')
    candidates_train = candidates.filter(id__nin=test_ids)
    export_features(
        candidates_train, candidates_test, options.x_train_filename, options.x_test_filename,
        options.popular_organizations_filename, options.popular_job_title_words_filename,
        options.verbose)
    export_classes(candidates_train, options.y_train_filename)
    export_classes(candidates_test, options.y_test_filename)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    sys.exit(0)
