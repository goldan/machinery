"""Feature classes for Candidate experience classification."""
import copy
import string
from itertools import product
from types import MethodType

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from machinery.common.atlas import (GENDER_MALE, PROFILE_TYPES_BY_TYPE_ID,
                                    Organization)
from machinery.common.features import (AttributeBool, AttributeInt,
                                       AttributeLen, BoolFeature, ListFeature,
                                       SetFeature, SubAttributeString,
                                       make_list_of_values_features)


class Gender(AttributeBool):
    """Gender feature. 1 for male, 0 for women (no sexism)."""

    attribute_name = "gender"

    def _postprocess(self, gender):
        """Convert gender code to binary format."""
        return gender == GENDER_MALE


class PhotoExists(BoolFeature):
    """Feature indicating if any photos exist for the profile."""

    def _postprocess(self, profile):
        """Return True if any photo exists for the profile.

        Args:
            profile to evaluate.

        Returns:
            True if profile.profile_pic_url exists or profile.photos is not empty.
        """
        photos = AttributeLen("photos")._evaluate(profile)
        return getattr(profile, "profile_pic_url", None) or photos


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


class LocationExists(BoolFeature):
    """Feature indicating if location exists for the profile."""

    def _postprocess(self, profile):
        """Return True if location or location_raw exist for the profile.

        Args:
            profile to evaluate.

        Returns:
            True if profile.location or profile.location_raw exist.
        """
        return getattr(profile, "location", None) or getattr(profile, "location_raw", None)


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

    def __init__(self, organizations):
        """Store organizations to search for."""
        super(OrganizationNames, self).__init__()
        self.organizations = organizations

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

    def __init__(self, job_titles_words):
        """Store job titles words to search for."""
        super(JobTitlesWords, self).__init__()
        self.job_titles_words = job_titles_words

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

    def _postprocess(self, candidate, profile_feature=profile_feature, type_id=profile_type_id):
        """Evaluate the candidate feature using underlying profile feature.

        Args:
            candidate: data point to evaluate.
            profile_feature: profile feature used to evaluate the profile.
            type_id: type id of the profile to be evaluated.

        Returns:
            evaluation of the feature for the given profile.
        """
        profiles = dict((profile.type_id, profile) for profile in candidate.profiles
                        if hasattr(profile, "type_id"))
        profile = profiles.get(type_id)
        return profile_feature._evaluate(profile)

    feature._postprocess = MethodType(_postprocess, feature)
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


def get_features(popular_organizations, popular_job_titles_words):
    """Get a list of all features for candidates.

    It includes:
        * candidate-level features (years_experience)
        * profile-level features (first_name)
        * features aggregated from profiles (organization names, job titles words).

    Args:
        popular_organizations: list of (normalized) organization names.
        popular_job_titles_words: list of (normalized) words that
            occur in job titles.

    Returns:
        tuple of all candidates feature instances.
    """
    candidates_base_features = [AttributeInt("years_experience")]

    profile_base_features = [
        BoolFeature("exists"),
        AttributeBool("fake"),
        AttributeBool("deleted"),
        AttributeBool("first_name"),
        AttributeBool("last_name"),
        AttributeBool("full_name"),
        AttributeInt("age"),
        Gender(),
        PhotoExists(),
        AttributeBool("bio"),
        AttributeBool("url"),
        LocationExists(),
        SubAttributeString("location", "country"),
        SubAttributeString("location", "state"),
        SubAttributeString("location", "city"),
        AttributeLen("organizations"),
        AttributeLen("websites"),
        AttributeLen("chats"),
        AttributeInt("followers"),
        AttributeInt("following")]

    profile_features = (profile_base_features +
                        make_list_of_values_features(SkillsFeature) +
                        make_list_of_values_features(FootprintsFeature))

    candidate_profile_features = make_candidate_features(profile_features)

    candidate_aggregated_features = [
        OrganizationNames(popular_organizations),
        JobTitlesWords(popular_job_titles_words)]

    features = (candidates_base_features +
                candidate_profile_features +
                candidate_aggregated_features)

    return features
