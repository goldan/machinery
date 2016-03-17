"""Tests for features used for experience classification.

Note that only custom new classes are tested. Features constructed from
the common classes (like AttributeBool) are not tested, because they are
already tested in common.tests.
"""

import unittest
from StringIO import StringIO

from featureforge.validate import EQ, RAISES
from machinery.common.atlas import PROFILE_TYPES_BY_TYPE_ID
from machinery.common.features import AttributeBool, DictKeyInt, AttributeInt
from machinery.common.tests import create_obj as obj
from machinery.common.tests import BaseFeatureTestCase
from machinery.common.tests.features import BaseAttributeTestCase
from machinery.experience.features import (FootprintsFeature,
                                           Gender, JobTitlesWords,
                                           OrganizationNames,
                                           ProfileJobTitlesWords,
                                           SkillsFeature,
                                           candidate_sourcerer_feature,
                                           make_candidate_features,
                                           profile_any_exists_features,
                                           profile_bag_of_words_features,
                                           profile_most_common_features,
                                           profile_sum_features)


def candidate(**kwargs):
    """Create a Candidate mock object with a given profile attributes.

    Args:
        **kwargs: attributes to add to candidate profile.
        If type_id is not specified, the first one in types list is used.

    Returns:
        Candidate mock object with profile having specified attributes.
    """
    type_id = kwargs.pop("type_id", None)
    if type_id is None:
        type_id = PROFILE_TYPES_BY_TYPE_ID.keys()[0]
    return obj(profiles=[obj(type_id=type_id, **kwargs)])


class BaseGenderTestCase(BaseFeatureTestCase):
    """Test gender value evaluation, as a standalone value, not a profile attribute."""

    @classmethod
    def setUpClass(cls):
        """Skip testing this very class, because it's a base 'abstract' class."""
        if cls is BaseGenderTestCase:
            raise unittest.SkipTest("Skip BaseGenderTestCase tests, it's a base class")
        super(BaseGenderTestCase, cls).setUpClass()

    fixtures = {
        'test_male': (u'm', EQ, True),
        'test_female': (u'f', EQ, False),
        'test_other': (u'x', EQ, False),
        'test_empty': ('', EQ, False),
        'test_none': (None, EQ, True),
    }


class GenderTestCase(BaseAttributeTestCase):
    """Test Gender evaluation."""

    feature = Gender()
    base_test_class = BaseGenderTestCase
    default_value = True
    attribute_name = "gender"


class SkillsFeatureTestCase(BaseFeatureTestCase):
    """Test evaluation of profile skills feature."""

    feature = SkillsFeature()
    fixtures = {
        'test_skills': (obj(skills=[obj(points=i) for i in range(10)]),
                        EQ, range(10)),
        'test_skills_empty': (obj(skills=[]), EQ, []),
        'test_skills_no_points': (obj(skills=[obj(other_points=1)]), EQ, []),
        'test_other_attr': (obj(otherattr=[1, 2, 5]), EQ, []),
        'test_none': (None, EQ, []),
    }


class FootprintsFeatureTestCase(BaseFeatureTestCase):
    """Test evaluation of profile digital footprints feature."""

    feature = FootprintsFeature()
    fixtures = {
        'test_base': (obj(digital_footprint_scores=[obj(value=i) for i in range(10)]),
                      EQ, range(10)),
        'test_empty': (obj(digital_footprint_scores=[]), EQ, []),
        'test_no_footprints': (obj(digital_footprint_scores=[obj(other_values=1)]), EQ, []),
        'test_other_attr': (obj(otherattr=[1, 2, 5]), EQ, []),
        'test_none': (None, EQ, []),
    }


class MakeCandidateFeaturesAttributeBoolTestCase(BaseFeatureTestCase):
    """Test evaluation of candidate attribute bool feature."""

    type_id = PROFILE_TYPES_BY_TYPE_ID.keys()[0]
    fixtures = {
        'test_true': (candidate(fake=True), EQ, True),
        'test_false': (candidate(fake=False), EQ, False),
        'test_no': (candidate(), EQ, False),
        'test_empty': (candidate(fake=''), EQ, False),
        'test_other_prof': (candidate(fake=True, type_id=type_id+1), EQ, False),
        'test_other_attr': (candidate(otherattr=True), EQ, False),
        'test_none': (None, EQ, False),
    }

    @property
    def feature(self):
        """Get feature object to test."""
        return make_candidate_features([AttributeBool("fake")])[0]


class OrganizationNamesTestCase(BaseFeatureTestCase):
    """Test evaluation of candidate organization names feature."""

    feature = OrganizationNames(StringIO("\n".join(
        ("google, 150", "facebook, 120", "amazon, 110", "yandex, 50"))))
    fixtures = {
        'test_base': (obj(profiles=[
            obj(type_id=0, organizations=[
                obj(name='Google')]),
            obj(type_id=1, organizations=[
                obj(name='Facebook'),
                obj(name='Yandex')]),
            obj(type_id=2, organizations=None),
            obj(type_id=3)]),
                      EQ, set(["google", "facebook"])),
        'test_other': (obj(profiles=[
            obj(type_id=0, organizations=[
                obj(name='yandex')]),
            obj(type_id=1, organizations=[
                obj(name='ibm'),
                obj(name='trueskills')])]),
                       EQ, set()),
        'test_empty': (obj(profiles=[
            obj(type_id=0, organizations=[]),
            obj(type_id=1)]),
                       EQ, set()),
        'test_other_attr': (obj(otherattr=[1, 2, 5]), EQ, set()),
        'test_none': (None, EQ, set()),
    }


class JobTitlesWordsTestCase(BaseFeatureTestCase):
    """Test evaluation of candidate job titles words feature."""

    feature = JobTitlesWords(StringIO("\n".join(
        ("software, 150", "engineer, 120", "manager, 110", "ceo, 50"))))
    fixtures = {
        'test_base': (obj(profiles=[
            obj(type_id=0, organizations=[
                obj(title='Software Engineers')]),
            obj(type_id=1, organizations=[
                obj(title='Engineer'),
                obj(name='Yandex')]),
            obj(type_id=2, organizations=None),
            obj(type_id=3)]),
                      EQ, set(["software", "engineer"])),
        'test_other': (obj(profiles=[
            obj(type_id=0, organizations=[
                obj(title='CEO')]),
            obj(type_id=1, organizations=[
                obj(title='manager'),
                obj(title='')])]),
                       EQ, set(["manager"])),
        'test_empty': (obj(profiles=[
            obj(type_id=0, organizations=[]),
            obj(type_id=1)]),
                       EQ, set()),
        'test_other_attr': (obj(otherattr=[1, 2, 5]), EQ, set()),
        'test_none': (None, EQ, set()),
    }


class ProfileMostCommonFeaturesTestCase(BaseFeatureTestCase):
    """Test evaluation of profile_most_common_features feature factory."""

    feature = profile_most_common_features([AttributeInt("age")])[0]
    fixtures = {
        'test_base': (obj(profiles=[
            obj(age=20), obj(age=10), obj(age=10)]),
                      EQ, 10),
        'test_other': (obj(profiles=[
            obj(gender='m'), obj(age=10), obj(gender='f')]),
                       EQ, 10),
        'test_empty': (obj(profiles=[obj()]), EQ, 0),
        'test_none': (None, EQ, 0)
    }


class ProfileAnyExistsFeaturesTestCase(BaseFeatureTestCase):
    """Test evaluation of profile_any_exists_features feature factory."""

    feature = profile_any_exists_features([AttributeInt("age")])[0]
    fixtures = {
        'test_base': (obj(profiles=[
            obj(age=20), obj(gender=10), obj(gender=10)]),
                      EQ, True),
        'test_not': (obj(profiles=[
            obj(gender='m'), obj(gender=10), obj(gender='f')]),
                     EQ, False),
        'test_empty': (obj(profiles=[obj()]), EQ, False),
        'test_none': (None, EQ, False)
    }


class ProfileSumFeaturesTestCase(BaseFeatureTestCase):
    """Test evaluation of profile_sum_features feature factory."""

    feature = profile_sum_features([AttributeInt("age")])[0]
    fixtures = {
        'test_base': (obj(profiles=[
            obj(age=20), obj(age=10), obj(age=10)]),
                      EQ, 40),
        'test_not': (obj(profiles=[
            obj(gender='m'), obj(age=10), obj(gender='f')]),
                     EQ, 10),
        'test_one': (obj(profiles=[
            obj(age='m'), obj(age=10), obj(gender='f')]),
                     RAISES, ValueError),
        'test_empty': (obj(profiles=[obj()]), EQ, 0),
        'test_none': (None, EQ, 0)
    }


class ProfileBagOfWordsFeaturesTestCase(BaseFeatureTestCase):
    """Test evaluation of profile_bag_of_words_features feature factory."""

    feature = profile_bag_of_words_features([ProfileJobTitlesWords()])[0]
    fixtures = {
        'test_base': (obj(profiles=[
            obj(organizations=[
                obj(title="software engineer"),
                obj(title="ceo")]),
            obj(organizations=[obj(title="java engineer")])]),
                      EQ, set(["ceo", "software", "java", "engineer"])),
        'test_empty': (obj(profiles=[obj()]), EQ, set()),
        'test_none': (None, EQ, set())
    }


class CandidateSourcererFeatureTestCase(BaseFeatureTestCase):
    """Test evaluation of candidate_sourcerer_feature feature factory."""

    feature = candidate_sourcerer_feature(
        DictKeyInt('age'), {'id1': {'age': 10}, 'id2': {'gender': 'm'}})
    fixtures = {
        'test_base': (obj(id='id1'), EQ, 10),
        'test_other': (obj(id='id2'), EQ, 0),
        'test_none': (None, EQ, 0)
    }


if __name__ == "__main__":
    unittest.main()
