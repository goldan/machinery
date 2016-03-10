"""Tests for features used for experience classification.

Note that only custom new classes are tested. Features constructed from
the common classes (like AttributeBool) are not tested, because they are
already tested in common.tests.
"""

import unittest

from featureforge.validate import EQ

from atlas_project.candidate.profile_defs import PROFILE_TYPES_BY_TYPE_ID
from machinery.common.tests import create_obj as obj
from machinery.common.tests import BaseFeatureTestCase
from machinery.common.tests.features import BaseAttributeTestCase
from machinery.experience.features import (FootprintsFeature, Gender,
                                           JobTitlesWords, LocationExists,
                                           OrganizationNames, PhotoExists,
                                           SkillsFeature,
                                           make_candidate_features)


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
        'test_none': (None, EQ, False),
    }


class GenderTestCase(BaseAttributeTestCase):
    """Test Gender evaluation."""

    feature = Gender()
    base_test_class = BaseGenderTestCase
    default_value = False
    attribute_name = "gender"


class PhotoExistsTestCase(BaseFeatureTestCase):
    """Test evaluation of profile photo exists feature."""

    feature = PhotoExists()
    url = 'http://google.com'
    fixtures = {
        'test_picurl_exists': (obj(profile_pic_url=url), EQ, True),
        'test_picurl_empty': (obj(profile_pic_url=''), EQ, False),
        'test_photos_exist': (obj(photos=[obj(url=url, website='gravatar')]),
                              EQ, True),
        'test_photos_empty': (obj(photos=[]), EQ, False),
        'test_other_attr': (obj(otherattr=url), EQ, False),
        'test_none': (None, EQ, False),
    }


class LocationExistsTestCase(BaseFeatureTestCase):
    """Test evaluation of profile location exists feature."""

    feature = LocationExists()
    fixtures = {
        'test_location_exists': (obj(location=obj(city='New York')),
                                 EQ, True),
        'test_location_empty': (obj(location=''), EQ, False),
        'test_location_raw_exist': (obj(location_raw='New York, NY, USA'),
                                    EQ, True),
        'test_location_raw_empty': (obj(location_raw=''), EQ, False),
        'test_other_attr': (obj(otherattr='New York'), EQ, False),
        'test_none': (None, EQ, False),
    }


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


class MakeCandidateFeaturesLocationExistsTestCase(BaseFeatureTestCase):
    """Test evaluation of candidate location exists feature."""

    type_id = PROFILE_TYPES_BY_TYPE_ID.keys()[0]
    feature = make_candidate_features([LocationExists()])[0]
    fixtures = {
        'test_location_exists': (candidate(location=obj(city='New York')),
                                 EQ, True),
        'test_location_exists_other_profile': (
            candidate(location=obj(city='New York'), type_id=type_id+1), EQ, False),
        'test_location_empty': (candidate(location=''), EQ, False),
        'test_location_raw_exist': (candidate(location_raw='New York, NY, USA'),
                                    EQ, True),
        'test_location_raw_exist_other_profile': (
            candidate(location_raw='New York, NY, USA', type_id=type_id+1), EQ, False),
        'test_location_raw_empty': (candidate(location_raw=''), EQ, False),
        'test_other_attr': (candidate(otherattr='New York'), EQ, False),
        'test_none': (None, EQ, False),
    }


class OrganizationNamesTestCase(BaseFeatureTestCase):
    """Test evaluation of candidate organization names feature."""

    feature = OrganizationNames(set(["google", "facebook", "amazon"]))
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

    feature = JobTitlesWords(set(["software", "engineer", "manager"]))
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


if __name__ == "__main__":
    unittest.main()
