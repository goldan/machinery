"""Tests for common features."""
import unittest

from featureforge.validate import EQ, RAISES
from machinery.common.features import (AttributeBool, AttributeInt,
                                       AttributeLen, AttributeString,
                                       BoolFeature, Exists, IntFeature,
                                       LenFeature, MaxFeature, SetFeature,
                                       StringFeature, SumFeature)
from machinery.common.tests import BaseFeatureTestCase


class StringFeatureTestCase(BaseFeatureTestCase):
    """Test StringFeature evaluation."""

    feature = StringFeature()
    fixtures = {
        'test_str': (u'abc', EQ, u'abc'),
        'test_bool': (True, EQ, u'True'),
        'test_int': (5, EQ, '5'),
        'test_empty': ('', EQ, u''),
        'test_none': (None, EQ, u''),
    }


class BoolFeatureTestCase(BaseFeatureTestCase):
    """Test BoolFeature evaluation."""

    feature = BoolFeature()
    fixtures = {
        'test_str': (u'abc', EQ, True),
        'test_bool': (True, EQ, True),
        'test_bool_false': (False, EQ, False),
        'test_int': (5, EQ, True),
        'test_empty': ('', EQ, False),
        'test_none': (None, EQ, False),
    }


class IntFeatureTestCase(BaseFeatureTestCase):
    """Test IntFeature evaluation."""

    feature = IntFeature()
    fixtures = {
        'test_str': (u'abc', RAISES, ValueError),
        'test_bool': (True, EQ, 1),
        'test_int': (5, EQ, 5),
        'test_empty': ('', RAISES, ValueError),
        'test_none': (None, EQ, 0),
    }


class LenFeatureTestCase(BaseFeatureTestCase):
    """Test LenFeature evaluation."""

    feature = LenFeature()
    fixtures = {
        'test_str': (u'abc', EQ, 3),
        'test_bool': (True, RAISES, TypeError),
        'test_int': (5, RAISES, TypeError),
        'test_empty': ('', EQ, 0),
        'test_none': (None, EQ, 0),
        'test_list': ([1, 2, 5], EQ, 3),
        'test_list_empty': ([], EQ, 0),
    }


class MaxFeatureTestCase(BaseFeatureTestCase):
    """Test MaxFeature evaluation."""

    feature = MaxFeature()
    fixtures = {
        'test_str': (u'abc', RAISES, ValueError),
        'test_bool': (True, RAISES, TypeError),
        'test_int': (5, RAISES, TypeError),
        'test_empty': ('', EQ, 0),
        'test_none': (None, EQ, 0),
        'test_list': ([1, 2, 5], EQ, 5),
        'test_list_empty': ([], EQ, 0),
    }


class SumFeatureTestCase(BaseFeatureTestCase):
    """Test SumFeature evaluation."""

    feature = SumFeature()
    fixtures = {
        'test_str': (u'abc', RAISES, TypeError),
        'test_bool': (True, RAISES, TypeError),
        'test_int': (5, RAISES, TypeError),
        'test_empty': ('', EQ, 0),
        'test_none': (None, EQ, 0),
        'test_list': ([1, 2, 5], EQ, 8),
        'test_list_empty': ([], EQ, 0),
    }


class SetFeatureTestCase(BaseFeatureTestCase):
    """Test SetFeature evaluation."""

    feature = SetFeature()
    fixtures = {
        'test_str': (u'abc', EQ, set(['a', 'b', 'c'])),
        'test_bool': (True, RAISES, TypeError),
        'test_int': (5, RAISES, TypeError),
        'test_empty': ('', EQ, set()),
        'test_none': (None, EQ, set()),
        'test_list': ([1, 2, 5], EQ, set([1, 2, 5])),
        'test_list_empty': ([], EQ, set()),
    }


class ExistsTestCase(BaseFeatureTestCase):
    """Test Exists evaluation."""

    feature = Exists()
    fixtures = {
        'test_str': (u'abc', EQ, True),
        'test_bool': (True, EQ, True),
        'test_bool_false': (False, EQ, False),
        'test_int': (5, EQ, True),
        'test_empty': ('', EQ, False),
        'test_none': (None, EQ, False),
    }


class BaseAttributeTestCase(BaseFeatureTestCase):
    """Base class for testing AttributeFeature subclasses.

    Attributes:
        base_test_class: TestCase subclass with tests
            for value type of the attribute.
            E.g. if attribute is supposed to contain int,
            then base_test_class will be IntFeatureTestCase.
            It is used to reuse fixtures from.
        default_value: value that the feature is expected to have
            if attribute does not exist or object is None.
    """

    base_test_class = None
    default_value = None

    @classmethod
    def setUpClass(cls):
        """Skip testing this very class, because it's a base 'abstract' class."""
        if cls is BaseAttributeTestCase:
            raise unittest.SkipTest("Skip BaseAttributeTestCase tests, it's a base class")
        super(BaseAttributeTestCase, cls).setUpClass()

    @property
    def fixtures(self):
        """Generate fixtures for testing.

        * take fixtures of the base_test_class and test them against an object
            having a single attribute and values from the fixtures.
        * take the same fixtures and test against an object having
            a different attribute and values from the fixtures.
        * test against None object.

        Returns:
            dictionary of tuples with fixtures.
        """
        fixs = {}
        for key, values in self.base_test_class.fixtures.items():
            fixs[key + "_attr"] = (
                type('myobj', (), {'myattr': values[0]})(),
                values[1], values[2])
            fixs[key + "_other_attr"] = (
                type('myobj', (), {'other_attr': values[0]})(),
                EQ, self.default_value)
            fixs['test_none'] = (None, EQ, self.default_value)
        return fixs


class AttributeStringTestCase(BaseAttributeTestCase):
    """Test AttributeString evaluation."""

    feature = AttributeString("myattr")
    base_test_class = StringFeatureTestCase
    default_value = ''


class AttributeBoolTestCase(BaseAttributeTestCase):
    """Test AttributeBool evaluation."""

    feature = AttributeBool("myattr")
    base_test_class = BoolFeatureTestCase
    default_value = False


class AttributeIntTestCase(BaseAttributeTestCase):
    """Test AttributeInt evaluation."""

    feature = AttributeInt("myattr")
    base_test_class = IntFeatureTestCase
    default_value = 0


class AttributeLenTestCase(BaseAttributeTestCase):
    """Test AttributeLen evaluation."""

    feature = AttributeLen("myattr")
    base_test_class = LenFeatureTestCase
    default_value = 0


if __name__ == "__main__":
    unittest.main()
