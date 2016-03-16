"""Tests for common features."""
import unittest

from featureforge.validate import EQ, RAISES
from machinery.common.features import (AttributeBool, AttributeInt,
                                       AttributeLen, AttributeString,
                                       BoolFeature, DictKeyBool, DictKeyInt,
                                       DictKeyLen, DictKeyString, Exists,
                                       IntFeature, LenFeature, ListFeature,
                                       MaxFeature, SetFeature, StringFeature,
                                       SubAttributeSet, SubAttributeString,
                                       SumFeature,
                                       make_list_of_values_features)
from machinery.common.tests import BaseFeatureTestCase, create_obj


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
        'test_list': ([1, 2, 5, 0], EQ, 4),
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
        'test_list': ([1, 2, 5, 0], EQ, 5),
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
        'test_list': ([1, 2, 5, 0], EQ, 8),
        'test_list_empty': ([], EQ, 0),
    }


class ListFeatureTestCase(BaseFeatureTestCase):
    """Test ListFeature evaluation."""

    feature = ListFeature()
    fixtures = {
        'test_str': (u'abc', EQ, ['a', 'b', 'c']),
        'test_bool': (True, RAISES, TypeError),
        'test_int': (5, RAISES, TypeError),
        'test_empty': ('', EQ, []),
        'test_none': (None, EQ, []),
        'test_tuple': ((1, 2, 5), EQ, [1, 2, 5]),
        'test_tuple_empty': ([], EQ, []),
        'test_list': ([1, 2, 5], EQ, [1, 2, 5]),
    }


class SetFeatureTestCase(BaseFeatureTestCase):
    """Test SetFeature evaluation."""

    feature = SetFeature()
    fixtures = {
        'test_str': (u'abc', EQ, set(['abc'])),
        'test_bool': (True, EQ, set([True])),
        'test_int': (5, EQ, set([5])),
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


class BaseDictKeyTestCase(BaseFeatureTestCase):
    """Base class for testing DictKeyFeature subclasses.

    Attributes:
        base_test_class: TestCase subclass with tests
            for value type of the dict value.
            E.g. if dict is supposed to contain int,
            then base_test_class will be IntFeatureTestCase.
            It is used to reuse fixtures from.
        default_value: value that the feature is expected to have
            if key does not exist or object is None.
        key: dict key for fixtures.
        other_key: some other key to check that only specified key is taken.
    """

    base_test_class = None
    default_value = None
    key = "mykey"
    other_key = "other_key"

    @classmethod
    def setUpClass(cls):
        """Skip testing this very class, because it's a base 'abstract' class."""
        if cls is BaseDictKeyTestCase:
            raise unittest.SkipTest("Skip BaseDictKeyTestCase tests, it's a base class")
        super(BaseDictKeyTestCase, cls).setUpClass()

    @property
    def fixtures(self):
        """Generate fixtures for testing.

        * take fixtures of the base_test_class and test them against an dict
            having a single key and values from the fixtures.
        * take the same fixtures and test against a dict having
            a different key and values from the fixtures.
        * test against None object.

        Returns:
            dictionary of tuples with fixtures.
        """
        fixs = {}
        for key, values in self.base_test_class.fixtures.items():
            fixs[key] = ({self.key: values[0]}, values[1], values[2])
            fixs[key + "_other"] = ({self.other_key: values[0]},
                                    EQ, self.default_value)
            fixs['test_none'] = (None, EQ, self.default_value)
        return fixs


class DictKeyStringTestCase(BaseDictKeyTestCase):
    """Test DictKeyString evaluation."""

    feature = DictKeyString("mykey")
    base_test_class = StringFeatureTestCase
    default_value = ''


class DictKeyBoolTestCase(BaseDictKeyTestCase):
    """Test DictKeyBool evaluation."""

    feature = DictKeyBool("mykey")
    base_test_class = BoolFeatureTestCase
    default_value = False


class DictKeyIntTestCase(BaseDictKeyTestCase):
    """Test DictKeyInt evaluation."""

    feature = DictKeyInt("mykey")
    base_test_class = IntFeatureTestCase
    default_value = 0


class DictKeyLenTestCase(BaseDictKeyTestCase):
    """Test DictKeyLen evaluation."""

    feature = DictKeyLen("mykey")
    base_test_class = LenFeatureTestCase
    default_value = 0


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
        attribute_name: name of the attribute for fixtures.
        other_attribute_name: some other name of an attribute
            to check that only specified attribute is taken.
    """

    base_test_class = None
    default_value = None
    attribute_name = "myattr"
    other_attribute_name = "other_attr"

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
                create_obj(**{self.attribute_name: values[0]}),
                values[1], values[2])
            fixs[key + "_attr_other"] = (
                create_obj(**{self.other_attribute_name: values[0]}),
                EQ, self.default_value)
            fixs['test_none'] = (None, EQ, self.default_value)
        return fixs


class BaseSubAttributeTestCase(BaseAttributeTestCase):
    """Base class for testing SubAttributeFeature subclasses.

    Attributes:
        subattribute_name: name of the subattribute for fixtures.
        other_subattribute_name: some other name of a subattribute
            to check that only specified subattribute is taken.
    """

    subattribute_name = "mysubattr"
    other_subattribute_name = "other_subattr"

    @classmethod
    def setUpClass(cls):
        """Skip testing this very class, because it's a base 'abstract' class."""
        if cls is BaseSubAttributeTestCase:
            raise unittest.SkipTest("Skip BaseSubAttributeTestCase tests, it's a base class")
        super(BaseSubAttributeTestCase, cls).setUpClass()

    @property
    def fixtures(self):
        """Generate fixtures for testing.

        * take fixtures of the base_test_class and test them against an object
            having a single attribute and subattribute and values from the fixtures.
        * take the same fixtures and test against an object having
            the same attribute but a different subattribute and values from the fixtures.
        * take the same fixtures and test against an object having
            a different attribute and values from the fixtures.
        * test against None object.

        Returns:
            dictionary of tuples with fixtures.
        """
        fixs = {}
        for key, values in self.base_test_class.fixtures.items():
            fixs[key + "_attr"] = (
                create_obj(**{self.attribute_name:
                              create_obj(_clsname='mysubobj',
                                         **{self.subattribute_name: values[0]})}),
                values[1], values[2])
            fixs[key + "_attr_other_sub"] = (
                create_obj(**{self.attribute_name:
                              create_obj(_clsname='mysubobj',
                                         **{self.other_subattribute_name: values[0]})}),
                EQ, self.default_value)
            fixs[key + "_attr_other"] = (
                create_obj(**{self.other_attribute_name: values[0]}),
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


class SubAttributeStringTestCase(BaseSubAttributeTestCase):
    """Test SubAttributeString evaluation."""

    feature = SubAttributeString("myattr", "mysubattr")
    base_test_class = StringFeatureTestCase
    default_value = ''


class SubAttributeSetTestCase(BaseSubAttributeTestCase):
    """Test SubAttributeSet evaluation."""

    feature = SubAttributeSet("myattr", "mysubattr")
    base_test_class = SetFeatureTestCase
    default_value = set()


class ListOfValuesFeaturesLenTestCase(LenFeatureTestCase):
    """Test list of values feature len evaluation."""

    feature = make_list_of_values_features(ListFeature)[0]


class ListOfValuesFeaturesMaxTestCase(MaxFeatureTestCase):
    """Test list of values feature max evaluation."""

    feature = make_list_of_values_features(ListFeature)[1]


class ListOfValuesFeaturesSumTestCase(SumFeatureTestCase):
    """Test list of values feature sum evaluation."""

    feature = make_list_of_values_features(ListFeature)[2]


if __name__ == "__main__":
    unittest.main()
