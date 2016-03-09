"""Base classes for tests."""
import unittest

from featureforge.validate import FeatureFixtureCheckMixin


class BaseFeatureTestCase(unittest.TestCase, FeatureFixtureCheckMixin):
    """Base test class. It skips fuzzy feature testing.

    Attributes:
        feature: Feature instance to test.
        fixtures: tuples defining test objects and expected values.
    """

    feature = None  # Needs to be defined on subclasses
    fixtures = None

    @classmethod
    def setUpClass(cls):
        """Skip testing this very class, because it's a base 'abstract' class."""
        if cls is BaseFeatureTestCase:
            raise unittest.SkipTest("Skip BaseFeatureTestCase tests, it's a base class")
        super(BaseFeatureTestCase, cls).setUpClass()

    def test_fixtures(self):
        """Test fixtures defined in self.fixtures."""
        self.assert_feature_passes_fixture(self.feature, self.fixtures)


def create_obj(attribute_dict, name='myobj'):
    """Create an object() subclass instance with specified attributes.

    Args:
        attribute_dict: dict of attributes and values for the object.
        name: class name for the class implicitly created.

    Returns:
        instance of the newly created class (named @name) with specified attributes.
    """
    return type(name, (), attribute_dict)()
