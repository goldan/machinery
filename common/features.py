"""Common Feature objects used to create specific features."""
from featureforge.feature import Feature
from schema import Schema


class EvaluationError(Exception):
    """Feature evaluation failed, return default feature value."""

    pass


class BaseFeature(Feature):
    """Base class for all features.

    Attributes:
        value_type: type instance indicating feature value (e.g. int/bool).
        default: default value for the feature, used when
            feature evaluation fails.
        output_schema: Schema instance indicating expected type of feature values.
    """

    value_type = None
    default = None
    output_schema = None

    def __init__(self, name=None):
        """Set output_schema according to feature value_type.

        Args:
            name: feature name to be set.
        """
        if self.value_type:
            self.output_schema = Schema(self.value_type)
        if name:
            self._name = name

    def _preprocess(self, data_point):
        """Feature evaluation step 1/4: preprocess data_point.

        Args:
            data_point: data point whose feature is evaluated.

        Returns:
            preprocessed data point value.
        """
        return data_point

    def _process(self, data_point):
        """Feature evaluation step 2/4: process data_point.

        Args:
            data_point: data point whose feature is evaluated.

        Returns:
            data point feature value.
        """
        return data_point

    def _postprocess(self, feature_value):
        """Feature evaluation step 3/4: postprocess feature value.

        Args:
            feature_value: feature evaluated value.

        Returns:
            data point feature value.
        """
        return feature_value

    def _evaluate(self, data_point):
        """Evaluate feature value for the data point.

        Steps:
            - preprocess data point.
            - process data point and return feature value.
            - postprocess feature value and return it.
            - cast value type to self.value_type.

        Args:
            data_point: data point whose feature is evaluated.

        Returns:
            data point feature value.

        Raises:
            EvaluationError, if (OR):
            - data point is None;
            - any of _preprocess/_process/_postprocess methods return None.
        """
        try:
            if data_point is None:
                raise EvaluationError
            prep = self._preprocess(data_point)
            if prep is None:
                raise EvaluationError
            value = self._process(prep)
            if value is None:
                raise EvaluationError
            post = self._postprocess(value)
            if post is None:
                raise EvaluationError
            value = self.value_type(post)
        except EvaluationError:
            value = self.default
        return value


class StringFeature(BaseFeature):
    """Feature which output value is unicode."""

    value_type = unicode
    default = u''


class BoolFeature(BaseFeature):
    """Feature which output value is boolean."""

    value_type = bool
    default = False


class IntFeature(BaseFeature):
    """Feature which output value is integer."""

    value_type = int
    default = 0


class LenFeature(IntFeature):
    """Feature which output value is length of data point."""

    def _postprocess(self, data_point):
        """Feature evaluation step 3/4: postprocess feature value.

        It's done in postprocess, so that it can be combined with
        classes overriding process, e.g. AttributeFeature.

        Args:
            feature_value: feature evaluated value.

        Returns:
            length of data point.
        """
        return len(data_point)


class MaxFeature(IntFeature):
    """Feature which output value is max value of (an iterable) data point."""

    def _process(self, data_point):
        """Feature evaluation step 2/4: process data_point.

        Args:
            data_point: iterable data point whose feature is evaluated.

        Returns:
            max value of (iterable) data point.
        """
        try:
            return max(data_point)
        except ValueError:
            raise EvaluationError


class SumFeature(IntFeature):
    """Feature which output value is sum of (an iterable) data point."""

    def _process(self, data_point):
        """Feature evaluation step 2/4: process data_point.

        Args:
            data_point: iterable data point whose feature is evaluated.

        Returns:
            sum of (iterable) data point.
        """
        return sum(data_point)


class SetFeature(BaseFeature):
    """Feature which output value is a set."""

    value_type = set
    default = set()


class Exists(BoolFeature):
    """Feature which checks that data_point evaluates to True."""

    _name = "exists"


class AttributeFeature(BaseFeature):
    """Base class for features evaluating object's attribute.

    It is not meant to be instantiating directly, but only subclassed.
    """

    attribute_name = None

    def __init__(self, attribute_name=None):
        """Set attribute name and _name from it."""
        super(AttributeFeature, self).__init__()
        if attribute_name:
            self.attribute_name = attribute_name
        self._name = self.attribute_name

    def _process(self, data_point):
        """Get data_point object's attribute value.

        Args:
            data_point: object whose feature is evaluated.

        Returns:
            attribute value of the object.

        Raises:
            EvaluationError, if the object does not have the attribute.
        """
        try:
            return getattr(data_point, self.attribute_name)
        except AttributeError:
            raise EvaluationError


class AttributeString(AttributeFeature, StringFeature):
    """Feature that evaluates object's attribute and outputs a string."""

    pass


class AttributeBool(AttributeFeature, BoolFeature):
    """Feature that evaluates object's attribute and outputs a boolean."""

    pass


class AttributeInt(AttributeFeature, IntFeature):
    """Feature that evaluates object's attribute and outputs an integer."""

    pass


class AttributeLen(AttributeFeature, LenFeature):
    """Feature that evaluates object's attribute and outputs an length of it."""

    pass
