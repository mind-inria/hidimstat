from hidimstat._utils.utils import get_fitted_attributes


def test_generated_attributes():
    """Test function for getting generated attribute"""

    class MyClass:
        def __init__(self):
            self.attr1 = 1
            self.attr2_ = 2
            self._attr3 = 3
            self.attr4__ = 4
            self.attr5_ = 5

    attributes = get_fitted_attributes(MyClass())
    assert attributes == ["attr2_", "attr5_"]
