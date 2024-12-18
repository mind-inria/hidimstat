import hidimstat

def test_version():
    """Test that the version is set correctly."""
    assert isinstance(hidimstat.__version__, str)
    assert len(hidimstat.__version__.split(".")) == 3
