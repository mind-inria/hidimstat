def test_version():
    """Test that the version is set correctly."""
    import hidimstat
    assert isinstance(hidimstat.__version__, str)
    assert len(hidimstat.__version__.split(".")) == 3
