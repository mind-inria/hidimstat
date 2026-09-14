import pytest

from hidimstat._utils.docstring import (
    _aggregate_docstring,
    _detection_section,
    _parse_docstring,
    _reindent,
)

INPUT1 = [
    "",
    "Summary header",
    "",
    "Parameters",
    "----------",
    "lines : list of str",
    "   Lines of the docstring to parse.",
    "",
    "Returns",
    "-------",
    "list of list of str",
    "   blabla",
    "",
    "Notes",
    "-----",
    "Some notes.",
    "",
]
section_indices1 = [(1, 2), (3, 7), (8, 17)]

INPUT2 = [
    "",
    "One liner comment",
    "",
]
section_indices2 = [(1, 3)]

INPUT3 = [
    "",
    "One liner comment",
    "Gotcha, it's got two lines",
]
section_indices3 = [(1, 4)]


@pytest.fixture
def docstring_section(input, section_indices):
    start, end = section_indices[0]
    expected_list = [input[start:end]]

    for i in range(1, len(section_indices)):
        start, end = section_indices[i]
        expected_list.append(input[start:end])

    return input, expected_list


@pytest.fixture
def docstring_dict(input, section_indices):
    start, end = section_indices[0]
    expected_dict = {"short": input[start:end]}

    for i in range(1, len(section_indices)):
        start, end = section_indices[i]
        expected_dict["".join(input[start].split())] = input[start:end]

    return input, expected_dict


@pytest.mark.parametrize(
    "input, section_indices",
    [
        (INPUT1, section_indices1),
        (INPUT2, section_indices2),
        (INPUT3, section_indices3),
    ],
)
def test_smoke_detection_section(docstring_section):
    """
    Smoke test for section detection function
    """
    input_doc, expected_result = docstring_section
    sections = _detection_section(input_doc)

    assert expected_result == sections


def test_detection_section_empty_input():
    """No lines at all should not raise, and yields a single empty section."""
    assert _detection_section([]) == [[]]


def test_detection_section_single_line():
    """A single-line input (only the blank line[0] slot) yields an empty section."""
    assert _detection_section(["only one line"]) == [[]]


def test_detection_section_no_section_headers():
    """A docstring with no '-------' underline anywhere is treated as one section."""
    lines = ["", "Just a summary", "with two lines", ""]
    result = _detection_section(lines)
    assert result == [lines[1:4]]


@pytest.mark.parametrize(
    "input, section_indices",
    [
        (INPUT1, section_indices1),
        (INPUT2, section_indices2),
        (INPUT3, section_indices3),
    ],
)
def test_smoke_parse_docstring(docstring_dict):
    """
    Smoke test for docstring parsing function
    """
    input_doc, expected_result = docstring_dict

    sections = _parse_docstring("\n".join(input_doc))
    assert sections == expected_result


def test_parse_docstring_no_sections_only_short():
    """A docstring with no recognized section falls back to only 'short'."""
    doc = "\nJust a summary\nNo sections here\n"
    result = _parse_docstring(doc)
    assert list(result.keys()) == ["short"]
    assert result["short"] == ["Just a summary", "No sections here", ""]


def test_parse_docstring_key_strips_internal_whitespace():
    """Section header words are concatenated (whitespace removed) to form the dict key."""
    doc = "\nSummary header\n\nOther Section\n----------\ncontent line\n"
    result = _parse_docstring(doc)
    assert "OtherSection" in result
    assert result["OtherSection"] == [
        "Other Section",
        "----------",
        "content line",
        "",
    ]


def test_smoke_reindent():
    doc_test = [
        "   ",
        "   Short summary  ",
        " ",
        "   Parameters ",
        "   ---------- ",
        "   str",
        "     some input",
        "",
    ]
    output = _reindent(doc_test)
    expected = "\nShort summary\n\nParameters\n----------\nstr\nsome input\n"
    assert output == expected


def test_reindent_empty_list():
    assert _reindent([]) == ""


def test_reindent_embedded_newline_with_trailing_blank_line():
    """With a trailing '' element (as real callers provide), no content is lost."""
    lines = ["  hello  ", "world\n  foo  ", "  bar"]
    assert _reindent(lines) == "hello\nworld\nfoo\nbar"


def test_smoke_aggregate_docstring():
    """
    Smoke test for docstring aggregation function
    """
    doc_minimum_1 = """
    Short Summary

    Parameters
    ----------
    param_1: ndarray of shape (n_sampling,)
        short description

    param_2: float, default=2.0
        short description 2

    Returns
    -------
    ndarray of shape (n_sampling,)
        short description for return

    References
    ----------
    .. footbibliography::

    Notes
    -----

    Complementary information
    """
    doc_minimum_2 = """
    Name object

    Description

    Parameters
    ----------
    param2_1: float, default=2.0
        description param2_1

    param2_2: ndarray of shape (n_sampling,)
        description param2_2

    Returns
    -------
    time_example (float)
        description of return
    """
    doc_minimum_3 = """
    Short Description

    Parameters
    ----------
    param_first: int, default=10
        integer interpretation

    param_second: float, default=2.0
        float interpretation

    Returns
    -------
    None


    References
    ----------
    .. footbibliography::

    """
    final_doc = _aggregate_docstring(
        [doc_minimum_1, None, doc_minimum_2, doc_minimum_3],
        """
        Returns
        -------
        3D ndarray (n_tests, )
        Vector of aggregated p-values
        """,
    )
    assert (
        final_doc
        == """Short Summary
Parameters
----------
param_1: ndarray of shape (n_sampling,)
short description

param_2: float, default=2.0
short description 2
param2_1: float, default=2.0
description param2_1

param2_2: ndarray of shape (n_sampling,)
description param2_2
param_first: int, default=10
integer interpretation

param_second: float, default=2.0
float interpretation

Returns
-------
3D ndarray (n_tests, )
Vector of aggregated p-values
"""
    )


_RETURNS_TRAILING_NEWLINE = """
        Returns
        -------
        final return
        """


def test_aggregate_docstring_missing_parameters_section_raises():
    """A docstring lacking a 'Parameters' section raises KeyError."""
    doc_no_params = """
    Summary only, no parameters section

    Returns
    -------
    None
    """
    with pytest.raises(KeyError):
        _aggregate_docstring([doc_no_params], _RETURNS_TRAILING_NEWLINE)


def test_aggregate_docstring_all_none_raises():
    """If every docstring in the list is None, there's nothing to build the summary from."""
    with pytest.raises(IndexError):
        _aggregate_docstring([None, None], _RETURNS_TRAILING_NEWLINE)
