from copy import deepcopy


def _detection_section(lines):
    """
    Detect sections in a numpy-style docstring by identifying section headers and their underlines.

    Parameters
    ----------
    lines : list of str
        Lines of the docstring to parse.

    Returns
    -------
    list of list of str
        List of sections, where each section is a list of lines belonging to that section.
        The first section is the summary, followed by other sections like Parameters, Returns, etc.
    """
    sections = []
    index_line = 1
    begin_section = index_line
    while len(lines) > index_line:
        if "-------" in lines[index_line]:
            sections.append(lines[begin_section : index_line - 2])
            begin_section = index_line - 1
        index_line += 1
    sections.append(lines[begin_section : len(lines)])
    return sections


def _parse_docstring(docstring):
    """
    Parse a numpy-style docstring into its component sections.

    Parameters
    ----------
    docstring : str
        The docstring to parse, following numpy docstring format.

    Returns
    -------
    dict
        Dictionary containing docstring sections with keys like 'short' (summary),
        'Parameters', 'Returns', etc. Values are the text content of each section.
    """
    lines = docstring.split("\n")
    section_texts = _detection_section(lines)
    sections = {"short": section_texts[0]}
    for section_text in section_texts:
        if len(section_text) <= 1 or "---" not in section_text[1]:
            sections["short"] = section_text
        else:
            sections["".join(section_text[0].split())] = section_text
    return sections


def _reindent(string):
    """
    Reindent a string by stripping whitespace and normalizing line breaks.

    Parameters
    ----------
    string : list of str
        The string content to reindent.

    Returns
    -------
    str
        Reindented string with normalized line breaks and indentation.
    """
    new_string = deepcopy(string)
    for i in range(len(new_string)):
        new_string[i] = "\n" + new_string[i]
    new_string = "".join(new_string)
    return "\n".join(l.strip() for l in new_string.strip().split("\n"))


def _aggregate_docstring(list_docstring, returns_docstring):
    """
    Combine multiple docstrings into a single docstring.

    This function takes a list of docstrings, parses each one, and combines them into
    a single coherent docstring. It keeps the summary from the first docstring,
    combines all parameter sections, and uses the return section from the last docstring.

    Parameters
    ----------
    list_docstring : list
        List of docstrings to be combined. Each docstring should follow
        numpy docstring format.

    Returns
    -------
    doctring: str
        A combined docstring containing:
        - Summary from first docstring
        - Combined parameters from all docstrings
        - Returns section from last docstring
        The returned docstring is properly reindented.
    """
    list_line = []
    for index, docstring in enumerate(list_docstring):
        if docstring is not None:
            list_line.append(_parse_docstring(docstring=docstring))

    # add summary
    final_docstring = deepcopy(list_line[0]["short"])
    # add parameter
    final_docstring += list_line[0]["Parameters"]
    for i in range(1, len(list_line)):
        # add paraemter after remove the title section
        final_docstring += list_line[i]["Parameters"][2:]
    # the last return
    final_docstring += returns_docstring
    return _reindent(final_docstring)
