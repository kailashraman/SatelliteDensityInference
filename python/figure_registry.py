"""The figure-naming contract: which module produces each paper figure.

`FIGURES` and `TABLES` are derived by parsing drafts_temp/*.tex for
`\\includegraphics{...}` and `\\input{...}`, in the order they first appear
across the two drafts (temp_part1.tex then temp_part2.tex). The
`SatGen_take2/` prefix on `\\includegraphics` paths is the paper build tree,
not a directory in this repository, so only the basename is kept; dwarf
names contain spaces and are kept verbatim, not sanitised.

`REGISTRY` and `TABLE_REGISTRY` map each basename to the python/ module that
produces it. No plot scripts exist yet (that is steps 10-22), so every entry
is None here -- this module is the single list those steps fill in, one
entry at a time, not a working dispatcher.
"""

import re

import config

_INCLUDEGRAPHICS = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}')
_INPUT = re.compile(r'\\input\{([^}]*)\}')


def _parse_tex_names():
    """Basenames referenced by \\includegraphics and \\input, first-seen order.

    Does NOT strip LaTeX `%` comments before matching, so a commented-out
    `\\includegraphics{...}`/`\\input{...}` line would still register its
    basename as required. No line in either draft is commented out like that
    today (verified), so this is a documented limitation, not a live bug.
    """
    figures = []
    tables = []
    for tex_path in sorted(config.DRAFTS_DIR.glob('*.tex')):
        text = tex_path.read_text()
        for match in _INCLUDEGRAPHICS.finditer(text):
            name = match.group(1).rsplit('/', 1)[-1]
            if name not in figures:
                figures.append(name)
        for match in _INPUT.finditer(text):
            name = match.group(1).rsplit('/', 1)[-1]
            if name not in tables:
                tables.append(name)
    return figures, tables


# pdf basenames (with extension) and table basenames (without -- \input in
# these drafts never carries one), in first-seen order across the two drafts.
FIGURES, TABLES = _parse_tex_names()

# basename -> producing python/<module>.py name, or None until that step lands.
REGISTRY = {name: None for name in FIGURES}
TABLE_REGISTRY = {name: None for name in TABLES}
