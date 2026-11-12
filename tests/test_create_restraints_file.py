"""
Tests for scripts/post/create_restraints_file.py, which writes the
phenix.refine occupancy constraints (qFit_occupancy.params) used by
qfit_final_refine_xray.sh.

The script is run from the source tree rather than from PATH, because
`pip install -e .` installs a copy of it that can be stale.
"""

import os.path as op
import re
import subprocess
import sys

import iotbx.pdb
import iotbx.phil

SCRIPT = op.join(
    op.dirname(op.dirname(op.abspath(__file__))),
    "scripts",
    "post",
    "create_restraints_file.py",
)

BACKBONE = ("N", "CA", "C", "O")


def _pdb_line(serial, name, altloc, resn, chain, resseq, icode, xyz, occ, record="ATOM"):
    x, y, z = xyz
    return (
        f"{record:<6}{serial:5d}  {name:<3}{altloc:1}{resn:>3} {chain:1}"
        f"{resseq:4d}{icode:1}   {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{20.0:6.2f}"
        f"          {name[0]:>2}"
    )


def make_pdb(residues, waters=()):
    """Build a PDB string of ALA residues.

    residues: list of (chain, resseq, altlocs) or
              (chain, resseq, altlocs, icode, shared_backbone) tuples.
              altlocs == "" means a single conformer. With shared_backbone,
              N/CA/C/O have a blank altloc and only CB is split.
    waters:   list of (chain, resseq) HOH residues appended as HETATM.
    """
    lines = []
    serial = 1
    for i, res in enumerate(residues):
        chain, resseq, altlocs = res[:3]
        icode = res[3] if len(res) > 3 else ""
        shared_backbone = res[4] if len(res) > 4 else False
        alts = list(altlocs) or [""]
        occ = round(1.0 / len(alts), 2)
        x0 = 3.8 * i
        if shared_backbone:
            for j, name in enumerate(BACKBONE):
                lines.append(_pdb_line(serial, name, "", "ALA", chain, resseq, icode,
                                       (x0 + 0.5 * j, 0.0, 0.0), 1.0))
                serial += 1
            for k, alt in enumerate(alts):
                lines.append(_pdb_line(serial, "CB", alt, "ALA", chain, resseq, icode,
                                       (x0, 1.5, 0.3 * k), occ))
                serial += 1
            continue
        for k, alt in enumerate(alts):
            for j, name in enumerate(BACKBONE + ("CB",)):
                lines.append(_pdb_line(serial, name, alt, "ALA", chain, resseq, icode,
                                       (x0 + 0.5 * j, 0.2 * k, 0.3 * k), occ))
                serial += 1
    for i, (chain, resseq) in enumerate(waters):
        lines.append(_pdb_line(serial, "O", "", "HOH", chain, resseq, "",
                               (0.0, 10.0 + 3.0 * i, 0.0), 1.0, record="HETATM"))
        serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


SELECTION_RE = re.compile(r"^\s*selection = altid (\S+) and \((.*)\)\s*$")
RESIDUE_RE = re.compile(r"\(chain (\S+) and (?:resid|resseq) (\S+)\)")


def parse_params(text):
    """Return a list of groups; each group is a list of (altloc, [(chain, resid), ...])."""
    groups = []
    current = None
    for line in text.splitlines():
        if "constrained_group {" in line:
            assert current is None, "nested constrained_group"
            current = []
        elif line.strip().startswith("selection"):
            assert current is not None, "selection outside constrained_group"
            m = SELECTION_RE.match(line)
            assert m, f"unexpected selection line: {line!r}"
            current.append((m.group(1), RESIDUE_RE.findall(m.group(2))))
        elif line.strip() == "}" and current is not None:
            groups.append(current)
            current = None
    assert current is None, "unterminated constrained_group"
    return groups


def summarize(groups):
    """Reduce parsed groups to (chain, [resids], [altlocs]) and check that
    every selection in a group covers the same residues."""
    out = []
    for group in groups:
        residue_sets = {tuple(res) for _, res in group}
        assert len(residue_sets) == 1, f"selections in a group differ: {group}"
        residues = residue_sets.pop()
        chains = {c for c, _ in residues}
        assert len(chains) == 1, f"group spans chains: {group}"
        out.append((chains.pop(), [r for _, r in residues], [a for a, _ in group]))
    return out


def check_with_iotbx(pdb_text, params_text):
    """Check the selections against the model with the same selection parser
    phenix.refine uses: every selection is non-empty, no atom is in more
    than one selection, and every atom with a non-blank altloc in a
    multi-conformer protein residue is covered."""
    iotbx.phil.parse(params_text)  # valid PHIL syntax
    hierarchy = iotbx.pdb.input(source_info=None, lines=pdb_text).construct_hierarchy()
    cache = hierarchy.atom_selection_cache()
    seen = set()
    for line in params_text.splitlines():
        m = SELECTION_RE.match(line)
        if not m:
            continue
        sel = set(cache.selection(f"altid {m.group(1)} and ({m.group(2)})").iselection())
        assert sel, f"empty selection: {line.strip()}"
        assert not (sel & seen), f"atoms in more than one selection: {line.strip()}"
        seen |= sel
    expected = set()
    for rg in hierarchy.residue_groups():
        if rg.atom_groups()[0].resname == "HOH":
            continue
        named = {ag.altloc for ag in rg.atom_groups() if ag.altloc.strip()}
        if len(named) >= 2:
            expected |= {a.i_seq for a in rg.atoms() if a.parent().altloc.strip()}
    assert seen == expected


def run_script(tmp_path, pdb_text):
    pdb = tmp_path / "model.pdb"
    pdb.write_text(pdb_text)
    subprocess.check_call([sys.executable, SCRIPT, str(pdb)], cwd=tmp_path)
    params_text = (tmp_path / "qFit_occupancy.params").read_text()
    check_with_iotbx(pdb_text, params_text)
    return summarize(parse_params(params_text))


def test_segment_two_altlocs(tmp_path):
    pdb = make_pdb([("A", 1, ""), ("A", 2, "AB"), ("A", 3, "AB"), ("A", 4, "AB"), ("A", 5, "")])
    assert run_script(tmp_path, pdb) == [("A", ["2", "3", "4"], ["A", "B"])]


def test_segment_three_altlocs(tmp_path):
    pdb = make_pdb([("A", 1, ""), ("A", 2, "ABC"), ("A", 3, "ABC"), ("A", 4, "")])
    assert run_script(tmp_path, pdb) == [("A", ["2", "3"], ["A", "B", "C"])]


def test_segment_at_end_of_chain_and_file(tmp_path):
    # The last residue of the file is multi-conformer.
    pdb = make_pdb([("A", 1, ""), ("A", 2, "AB"), ("A", 3, "AB")])
    assert run_script(tmp_path, pdb) == [("A", ["2", "3"], ["A", "B"])]


def test_segment_at_end_of_chain_followed_by_waters(tmp_path):
    pdb = make_pdb([("A", 1, ""), ("A", 2, "AB"), ("A", 3, "AB")],
                   waters=[("A", 101), ("A", 102)])
    assert run_script(tmp_path, pdb) == [("A", ["2", "3"], ["A", "B"])]


def test_two_chains_with_segments(tmp_path):
    # Chain A ends in a multi-conformer segment and chain B starts with one,
    # and both chains use the same residue numbers.
    pdb = make_pdb([
        ("A", 1, ""), ("A", 2, "AB"), ("A", 3, "AB"),
        ("B", 1, "AB"), ("B", 2, "AB"), ("B", 3, ""), ("B", 4, "ABC"),
    ])
    assert run_script(tmp_path, pdb) == [
        ("A", ["2", "3"], ["A", "B"]),
        ("B", ["1", "2"], ["A", "B"]),
        ("B", ["4"], ["A", "B", "C"]),
    ]


def test_mixed_single_and_multi_stretches(tmp_path):
    pdb = make_pdb([
        ("A", 1, "AB"), ("A", 2, ""), ("A", 3, ""),
        ("A", 4, "AB"), ("A", 5, "AB"), ("A", 6, ""),
        ("A", 7, "ABC"), ("A", 8, ""), ("A", 9, "AB"),
    ])
    assert run_script(tmp_path, pdb) == [
        ("A", ["1"], ["A", "B"]),
        ("A", ["4", "5"], ["A", "B"]),
        ("A", ["7"], ["A", "B", "C"]),
        ("A", ["9"], ["A", "B"]),
    ]


def test_altloc_set_change_splits_segment(tmp_path):
    # Without a split, the selection for C would cover only residue 4, and
    # residues 2-3 would be constrained to occ(A) + occ(B) = 1 - occ(C).
    pdb = make_pdb([
        ("A", 1, ""), ("A", 2, "AB"), ("A", 3, "AB"),
        ("A", 4, "ABC"), ("A", 5, "ABC"), ("A", 6, "AB"), ("A", 7, ""),
    ])
    assert run_script(tmp_path, pdb) == [
        ("A", ["2", "3"], ["A", "B"]),
        ("A", ["4", "5"], ["A", "B", "C"]),
        ("A", ["6"], ["A", "B"]),
    ]


def test_insertion_codes(tmp_path):
    # "resseq 52" would also select 52A; "resid" does not.
    pdb = make_pdb([
        ("A", 51, ""), ("A", 52, "AB"), ("A", 52, "AB", "A"), ("A", 53, ""),
        ("A", 54, "ABC", "B"),
    ])
    assert run_script(tmp_path, pdb) == [
        ("A", ["52", "52A"], ["A", "B"]),
        ("A", ["54B"], ["A", "B", "C"]),
    ]


def test_blank_altloc_backbone(tmp_path):
    # Residue 2 has a shared (blank-altloc) backbone and a split side chain:
    # it is multi-conformer, and the blank atoms are not put in a selection.
    pdb = make_pdb([
        ("A", 1, ""), ("A", 2, "AB", "", True), ("A", 3, "AB"), ("A", 4, ""),
    ])
    assert run_script(tmp_path, pdb) == [("A", ["2", "3"], ["A", "B"])]


def test_blank_altloc_atom_in_split_residue(tmp_path):
    # Residue 2 has split backbone (A/B) but a single blank-altloc CB.
    # Previously this raised IndexError on the blank altloc.
    lines = []
    kept_cb = False
    for line in make_pdb([("A", 1, ""), ("A", 2, "AB"), ("A", 3, "")]).splitlines():
        if line[12:16] == " CB " and line[22:26] == "   2":
            if kept_cb:
                continue
            kept_cb = True
            line = line[:16] + " " + line[17:54] + "  1.00" + line[60:]
        lines.append(line)
    pdb = "\n".join(lines) + "\n"
    assert run_script(tmp_path, pdb) == [("A", ["2"], ["A", "B"])]


def test_no_multiconformer_residues(tmp_path):
    pdb = make_pdb([("A", 1, ""), ("A", 2, "")])
    assert run_script(tmp_path, pdb) == []
