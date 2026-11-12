"""
Tests for redistribute_cull_low_occupancies.

qfit_final_refine_xray.sh reads the last line this tool prints and repeats
occupancy refinement until it is 1 or less, so that line must count only
residues that lost a low-occupancy conformer.
"""

import subprocess

import iotbx.pdb
import pytest


def make_pdb(residues):
    """residues: list of (resseq, {altloc: occupancy}); {"": 1.0} = single conformer."""
    lines = []
    serial = 1
    for i, (resseq, confs) in enumerate(residues):
        for k, (alt, occ) in enumerate(confs.items()):
            for j, name in enumerate(("N", "CA", "C", "O", "CB")):
                x, y, z = 3.8 * i + 0.5 * j, 0.2 * k, 0.3 * k
                lines.append(
                    f"ATOM  {serial:5d}  {name:<3}{alt:1}ALA A{resseq:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{20.0:6.2f}"
                    f"          {name[0]:>2}"
                )
                serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


def run_tool(tmp_path, residues, cutoff="0.09"):
    (tmp_path / "model.pdb").write_text(make_pdb(residues))
    out = subprocess.run(
        ["redistribute_cull_low_occupancies", "-occ", cutoff, "model.pdb"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    ).stdout
    hierarchy = iotbx.pdb.input(str(tmp_path / "model_norm.pdb")).construct_hierarchy()
    occs = {}
    for rg in hierarchy.residue_groups():
        occs[rg.resseq_as_int()] = {
            ag.altloc: round(ag.atoms()[0].occ, 2) for ag in rg.atom_groups()
        }
    return int(out.strip().splitlines()[-1]), occs


@pytest.mark.parametrize(
    "residues, expected_count",
    [
        # Nothing below the cutoff: the refinement loop must be allowed to stop.
        ([(1, {"": 1.0}), (2, {"A": 0.5, "B": 0.5}),
          (3, {"A": 0.4, "B": 0.3, "C": 0.3}), (4, {"": 1.0})], 0),
        ([(1, {"": 1.0}), (2, {"A": 0.5, "B": 0.45, "C": 0.05}),
          (3, {"A": 0.5, "B": 0.5})], 1),
        ([(1, {"A": 0.95, "B": 0.05}), (2, {"A": 0.5, "B": 0.45, "C": 0.05}),
          (3, {"A": 0.5, "B": 0.5})], 2),
    ],
    ids=["none-low", "one-residue-low", "two-residues-low"],
)
def test_count_is_residues_with_low_occupancy_conformers(tmp_path, residues, expected_count):
    count, _ = run_tool(tmp_path, residues)
    assert count == expected_count


def test_low_conformers_culled_and_occupancies_renormalized(tmp_path):
    count, occs = run_tool(tmp_path, [
        (1, {"A": 0.95, "B": 0.05}),             # one conformer left
        (2, {"A": 0.5, "B": 0.45, "C": 0.05}),   # C culled, A/B renormalized
        (3, {"A": 0.6, "B": 0.6}),               # inflated sum, nothing culled
    ])
    assert count == 2
    assert occs[1] == {"": 1.0}
    assert set(occs[2]) == {"A", "B"} and abs(sum(occs[2].values()) - 1) < 0.011
    # Residues without low conformers are still renormalized (8564df2).
    assert occs[3] == {"A": 0.5, "B": 0.5}
