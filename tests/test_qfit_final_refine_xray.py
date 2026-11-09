"""
Tests for the R-free flag handling in scripts/post/qfit_final_refine_xray.sh.

Phenix is not needed: the script is run end to end with mock versions of
phenix.mtz.dump, phenix.refine and the other tools it calls placed first on
PATH. phenix.mtz.dump prints a fixture, and phenix.refine records its
arguments and writes the output files the script expects. The tests check
which free-flag options and which .mtz each phenix.refine call receives.
"""

import os
import os.path as op
import shutil
import subprocess

import pytest

SCRIPT = op.join(
    op.dirname(op.dirname(op.abspath(__file__))),
    "scripts",
    "post",
    "qfit_final_refine_xray.sh",
)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

# Header and column table of `phenix.mtz.dump 7ur7.mtz` (abridged). The
# R-free-flags-1 column checks that labels are matched in full.
MTZ_DUMP_HEADER = """\
Processing: {name}.mtz
Title: phenix.cif_as_mtz
Space group symbol from file: P3121
Space group number from file: 152
Space group from matrices: P 31 2 1 (No. 152)
Point group symbol from file: 32
Number of crystals: 2
Number of Miller indices: 22354
Resolution range: 36.5385 1.20557
History:
Crystal 2:
  Name: crystal_0
  Project: project_0
  Id: 2
  Unit cell: (42.191, 42.191, 68.45, 90, 90, 120)
  Number of datasets: 1
  Dataset 1:
    Name: dataset
    Id: 1
    Wavelength: 0.97918
    Number of columns: 13
    label          #valid  %valid     min     max type
    H               22354 100.00%    0.00   30.00 H: index h,k,l
    K               22354 100.00%    0.00   17.00 H: index h,k,l
    L               22354 100.00%  -56.00   56.00 H: index h,k,l
"""
FOBS_COLUMNS = """\
    FOBS            22125  98.98%    2.53 1022.48 F: amplitude
    SIGFOBS         22125  98.98%    1.00  220.33 Q: standard deviation
    IOBS            22125  98.98%   -1.96 7548.60 J: intensity
    SIGIOBS         22125  98.98%    0.12 1698.66 Q: standard deviation
"""


def free_column(label, max_value="1.00"):
    return f"    {label:<15} 22354 100.00%    0.00   {max_value:>5} I: integer\n"


MOCKS = {
    "phenix.mtz.dump": 'cat "${MOCK_MTZ_DUMP}"',
    "phenix.refine": r"""
n=$(ls "${MOCK_CALLS}" | wc -l | tr -d ' ')
printf '%s\n' "$@" > "${MOCK_CALLS}/refine_$(printf '%03d' "${n}")"
for a in "$@"; do
  case "${a}" in
    output.prefix=*) prefix="${a#*=}" ;;
    output.serial=*) serial="${a#*=}" ;;
  esac
done
for ext in pdb mtz log; do
  echo mock > "${prefix}_$(printf '%03d' "${serial}").${ext}"
done
""",
    "remove_duplicates": 'touch "$1.fixed" "${1%.pdb}_cleaned.pdb"',
    "redistribute_cull_low_occupancies": r"""
f="${@: -1}"
case "${f}" in
  *.fixed) touch "${f%.fixed}.f_norm.pdb" ;;
  *) touch "${f%.pdb}_norm.pdb" ;;
esac
echo 0
""",
    "phenix.pdbtools": 'f="${@: -1}"; touch "${f%.fixed}.f_modified.pdb"',
    "phenix.ready_set": r"""
for a in "$@"; do
  case "${a}" in pdb_file_name=*) f="${a#*=}" ;; esac
done
touch "${f%.pdb}.updated.pdb"
""",
    "phenix.elbow": "true",
    "create_restraints_file.py": "touch qFit_occupancy.params",
}


def run_script(tmp_path, mtz_dump, name="7ur7"):
    bindir = tmp_path / "bin"
    calls = tmp_path / "calls"
    work = tmp_path / "work"
    for d in (bindir, calls, work):
        d.mkdir()
    for tool, body in MOCKS.items():
        path = bindir / tool
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)
    dump = tmp_path / "mtz_dump.txt"
    dump.write_text(mtz_dump.format(name=name))
    (work / f"{name}.mtz").write_text("mock")
    (work / "multiconformer_model2.pdb").write_text("mock")
    env = dict(
        os.environ,
        PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}",
        PHENIX="/mock/phenix",
        MOCK_MTZ_DUMP=str(dump),
        MOCK_CALLS=str(calls),
    )
    proc = subprocess.run(
        ["bash", SCRIPT, f"{name}.mtz"],
        cwd=work, env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    refine_calls = [
        (calls / f).read_text().splitlines() for f in sorted(os.listdir(calls))
    ]
    return proc, refine_calls, work


def free_args(args):
    return [
        a for a in args
        if a.startswith("data_manager.miller_array.labels.name=")
        or "r_free_flags" in a
    ]


def check_later_rounds(refine_calls, name="7ur7"):
    # One coordinate round, one occupancy round, the final round.
    assert len(refine_calls) == 3
    assert f"{name}.mtz" in refine_calls[0]
    for args in refine_calls[1:]:
        # Later rounds refine against the first round's output, which
        # carries the free set used in the first round.
        assert f"{name}_002.mtz" in args
        assert f"{name}.mtz" not in args
        assert free_args(args) == []


@pytest.mark.parametrize("label", ["R-free-flags", "FreeR_flag", "FREE"])
def test_existing_free_flags_are_used(tmp_path, label):
    dump = MTZ_DUMP_HEADER + free_column(label) + FOBS_COLUMNS
    proc, refine_calls, work = run_script(tmp_path, dump)
    assert free_args(refine_calls[0]) == [
        "data_manager.miller_array.labels.name=FOBS,SIGFOBS",
        f"data_manager.miller_array.labels.name={label}",
        "xray_data.r_free_flags.generate=False",
    ]
    check_later_rounds(refine_calls)
    assert f"Rfree column: {label}" in proc.stdout
    assert "WARNING" not in proc.stderr
    params = (work / "7ur7_refine.params").read_text().splitlines()
    assert f"data_manager.miller_array.labels.name={label}" in params
    assert "xray_data.r_free_flags.generate=False" in params
    assert (work / "7ur7_qFit.pdb").exists()


def test_label_is_matched_in_full(tmp_path):
    # As in 7ur7.mtz: R-free-flags must be chosen, not R-free-flags-1.
    dump = (MTZ_DUMP_HEADER + free_column("R-free-flags")
            + free_column("R-free-flags-1", "19.00") + FOBS_COLUMNS)
    _, refine_calls, _ = run_script(tmp_path, dump)
    assert "data_manager.miller_array.labels.name=R-free-flags" in refine_calls[0]
    assert "data_manager.miller_array.labels.name=R-free-flags-1" not in refine_calls[0]


@pytest.mark.parametrize(
    "columns", ["", free_column("R-free-flags-1", "19.00")],
    ids=["no-free-column", "only-R-free-flags-1"],
)
def test_free_flags_generated_only_if_missing(tmp_path, columns):
    dump = MTZ_DUMP_HEADER + columns + FOBS_COLUMNS
    proc, refine_calls, work = run_script(tmp_path, dump)
    assert free_args(refine_calls[0]) == [
        "data_manager.miller_array.labels.name=FOBS,SIGFOBS",
        "xray_data.r_free_flags.generate=True",
    ]
    check_later_rounds(refine_calls)
    assert "WARNING" in proc.stderr and "NEW set of R-free flags" in proc.stderr
    params = (work / "7ur7_refine.params").read_text().splitlines()
    assert "xray_data.r_free_flags.generate=True" in params
