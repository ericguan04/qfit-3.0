#!/usr/bin/env python

import argparse
import os

from qfit.structure import Structure
from qfit.structure.rotamers import ROTAMERS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process multiconformer file for refinement."
    )
    parser.add_argument("structure", type=str, help="Path to the structure file")
    return parser.parse_args()


def find_constraint_segments(multiconformer):
    """
    Yield (chain_id, resids, altlocs) for each run of consecutive
    multi-conformer residues in one chain that have the same set of altlocs.

    A residue is multi-conformer if it has at least two non-blank altlocs;
    blank-altloc atoms are shared by all conformers and need no constraint.
    A run ends at a single-conformer residue, at a change in the altloc set,
    and at the end of a chain. The altloc set must be identical within a run:
    each selection in a constrained_group gets a single occupancy, so a
    residue lacking one of the group's altlocs would be constrained to sum
    to less than 1. Residue ids include the insertion code.
    """
    for chain in multiconformer:
        resids = []
        altlocs = None
        for residue in chain:
            if residue.resn[0] not in ROTAMERS:
                continue
            residue_altlocs = sorted(set(residue.altloc) - {""})
            if len(residue_altlocs) < 2:
                residue_altlocs = None  # single conformer
            if residue_altlocs != altlocs and resids:
                yield chain.id, resids, altlocs
                resids = []
            altlocs = residue_altlocs
            if altlocs is not None:
                resi, icode = residue.id
                resids.append(f"{resi}{icode}")
        if resids:
            yield chain.id, resids, altlocs


def create_refine_restraints(multiconformer):
    """
    For refinement, we need to create occupancy restraints for all residues in the same segment with the same altloc.
    This function will go through the qFit output and create a constraint file to be fed into refinement
    """
    fname = os.path.join("qFit_occupancy.params")
    f = open(fname, "w+")
    f.write("refinement {\n")
    f.write("  refine {\n")
    f.write("    occupancies {\n")
    for chain_id, resids, altlocs in find_constraint_segments(multiconformer):
        # One group per segment, with one selection per altloc, so that phenix
        # constrains the altloc occupancies of the segment to sum to 1.
        resi_selection = " or ".join(
            f"(chain {chain_id} and resid {resid})" for resid in resids
        )
        f.write("      constrained_group {\n")
        for a in altlocs:
            f.write(f"        selection = altid {a} and ({resi_selection})\n")
        f.write("      }\n")
    f.write("   }\n")
    f.write(" }\n")
    f.write("}\n")
    f.close()


def main():
    args = parse_args()
    structure = Structure.fromfile(args.structure).reorder()
    create_refine_restraints(structure)


if __name__ == "__main__":
    main()
