# batch_fit.py
# Fit one PDB atomic model into many cryo-ET density maps and record the correlation of the best fit for each. Useful for screening a batch of particles (e.g. full vs empty VLPs) against a known reference structure.
#
# Run headless from a terminal:
#   chimerax --nogui --exit batch_fit.py
#
# ChimeraX injects `session` into this script's namespace, so it is used
# directly below without importing it.

import os
import glob
import csv

from chimerax.core.commands import run
from chimerax.map_fit.fitcmd import fitmap  # returns Fit objects we can query

# ----------------------------- settings ------------------------------------
DENSITY_DIR = "/path/to/subtomograms"   # folder of .mrc maps, one particle each
PDB_ID      = "6N4V"                     # reference model, fetched from PDB by ID
RESOLUTION  = 10.0                       # angstroms; roughly match your tomogram
SEARCH      = 50                         # random starting orientations per map
OUT_CSV     = "fit_results.csv"
# ---------------------------------------------------------------------------

# Fetch the atomic model once and reuse it for every density map. Each fit
# repositions this same model, so there is no need to reopen it per map.
run(session, f"open {PDB_ID}")
model = session.models.list()[-1]

rows = []
maps = sorted(glob.glob(os.path.join(DENSITY_DIR, "*.mrc")))

for path in maps:
    run(session, f'open "{path}"')
    density = session.models.list()[-1]

    # Global rigid-body fit: simulate a map from the atoms at RESOLUTION,
    # then try SEARCH random starting orientations and keep the best.
    fits = fitmap(
        session,
        model.atoms,
        density,
        resolution=RESOLUTION,
        search=SEARCH,
    )

    best = max((f.correlation() for f in fits), default=None)
    rows.append(
        {
            "map": os.path.basename(path),
            "n_fits": len(fits),
            "best_correlation": best,
        }
    )

    # Free the density before loading the next one.
    run(session, f"close #{density.id_string}")

with open(OUT_CSV, "w", newline="") as fh:
    writer = csv.DictWriter(
        fh, fieldnames=["map", "n_fits", "best_correlation"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} results to {OUT_CSV}")
