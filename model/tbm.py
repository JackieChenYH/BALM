"""Python interface to tensorsparse 3.0 for the TBM comparison."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from .alma import _training_arrays


def tbm_start_seeds(seed, policy="paired_masks", count=20):
    """Return deterministic per-start seeds for each experiment family.

    paired_masks: Experiment 1; simulation: Experiments 4--7;
    hcp: the three HCP cohort comparisons.
    """
    if count < 1:
        raise ValueError("count must be positive")
    seeds = []
    for start in range(count):
        if policy == "paired_masks":
            value = int(hashlib.sha256(f"tbm-author-r-v1:{seed}:{start}".encode()).hexdigest()[:16], 16) % 2147483646 + 1
        elif policy == "simulation":
            raw = json.dumps([int(seed), "r_tbm", start], sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
            value = (int(hashlib.sha256(raw.encode()).hexdigest()[:16], 16) % (2**32 - 1)) % (2**31 - 1)
        elif policy == "hcp":
            value = (int(seed) + start * 7919) % (2**31 - 1)
        else:
            raise ValueError("policy must be paired_masks, simulation, or hcp")
        seeds.append(value)
    return seeds


def fit_tbm(z, train_mask, n, M, K, seed=0, *, start_seeds=None,
            seed_policy="paired_masks", max_iter=1000, tol=1e-10,
            rscript=None, output_dir=None):
    """Run unpenalized TBM with training-only imputation and 20 starts.

    Uses installed R package ``tensorsparse==3.0``. The first minimum training
    objective selects the result. Predicted directions are averaged for each
    unordered edge. Inputs have shape (subjects, n*(n-1)//2).

    ``start_seeds`` overrides the experiment-family seed policy. A persistent
    output_dir retains each native start, warnings and R session information;
    otherwise temporary exchange files are removed after loading the result.
    """
    z, mask = _training_arrays(z, train_mask, n, M, K)
    seeds = tbm_start_seeds(seed, seed_policy) if start_seeds is None else list(start_seeds)
    if not seeds or any(int(value) != value or value < 0 or value >= 2**31 - 1 for value in seeds):
        raise ValueError("R seeds must be integers in [0, 2**31-1)")
    if max_iter < 1 or int(max_iter) != max_iter or not np.isfinite(tol) or tol <= 0:
        raise ValueError("max_iter and tol must be positive")
    requested = str(rscript or os.environ.get("RSCRIPT", "Rscript"))
    executable = shutil.which(requested)
    if executable is None:
        raise FileNotFoundError("Rscript was not found; install R or pass rscript='/path/to/Rscript'.")
    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix="balm_tbm_") as temporary:
            return _run_tbm(z, mask, n, M, K, seeds, max_iter, tol, executable, Path(temporary))
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("TBM output_dir must be empty to preserve existing results")
    return _run_tbm(z, mask, n, M, K, seeds, max_iter, tol, executable, output)


def _run_tbm(z, mask, n, M, K, seeds, max_iter, tol, executable, output):
    i, j = np.triu_indices(n, 1)
    with (output / "training.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject", "edge", "node_i", "node_j", "train", "z"])
        for subject in range(len(z)):
            writer.writerows((subject, edge, i[edge], j[edge], int(mask[subject, edge]),
                              int(z[subject, edge]) if mask[subject, edge] else "")
                             for edge in range(len(i)))
    with (output / "seeds.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["start", "seed"])
        writer.writerows(enumerate(seeds))
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = "1"
    command = [executable, "--vanilla", str(Path(__file__).with_suffix(".R")),
               str(output), str(n), str(len(z)), str(M), str(K), str(max_iter), repr(tol)]
    with (output / "stdout.txt").open("w", encoding="utf-8") as stdout, (output / "stderr.txt").open("w", encoding="utf-8") as stderr:
        process = subprocess.run(command, stdout=stdout, stderr=stderr, env=env,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if process.returncode:
        details = (output / "stderr.txt").read_text("utf-8")
        raise RuntimeError(f"TBM R process failed (exit {process.returncode}):\n{details}")
    values = np.atleast_1d(np.genfromtxt(output / "topology_scores.csv", delimiter=",", names=True))
    labels = np.atleast_1d(np.genfromtxt(output / "labels.csv", delimiter=",", names=True))
    if not np.array_equal(values["subject"], np.repeat(np.arange(len(z)), len(i))) or not np.array_equal(values["edge"], np.tile(np.arange(len(i)), len(z))):
        raise ValueError("TBM output ordering differs from the input")
    with (output / "starts.csv").open(newline="", encoding="utf-8") as stream:
        starts = list(csv.DictReader(stream))
    with (output / "selected.csv").open(newline="", encoding="utf-8") as stream:
        selected = next(csv.DictReader(stream))
    score = values["topology_score"].reshape(z.shape)
    if not np.isfinite(score).all():
        raise FloatingPointError("TBM returned nonfinite predictions")
    cap_count = sum(row["iteration_cap_warning"] == "TRUE" for row in starts)
    return {"topology_score": score, "labels": labels["label"].astype(int),
            "optimization": {"starts": starts, "selected_start": int(selected["selected_start"]),
                             "training_filled_SSE": float(selected["training_filled_SSE"]),
                             "iteration_cap_count": cap_count, "converged": cap_count == 0,
                             "max_iter": max_iter, "tolerance": tol, "package": "tensorsparse 3.0"}}
