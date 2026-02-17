"""Multi-GPU arm-parallel execution utility.

Pattern: each experiment defines a list of "arms" (e.g., conditions, K values,
alpha sweeps).  This module distributes arms across GPUs by spawning one
subprocess per arm, each pinned to a specific ``CUDA_VISIBLE_DEVICES``.

Worker results are written to JSON temp files and collected by the
orchestrator for final aggregation.

Usage inside an experiment script::

    from experiments.rebuttal.multi_gpu import (
        add_multigpu_args, run_arms_parallel, is_worker,
    )

    # In argparse setup
    add_multigpu_args(parser)

    # In main()
    if args.num_gpus > 1 and not is_worker(args):
        arm_specs = [{"mode": "safe_cover"}, {"mode": "pareto"}]
        results = run_arms_parallel(args, arm_specs, __file__)
        # results is List[Dict] – one dict per arm, whatever the worker wrote
        ...
    else:
        # single-GPU / worker path
        ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── public helpers ──────────────────────────────────────────────────────────

def add_multigpu_args(parser: argparse.ArgumentParser) -> None:
    """Add --num_gpus and internal worker flags to an argparser."""
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to distribute arms across")
    parser.add_argument("--gpu_ids", type=str, default="",
                        help="Comma-separated GPU ids (e.g. '0,1,2,3'). "
                             "If empty, uses 0..num_gpus-1")
    # Internal flags – set automatically by the orchestrator
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_worker_gpu", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--_arm_json", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--_result_path", type=str, default="", help=argparse.SUPPRESS)


def is_worker(args: argparse.Namespace) -> bool:
    """Return True when running as a subprocess worker."""
    return getattr(args, "_worker", False)


def get_arm_spec(args: argparse.Namespace) -> Dict[str, Any]:
    """Deserialise the arm spec passed to a worker subprocess."""
    raw = getattr(args, "_arm_json", "")
    if not raw:
        return {}
    return json.loads(raw)


def get_gpu_ids(args: argparse.Namespace) -> List[int]:
    """Return list of GPU ids to use."""
    if args.gpu_ids:
        return [int(x) for x in args.gpu_ids.split(",")]
    return list(range(args.num_gpus))


def run_arms_parallel(
    args: argparse.Namespace,
    arm_specs: List[Dict[str, Any]],
    script_path: str,
    *,
    timeout: int = 14400,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Distribute *arm_specs* across GPUs and collect results.

    Each arm is run as ``python <script> --_worker --_worker_gpu <G>
    --_arm_json '<json>' --_result_path '<tmp>'`` plus all of the original
    CLI arguments (except the multi-gpu ones).

    Returns a list of dicts, one per arm, in the same order as *arm_specs*.
    """
    gpu_ids = get_gpu_ids(args)
    n_gpus = len(gpu_ids)

    # Build base command (reproduce the original CLI but strip multi-gpu args)
    base_cmd = _build_base_cmd(args, script_path)

    # Prepare temp files for results
    tmpdir = tempfile.mkdtemp(prefix="trident_multigpu_")
    result_paths = []
    for i in range(len(arm_specs)):
        result_paths.append(os.path.join(tmpdir, f"arm_{i}.json"))

    # Launch subprocesses – at most n_gpus at a time
    all_results: List[Optional[Dict[str, Any]]] = [None] * len(arm_specs)
    batches = _batch(list(enumerate(arm_specs)), n_gpus)

    for batch in batches:
        procs = []
        for batch_idx, (arm_idx, spec) in enumerate(batch):
            gpu_id = gpu_ids[batch_idx % n_gpus]
            arm_json = json.dumps(spec)
            cmd = base_cmd + [
                "--_worker",
                "--_worker_gpu", str(gpu_id),
                "--_arm_json", arm_json,
                "--_result_path", result_paths[arm_idx],
            ]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            label = spec.get("label", f"arm_{arm_idx}")
            print(f"  [GPU {gpu_id}] Launching arm {arm_idx}: {label}")
            if verbose:
                print(f"    CMD: {' '.join(cmd)}")

            proc = subprocess.Popen(
                cmd, env=env,
                stdout=None if verbose else subprocess.PIPE,
                stderr=None if verbose else subprocess.PIPE,
                text=True,
            )
            procs.append((arm_idx, proc, label))

        # Wait for batch to finish
        for arm_idx, proc, label in procs:
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
                if proc.returncode != 0:
                    print(f"  [WARN] arm {arm_idx} ({label}) exited with code {proc.returncode}")
                    if stderr:
                        print(f"    stderr (tail): {stderr[-500:]}")
                else:
                    print(f"  [OK] arm {arm_idx} ({label}) finished")
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"  [TIMEOUT] arm {arm_idx} ({label})")

    # Collect results
    for i, rp in enumerate(result_paths):
        if os.path.exists(rp):
            with open(rp) as f:
                all_results[i] = json.load(f)
        else:
            print(f"  [WARN] No result file for arm {i} ({result_paths[i]})")
            all_results[i] = {"_error": "no_result_file", "_arm_spec": arm_specs[i]}

    return all_results


def write_worker_result(args: argparse.Namespace, result: Dict[str, Any]) -> None:
    """Write the worker result dict to the temp file the orchestrator expects."""
    rp = getattr(args, "_result_path", "")
    if not rp:
        return
    Path(rp).parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)


# ── private helpers ─────────────────────────────────────────────────────────

def _build_base_cmd(args: argparse.Namespace, script_path: str) -> List[str]:
    """Reconstruct the CLI invocation, stripping multi-gpu flags."""
    # Use the script path directly via python -u
    cmd = [sys.executable, "-u", script_path]

    # Replay all the original CLI args EXCEPT the internal/multigpu ones
    skip_keys = {"num_gpus", "gpu_ids", "_worker", "_worker_gpu",
                 "_arm_json", "_result_path"}
    # Also skip keys that will be overridden by arm_spec
    # (handled by individual scripts via arm_spec merging)

    for key, val in vars(args).items():
        if key in skip_keys:
            continue
        if key.startswith("_"):
            continue
        if val is None:
            continue

        cli_key = f"--{key}"
        if isinstance(val, bool):
            if val:
                cmd.append(cli_key)
        elif isinstance(val, list):
            if val:
                cmd.append(cli_key)
                cmd.extend(str(v) for v in val)
        else:
            cmd += [cli_key, str(val)]

    return cmd


def _batch(items: list, batch_size: int) -> List[list]:
    """Split items into batches of at most batch_size."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def _json_default(obj):
    import numpy as np
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)
