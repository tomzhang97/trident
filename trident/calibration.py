"""
Calibration module for conformal prediction.
UPDATES:
- Fixed PickleError by replacing lambda with named factory function.
- Preserves defaultdict behavior after loading.
"""
import copy
import json
import logging
import math
import pickle
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

def _to_plain(obj):
    """Recursively convert defaultdicts/numpy types to plain python for JSON."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(i) for i in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# FIX: Top-level factory function for pickling support
def _nested_dd_factory():
    return defaultdict(list)

class ReliabilityCalibrator:
    def __init__(self, use_mondrian: bool = True):
        self.use_mondrian = use_mondrian
        self.conformal_calibrator = None 

    def set_conformal_calibrator(self, calibrator):
        self.conformal_calibrator = calibrator

    def fit(self, *args, **kwargs):
        pass 

    def _get_canonical_key(self, key: str) -> str:
        """Map specific facet types to canonical buckets."""
        if "BRIDGE_HOP" in key:
            return "BRIDGE_HOP"
        return key

    def to_pvalue(self, score: float, facet_type: Union[str, Any], text_length: int = 0) -> float:
        """
        Convert score to p-value with robust fallback.
        """
        import os
        debug = os.environ.get("TRIDENT_DEBUG_PVALUE", "0") == "1"

        # Safety: Coerce inputs
        if hasattr(facet_type, "value"):
            facet_type = facet_type.value
        facet_type = str(facet_type)
        score = float(score)

        if self.conformal_calibrator:
            try:
                p = self.conformal_calibrator.get_p_value(score, facet_type, text_length)
                if debug:
                    print(f"[PVALUE] type={facet_type}, score={score:.4f}, len={text_length}, p={p:.4f}")
                return p
            except KeyError:
                # 1. Canonical Fallback (e.g. BRIDGE_HOP3 -> BRIDGE_HOP)
                canon = self._get_canonical_key(facet_type)
                if canon != facet_type:
                    try:
                        p = self.conformal_calibrator.get_p_value(score, canon, text_length)
                        if debug:
                            print(f"[PVALUE] type={facet_type}→{canon}, score={score:.4f}, len={text_length}, p={p:.4f}")
                        return p
                    except KeyError:
                        pass

                # 2. Global Fallback (Fail-Closed)
                if debug:
                    print(f"[PVALUE] FALLBACK type={facet_type}, score={score:.4f} → p=1.0 (no calibration data)")
                return 1.0

        if debug:
            print(f"[PVALUE] NO CALIBRATOR → p=1.0")
        return 1.0

    def save(self, path: str):
        data = {
            "version": "v2.2",
            "use_mondrian": self.use_mondrian,
            "conformal": self.conformal_calibrator.to_dict() if self.conformal_calibrator else None
        }
        with open(path, "w") as f:
            json.dump(_to_plain(data), f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        
        cal = cls(use_mondrian=data.get("use_mondrian", True))
        if data.get("conformal"):
            cal.conformal_calibrator = SelectionConditionalCalibrator.from_dict(data["conformal"])
        return cal

# --- INNER CALIBRATOR ---

class SelectionConditionalCalibrator:
    def __init__(self, n_min: int = 50, use_mondrian: bool = True):
        self.n_min = n_min
        self.use_mondrian = use_mondrian
        # FIX: Use named factory function instead of lambda for pickle safety
        self.bins: Dict[str, Dict[str, List[float]]] = defaultdict(_nested_dd_factory)
        self.thresholds: Dict[str, float] = {}
        self.temp_data = []

    def add_calibration_sample(self, score: float, is_sufficient: bool, facet_type: str, text_length: int):
        self.temp_data.append({
            "s": float(score), "y": bool(is_sufficient), "ft": str(facet_type), "len": int(text_length)
        })

    def finalize(self):
        by_ft = defaultdict(list)
        for d in self.temp_data:
            by_ft[d["ft"]].append(d)
            
        for ft, items in by_ft.items():
            if self.use_mondrian:
                lengths = [x["len"] for x in items]
                if not lengths: continue
                med_len = float(np.median(lengths))
                
                short_neg = [x["s"] for x in items if x["len"] <= med_len and not x["y"]]
                long_neg  = [x["s"] for x in items if x["len"] > med_len and not x["y"]]
                
                # Fallback to single bucket if sparse
                if len(short_neg) < self.n_min or len(long_neg) < self.n_min:
                    all_neg = [x["s"] for x in items if not x["y"]]
                    all_neg.sort()
                    self.bins[ft]["all"] = all_neg
                else:
                    short_neg.sort()
                    long_neg.sort()
                    self.bins[ft]["short"] = short_neg
                    self.bins[ft]["long"] = long_neg
                    self.thresholds[ft] = med_len
            else:
                neg = [x["s"] for x in items if not x["y"]]
                neg.sort()
                self.bins[ft]["all"] = neg
        
        self.temp_data = []

    def get_p_value(self, score: float, facet_type: str, text_length: int) -> float:
        """
        Get p-value with robust fallback.

        NEVER raises. Fail-closed returns 1.0 if no calibration data.
        """
        import os
        debug = os.environ.get("TRIDENT_DEBUG_PVALUE", "0") == "1"

        # Canonicalize facet_type (BRIDGE_HOP1/2/.../BRIDGE -> BRIDGE_HOP)
        original_ft = facet_type
        if facet_type.startswith("BRIDGE_HOP") or facet_type == "BRIDGE":
            facet_type = "BRIDGE_HOP"

        ft_bins = self.bins.get(facet_type)
        if not ft_bins:
            if debug:
                print(f"[PVALUE DEBUG] NO_FT_BINS original_ft={original_ft} canon_ft={facet_type} "
                      f"available={list(self.bins.keys())} -> p=1.0")
            return 1.0

        # Prefer "all" as stable fallback (this is what calibrator usually saves)
        neg_scores = None
        if "all" in ft_bins and ft_bins["all"]:
            neg_scores = ft_bins["all"]

        # Otherwise fall back to any known partitions that exist
        if neg_scores is None:
            short = ft_bins.get("short", [])
            long = ft_bins.get("long", [])
            if short or long:
                neg_scores = sorted(list(short) + list(long))

        if not neg_scores:
            if debug:
                print(f"[PVALUE DEBUG] EMPTY_BINS ft={facet_type} keys={list(ft_bins.keys())} -> p=1.0")
            return 1.0

        # p-value computation (empirical right-tail)
        import bisect
        neg_scores_sorted = neg_scores if isinstance(neg_scores, list) else list(neg_scores)
        neg_scores_sorted.sort()

        n = len(neg_scores_sorted)
        idx = bisect.bisect_left(neg_scores_sorted, score)
        ge = n - idx
        p = (ge + 1.0) / (n + 1.0)

        if debug:
            print(f"[PVALUE DEBUG] HIT ft={facet_type} (orig={original_ft}) n={n} score={score:.4f} p={p:.6f} "
                  f"keys={list(ft_bins.keys())}")
        return float(p)

    def to_dict(self) -> Dict:
        return {
            "n_min": self.n_min,
            "use_mondrian": self.use_mondrian,
            "bins": _to_plain(self.bins),
            "thresholds": _to_plain(self.thresholds)
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SelectionConditionalCalibrator":
        obj = cls(n_min=data["n_min"], use_mondrian=data["use_mondrian"])
        
        # FIX: Use named factory function here too
        obj.bins = defaultdict(_nested_dd_factory)
        
        plain_bins = data.get("bins", {})
        for ft, inner in plain_bins.items():
            for k, v in inner.items():
                obj.bins[ft][k] = v
                
        obj.thresholds = data.get("thresholds", {})
        return obj

# --- RESTORED HELPER FOR IMPORT COMPATIBILITY ---

class CalibrationMonitor:
    def __init__(self, *args, **kwargs):
        self.stats = defaultdict(int)

    def update(self, *args, **kwargs):
        pass

    def record(self, *args, **kwargs):
        pass
        
    def get_metrics(self):
        return {}
