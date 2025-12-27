#!/usr/bin/env python3
"""
Diagnostic script to check calibrator loading and p-value computation.
Run this BEFORE the full experiment to verify the calibrator works.
"""

import sys
import json
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_calibrator.py <calibrator.json>")
        sys.exit(1)

    cal_path = sys.argv[1]
    print(f"=" * 60)
    print(f"CALIBRATOR DIAGNOSTIC")
    print(f"=" * 60)
    print(f"Loading: {cal_path}")

    # 1. Check file exists
    if not Path(cal_path).exists():
        print(f"❌ ERROR: File does not exist: {cal_path}")
        sys.exit(1)
    print(f"✅ File exists")

    # 2. Check JSON is valid
    try:
        with open(cal_path) as f:
            data = json.load(f)
        print(f"✅ JSON is valid")
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON: {e}")
        sys.exit(1)

    # 3. Check structure
    print(f"\nCalibrator structure:")
    print(f"  version: {data.get('version', 'MISSING')}")
    print(f"  use_mondrian: {data.get('use_mondrian', 'MISSING')}")

    conformal = data.get("conformal")
    if not conformal:
        print(f"❌ ERROR: No 'conformal' key in calibrator")
        sys.exit(1)

    print(f"  conformal.n_min: {conformal.get('n_min', 'MISSING')}")
    print(f"  conformal.use_mondrian: {conformal.get('use_mondrian', 'MISSING')}")

    bins = conformal.get("bins", {})
    print(f"\n  Facet types in bins: {list(bins.keys())}")

    # 4. Check each facet type
    print(f"\nPer-facet bin details:")
    for ft, ft_bins in bins.items():
        keys = list(ft_bins.keys())
        sizes = {k: len(v) if isinstance(v, list) else "?" for k, v in ft_bins.items()}
        print(f"  {ft}: keys={keys}, sizes={sizes}")

        # Check if "all" exists
        if "all" in ft_bins:
            all_size = len(ft_bins["all"])
            print(f"    ✅ 'all' bucket has {all_size} samples")
        else:
            print(f"    ⚠️  No 'all' bucket - will need short/long")

    # 5. Try loading with ReliabilityCalibrator
    print(f"\n" + "=" * 60)
    print(f"TESTING ReliabilityCalibrator.load()")
    print(f"=" * 60)

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from trident.calibration import ReliabilityCalibrator

        cal = ReliabilityCalibrator.load(cal_path)
        print(f"✅ ReliabilityCalibrator loaded successfully")
        print(f"  use_mondrian: {cal.use_mondrian}")
        print(f"  conformal_calibrator: {cal.conformal_calibrator is not None}")

        if cal.conformal_calibrator:
            cc = cal.conformal_calibrator
            print(f"  bins available: {list(cc.bins.keys())}")
    except Exception as e:
        print(f"❌ ERROR loading ReliabilityCalibrator: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 6. Test p-value computation
    print(f"\n" + "=" * 60)
    print(f"TESTING P-VALUE COMPUTATION")
    print(f"=" * 60)

    import os
    os.environ["TRIDENT_DEBUG_PVALUE"] = "1"

    test_cases = [
        ("ENTITY", 0.9, 100),
        ("ENTITY", 0.5, 100),
        ("ENTITY", 0.1, 100),
        ("RELATION", 0.8, 150),
        ("TEMPORAL", 0.7, 80),
        ("NUMERIC", 0.6, 120),
        ("BRIDGE_HOP", 0.85, 200),
        ("BRIDGE_HOP1", 0.85, 200),  # Test canonicalization
    ]

    for ft, score, text_len in test_cases:
        try:
            p = cal.to_pvalue(score, ft, text_len)
            status = "✅" if p < 1.0 else "⚠️ "
            print(f"  {status} to_pvalue({ft}, score={score}, len={text_len}) = {p:.6f}")
        except Exception as e:
            print(f"  ❌ to_pvalue({ft}, ...) raised: {e}")

    print(f"\n" + "=" * 60)
    print(f"SUMMARY")
    print(f"=" * 60)

    # Check if any p-values are < 1.0
    working = False
    for ft, score, text_len in test_cases[:5]:
        try:
            p = cal.to_pvalue(score, ft, text_len)
            if p < 1.0:
                working = True
                break
        except:
            pass

    if working:
        print(f"✅ Calibrator is working - some p-values < 1.0")
        print(f"\nIf experiments still show 100% abstention, the issue is in:")
        print(f"  1. Pipeline not calling to_pvalue()")
        print(f"  2. Config not loading calibrator path")
        print(f"  3. Multi-GPU worker isolation")
    else:
        print(f"❌ Calibrator NOT working - all p-values = 1.0")
        print(f"\nCheck:")
        print(f"  1. Bins have data (see above)")
        print(f"  2. 'all' bucket exists for each facet type")
        print(f"  3. Calibrator was trained with samples")

if __name__ == "__main__":
    main()
