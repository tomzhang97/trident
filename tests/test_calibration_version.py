"""Tests for calibration version metadata handling."""

from trident.calibration import ReliabilityCalibrator


def test_calibrator_load_preserves_version(tmp_path):
    """Loading a calibrator should restore its version metadata."""
    path = tmp_path / "calibrator.json"

    calibrator = ReliabilityCalibrator(use_mondrian=False, version="v-test")
    calibrator.save(path)

    loaded = ReliabilityCalibrator.load(path)

    assert loaded.version == "v-test"
