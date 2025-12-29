"""Tests for Safe-Cover algorithm."""

from trident.calibration import ReliabilityCalibrator
from trident.candidates import Passage
from trident.config import FacetConfig, SafeCoverConfig
from trident.facets import Facet, FacetType
from trident.safe_cover import SafeCoverAlgorithm, AbstentionReason


def test_safe_cover_selects_passage_when_threshold_met():
    """Test that passages are selected when p-values are below threshold."""
    facets = [Facet(
        facet_id="f1",
        facet_type=FacetType.ENTITY,
        template={"mention": "Einstein"}
    )]
    passages = [Passage(pid="p1", text="Albert Einstein wrote many papers.", cost=50)]

    calibrator = ReliabilityCalibrator()
    calibrator.tables["ENTITY"] = [(0.0, 1.0), (0.9, 0.01), (1.0, 0.0)]

    config = SafeCoverConfig(per_facet_alpha=0.1)
    algo = SafeCoverAlgorithm(config=config, calibrator=calibrator)

    # P-values indicating the passage supports the facet
    p_values = {("p1", "f1"): 0.05}  # Below threshold

    result = algo.run(facets, passages, p_values)
    assert len(result.selected_passages) == 1
    assert result.selected_passages[0].pid == "p1"
    assert not result.abstained


def test_safe_cover_carries_calibrator_version_into_certificates():
    """Certificates should record the calibrator version used for the run."""
    facets = [Facet(
        facet_id="f1",
        facet_type=FacetType.ENTITY,
        template={"mention": "Einstein"}
    )]
    passages = [Passage(pid="p1", text="Albert Einstein wrote many papers.", cost=50)]

    calibrator = ReliabilityCalibrator(version="cal_v123")
    calibrator.tables["ENTITY"] = [(0.0, 1.0), (0.9, 0.01), (1.0, 0.0)]

    config = SafeCoverConfig(per_facet_alpha=0.1)
    algo = SafeCoverAlgorithm(config=config, calibrator=calibrator)

    p_values = {("p1", "f1"): 0.05}

    result = algo.run(facets, passages, p_values)

    assert result.certificates, "Expected a certificate for the covered facet"
    assert result.certificates[0].calibrator_version == "cal_v123"


def test_safe_cover_records_bin_size_from_legacy_bins():
    """Certificates should capture bin size even when bins are simple lists."""
    facets = [Facet(
        facet_id="f1",
        facet_type=FacetType.ENTITY,
        template={"mention": "Einstein"}
    )]
    passages = [Passage(pid="p1", text="Albert Einstein wrote many papers.", cost=50)]

    calibrator = ReliabilityCalibrator(version="cal_v123")
    calibrator.tables["ENTITY"] = [(0.0, 1.0), (0.9, 0.01), (1.0, 0.0)]
    calibrator.bins["default"] = [0.2, 0.3, 0.4]

    config = SafeCoverConfig(per_facet_alpha=0.1)
    algo = SafeCoverAlgorithm(config=config, calibrator=calibrator)

    p_values = {("p1", "f1"): 0.05}

    result = algo.run(facets, passages, p_values)

    assert result.certificates, "Expected a certificate for the covered facet"
    assert result.certificates[0].bin_size == 3


def test_safe_cover_abstains_on_no_facets():
    """Test that Safe-Cover abstains when no facets are provided."""
    facets = []  # No facets
    passages = [Passage(pid="p1", text="Some text.", cost=50)]

    calibrator = ReliabilityCalibrator()
    config = SafeCoverConfig(per_facet_alpha=0.1)
    algo = SafeCoverAlgorithm(config=config, calibrator=calibrator)

    p_values = {}
    result = algo.run(facets, passages, p_values)

    assert result.abstained
    assert result.abstention_reason == AbstentionReason.NO_FACETS
    assert len(result.selected_passages) == 0


def test_safe_cover_abstains_on_no_passages():
    """Test that Safe-Cover abstains when no passages are provided."""
    facets = [Facet(
        facet_id="f1",
        facet_type=FacetType.ENTITY,
        template={"mention": "Einstein"}
    )]
    passages = []  # No passages

    calibrator = ReliabilityCalibrator()
    config = SafeCoverConfig(per_facet_alpha=0.1)
    algo = SafeCoverAlgorithm(config=config, calibrator=calibrator)

    p_values = {}
    result = algo.run(facets, passages, p_values)

    assert result.abstained
    assert result.abstention_reason == AbstentionReason.NO_PASSAGES
    assert len(result.selected_passages) == 0
    assert result.uncovered_facets == ["f1"]


def test_safe_cover_handles_uncoverable_facets():
    """Test that Safe-Cover handles facets with no covering passages."""
    facets = [
        Facet(facet_id="f1", facet_type=FacetType.ENTITY, template={"mention": "A"}),
        Facet(facet_id="f2", facet_type=FacetType.ENTITY, template={"mention": "B"}),
    ]
    passages = [Passage(pid="p1", text="About A", cost=50)]

    calibrator = ReliabilityCalibrator()
    config = SafeCoverConfig(per_facet_alpha=0.1)
    algo = SafeCoverAlgorithm(config=config, calibrator=calibrator)

    # Only f1 has a covering passage, f2 has p-value=1.0 (above threshold)
    p_values = {
        ("p1", "f1"): 0.05,  # Covers f1
        ("p1", "f2"): 1.0,   # Does not cover f2
    }

    result = algo.run(facets, passages, p_values)

    # Should abstain because f2 cannot be covered
    assert result.abstained
    assert result.abstention_reason == AbstentionReason.NO_COVERING_PASSAGES


def test_safe_cover_coverage_calculation():
    """Test coverage metrics calculation."""
    facets = [
        Facet(facet_id="f1", facet_type=FacetType.ENTITY, template={"mention": "A"}),
        Facet(facet_id="f2", facet_type=FacetType.ENTITY, template={"mention": "B"}),
    ]
    passages = [
        Passage(pid="p1", text="About A", cost=50),
        Passage(pid="p2", text="About B", cost=50),
    ]

    calibrator = ReliabilityCalibrator()
    config = SafeCoverConfig(per_facet_alpha=0.1, abstain_on_infeasible=False)
    algo = SafeCoverAlgorithm(config=config, calibrator=calibrator)

    # Both facets can be covered
    p_values = {
        ("p1", "f1"): 0.05,
        ("p2", "f2"): 0.05,
    }

    result = algo.run(facets, passages, p_values)

    assert not result.abstained
    assert len(result.covered_facets) == 2
    assert len(result.uncovered_facets) == 0


if __name__ == "__main__":
    print("Running test_safe_cover_selects_passage_when_threshold_met...")
    test_safe_cover_selects_passage_when_threshold_met()
    print("  PASSED")

    print("Running test_safe_cover_abstains_on_no_facets...")
    test_safe_cover_abstains_on_no_facets()
    print("  PASSED")

    print("Running test_safe_cover_abstains_on_no_passages...")
    test_safe_cover_abstains_on_no_passages()
    print("  PASSED")

    print("Running test_safe_cover_handles_uncoverable_facets...")
    test_safe_cover_handles_uncoverable_facets()
    print("  PASSED")

    print("Running test_safe_cover_coverage_calculation...")
    test_safe_cover_coverage_calculation()
    print("  PASSED")

    print("\nAll tests passed!")
