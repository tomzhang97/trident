from trident.calibration import ReliabilityCalibrator
from trident.candidates import Passage
from trident.config import FacetConfig, SafeCoverConfig
from trident.facets import Facet
from trident.safe_cover import SafeCoverAlgorithm


def constant_score(_passage: Passage, _facet: Facet) -> float:
    return 0.9


def bucket(_passage: Passage, facet: Facet) -> str:
    return facet.facet_type


def test_safe_cover_selects_passage_when_threshold_met():
    facets = [Facet("f1", "ENTITY", {"mention": "Einstein"})]
    passages = [Passage("p1", "Albert Einstein wrote many papers.", cost=50)]
    calibrator = ReliabilityCalibrator()
    calibrator.tables["ENTITY"] = [(0.0, 1.0), (0.9, 0.01), (1.0, 0.0)]
    config = SafeCoverConfig(per_facet={"f1": FacetConfig(alpha=0.05, max_tests=1)})
    algo = SafeCoverAlgorithm(calibrator, config, score_fn=constant_score, bucket_fn=bucket)
    result = algo.run(facets, passages)
    assert len(result.selected_passages) == 1
    assert result.selected_passages[0].pid == "p1"
    assert not result.abstained
