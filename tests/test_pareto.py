from trident.candidates import Passage
from trident.facets import Facet
from trident.pareto import ParetoKnapsack


def score(_passage: Passage, _facet: Facet) -> float:
    return 0.9


def pvalue_fn(score: float, _bucket: str) -> float:
    return 1.0 - score


def test_pareto_selects_within_budget():
    facets = [Facet("f1", "ENTITY", {"mention": "A"})]
    passages = [Passage("p1", "A", cost=10), Passage("p2", "B", cost=20)]
    optimizer = ParetoKnapsack(budget=15, relaxed_alpha=0.2)
    result = optimizer.run(facets, passages, score_fn=score, pvalue_fn=pvalue_fn)
    assert len(result.selected_passages) == 1
    assert result.selected_passages[0].pid == "p1"
    assert result.total_cost <= 15
