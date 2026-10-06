"""Problem statements shown in the dashboard: complete, filled from the data, consistent with the problem type."""
import pytest

from core.problem import load_instance
from ui.statements import SHOWN, STATEMENTS, checklist, formulation, network_figure, statement


@pytest.mark.parametrize("name", SHOWN)
def test_statement_complete(name):
    prob = load_instance(name)
    title, scenario, task = statement(name, prob)
    assert title and task and "{" not in scenario          # every placeholder was filled from the data
    assert f"{int(prob.supply.sum()):,}" in scenario or name == "A5"
    holds = [ok for _, ok, _ in checklist(prob)]
    assert all(holds) == prob.is_standard                   # the checklist agrees with the problem type
    blocks = formulation(prob)
    assert blocks and all(tex for _, tex in blocks)
    assert len(network_figure(prob).data) >= 3               # routes + sources + destinations


def test_every_statement_is_shown():
    assert set(STATEMENTS) == set(SHOWN)


def test_a5_mentions_the_proposed_plan_cost():
    assert "Rs 795" in statement("A5", load_instance("A5"))[1]
