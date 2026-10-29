import pytest

from app.domain import Chunk, SearchResult
from app.rag.grounding import context_coverage, has_relevant_context


def context(text):
    return [SearchResult(Chunk("1", text, "sample.md", None, 0), 0.999)]


def test_nonempty_high_scoring_unrelated_results_do_not_pass_gate():
    assert not has_relevant_context(
        "公司VPN服务器的IP地址是多少？", context("公司每位正式员工年度学习预算为5000元。")
    )
    assert not has_relevant_context("请问月球基地如何建设？", [])


def test_relevant_context_and_configurable_gate():
    results = context("值班人员确认P0事件后应在10分钟内通知安全负责人。")
    assert has_relevant_context("请问P0事件需要在几分钟内通知负责人？", results)
    assert not has_relevant_context("的了是在？", results)
    assert has_relevant_context("月球基地如何建设？", results, minimum_coverage=0)
    assert not has_relevant_context("月球基地如何建设？", [], minimum_coverage=0)
    assert 0 <= context_coverage("P0事件", results) <= 1


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_invalid_coverage_setting_is_rejected(value):
    from pydantic import ValidationError

    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(context_min_coverage=value)
