import pytest

from agentdesk.metrics import Proportion, is_significant


def test_interval_stays_inside_zero_and_one_at_the_edges() -> None:
    """Where the naive normal interval breaks: a perfect score on a small sample."""
    low, high = Proportion(successes=20, total=20).interval()
    assert high == 1.0
    assert 0.8 < low < 1.0  # not the meaningless [1.0, 1.0] a naive interval would give


def test_empty_sample_is_not_a_crash() -> None:
    empty = Proportion(successes=0, total=0)
    assert empty.value == 0.0
    assert empty.interval() == (0.0, 0.0)


def test_interval_narrows_as_the_sample_grows() -> None:
    small = Proportion(successes=45, total=50).interval()
    large = Proportion(successes=450, total=500).interval()
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_a_hundred_cases_cannot_resolve_four_points() -> None:
    """The honesty check: on n=100, 87% and 91% are the same measurement."""
    assert not is_significant(Proportion(87, 100), Proportion(91, 100))


def test_a_large_gap_is_significant() -> None:
    assert is_significant(Proportion(50, 96), Proportion(88, 96))


def test_significance_is_symmetric() -> None:
    before, after = Proportion(50, 96), Proportion(88, 96)
    assert is_significant(before, after) == is_significant(after, before)


@pytest.mark.parametrize(("successes", "total"), [(0, 10), (5, 10), (10, 10)])
def test_value_is_within_its_own_interval(successes: int, total: int) -> None:
    proportion = Proportion(successes, total)
    low, high = proportion.interval()
    assert low <= proportion.value <= high


def test_paired_test_uses_only_the_disagreements() -> None:
    """Items both methods got right carry no information about which is better."""
    from agentdesk.metrics import mcnemar_p_value

    # Twelve items where only the second method succeeded, none the other way: decisive.
    assert mcnemar_p_value(only_first=0, only_second=12) < 0.001
    # An even split: no evidence either way, however many items there are.
    assert mcnemar_p_value(only_first=10, only_second=10) == 1.0


def test_paired_test_is_more_powerful_than_comparing_intervals() -> None:
    """The reason to pair: independent intervals miss a real difference this sample can show."""
    from agentdesk.metrics import Proportion, is_significant, mcnemar_p_value

    # 50 questions: method A gets 26, method B gets 38, and B wins every disagreement.
    assert not is_significant(Proportion(26, 50), Proportion(38, 50))
    assert mcnemar_p_value(only_first=0, only_second=12) < 0.05


def test_no_disagreement_means_no_evidence() -> None:
    from agentdesk.metrics import mcnemar_p_value

    assert mcnemar_p_value(only_first=0, only_second=0) == 1.0


def test_double_encoded_tool_payload_is_decoded() -> None:
    """A provider returned the whole object as a JSON string inside its own first field.

    Validating that directly fails on every field at once — which reads like a broken schema
    rather than the transport quirk it is.
    """
    import json

    from agentdesk.llm.tools import parse_tool_payload

    inner = {"answers_the_question": True, "reasoning": "it does"}
    doubled = json.dumps({"answers_the_question": json.dumps(inner)})
    assert parse_tool_payload(doubled, inner.keys()) == inner


def test_a_string_field_that_is_not_the_payload_is_left_alone() -> None:
    import json

    from agentdesk.llm.tools import parse_tool_payload

    payload = {"answers_the_question": True, "reasoning": '{"unrelated": "json"}'}
    assert parse_tool_payload(json.dumps(payload), payload.keys()) == payload


def test_double_encoded_payload_missing_an_optional_field_is_still_decoded() -> None:
    """Models omit fields that have defaults; requiring every field would miss the common case."""
    import json

    from agentdesk.llm.tools import parse_tool_payload

    schema_fields = {"all_claims_supported", "unsupported_claim"}
    doubled = json.dumps({"all_claims_supported": json.dumps({"all_claims_supported": True})})
    assert parse_tool_payload(doubled, schema_fields) == {"all_claims_supported": True}


def test_a_decoded_object_with_foreign_keys_is_not_unwrapped() -> None:
    """Shape recognition, not guessing: an object that is not this schema is left alone."""
    import json

    from agentdesk.llm.tools import parse_tool_payload

    payload = {"all_claims_supported": json.dumps({"something": "else", "entirely": 1})}
    assert parse_tool_payload(json.dumps(payload), {"all_claims_supported"}) == payload
