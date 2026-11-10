"""Parsing, explanation, aggregation and metric conventions."""
from trialmatch.metrics import classification_report, evidence_scores
from trialmatch.pipeline import aggregate, explain, parse_output


def test_parse_json_and_fallbacks():
    p = parse_output('{"label": "not enough information", "evidence": [1, 9]}<|im_end|>', "inclusion", {0, 1, 2})
    assert p["label"] == "not enough information" and p["evidence"] == [1] and p["hallucinated_ids"] == [9]
    p = parse_output('{"label": "excluded", "evidence": [0', "exclusion", {0, 1})   # truncated JSON
    assert p["label"] == "excluded" and p["evidence"] == [0]
    assert parse_output('{"label": "included"}', "exclusion", {0})["label"] == "invalid"  # wrong label set
    assert parse_output("garbage", "inclusion", {0})["label"] == "invalid"


def test_explanation_flags_absent_evidence():
    assert "verified" in explain("exclusion", "not excluded", [])
    assert "[S2]" in explain("inclusion", "included", [(2, "Age 60.")])


def _r(t, l, e=(0,)):
    return {"criterion": f"{t}-{l}", "criterion_type": t, "label": l, "evidence": list(e)}


def test_aggregation_rules():
    assert aggregate([_r("inclusion", "included"), _r("exclusion", "excluded")])["verdict"].startswith("Likely ineligible")
    assert aggregate([_r("inclusion", "included"), _r("inclusion", "not enough information")])["verdict"].startswith("Undetermined")
    s = aggregate([_r("inclusion", "included"), _r("exclusion", "not excluded", ())])
    assert s["verdict"].startswith("Potentially eligible") and s["needs_verification"]


def test_evidence_metric_conventions():
    m = evidence_scores([[0, 1], [], []], [[1], [], [3]])
    assert m["micro"]["tp"] == 1 and m["micro"]["fp"] == 1 and m["micro"]["fn"] == 1
    assert m["macro_nonempty_gold"]["n_records"] == 1
    assert abs(m["macro_nonempty_gold"]["recall"] - 0.5) < 1e-9
    assert m["empty_gold"]["abstention_accuracy"] == 0.5


def test_classification_invalid_counts_as_error():
    rep = classification_report(["included", "not included"], ["included", "invalid"], ["included", "not included"])
    assert rep["accuracy"] == 0.5 and "invalid" in rep["confusion"]["cols_pred"]
