"""
Unit tests for Data Quality Validation.
"""

from src.validation.validate import validate_data


def test_validation_execution():
    res = validate_data("configs/config.yaml")
    assert "valid" in res
    assert "quarantined" in res
    assert res["valid"] > 0
    assert res["quarantined"] > 0  # Proves error quarantine caught the simulated invalid records
