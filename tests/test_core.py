import pytest
import malliavin_regression as mr

def test_api_imports():
    for name in ["generate_data", "brownian", "gradient_free_descent", "f0"]:
        assert hasattr(mr, name)

def test_generate_runs_smoke():
    d = mr.generate_data(10, 5, noise_var=1, name_model="sinus", show=False)
    assert d is not None

