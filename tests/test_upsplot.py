"""Tests for module upsplot."""

from uintahtools.upsplot import normalize, header

# Tests for normalize function
def test_normalize_1wrt5():
    assert normalize(1, 5) == 0.2

def test_normalize_1wrt5_flip():
    assert normalize(1, 5, flip=True) == 0.8

def test_normalize_1wrt5_varmin__2_varmax_2():
    assert normalize(1, varmax=2, varmin=-2) == 0.75

def test_normalize_6wrt5():
    assert normalize(6, varmax=5) == 1.2

# Tests for header function
def test_headers_for_px():
    assert header("p.x") == ["time", "patch", "matl", "partId", "x", "y", "z"]

def test_headers_for_pporepressure():
    assert header("p.porepressure") == ["time", "patch", "matl", "partId", "pw"]

def test_undefined_header():
    var = "p.undefined"
    assert header(var) == ["time", "patch", "matl", "partId", var]