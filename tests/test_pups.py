"""Tests for module pups."""

import pytest

from uintahtools.pups import normalize, header, construct_cmd, \
                                get_timestep

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
@pytest.fixture()
def before():
    print("\nBefore every test...............")

def test_headers_for_px(before):
    assert header("p.x") == ["time", "patch", "matl", "partId", "x", "y", "z"]

def test_headers_for_pporepressure(before):
    assert header("p.porepressure") == ["time", "patch", "matl", "partId", "pw"]

def test_undefined_header(before):
    var = "p.undefined"
    assert header(var) == ["time", "patch", "matl", "partId", var]

# Tests for the main body
@pytest.fixture()
def basics():
    puda = "/home/hilde/trunk/opt/StandAlone/puda"
    # uda = "/home/hilde/trunk/results/oedo.uda.000"
    uda = "/home/hilde/trunk/tests/1Dworking.uda"
    return (puda, uda)

def test_get_timestep(basics):
    puda, uda = basics
    time = 0.013
    assert get_timestep(time, uda) == "1300"

def test_get_timestep_time_between_steps(basics):
    puda, uda = basics
    time = 0.0135
    assert get_timestep(time, uda) == "1300"

def test_get_timestep_out_of_bounds(basics):
    puda, uda = basics
    time = 2.0
    with raises  get_timestep(time, uda) == "1300"

def test_construct_cmd(basics):
    var = "p.x"
    puda, uda = basics
    assert construct_cmd(var, uda) == [puda, "-partvar", var, uda]

def test_construct_cmd_with_time(basics):
    var = "p.x"
    puda, uda = basics
    time = 0.2
    assert construct_cmd(var, uda, time=time) == \
                        [puda, "-partvar", var, "-timesteplow", str(time),
                                                "-timestephigh", str(time), uda]