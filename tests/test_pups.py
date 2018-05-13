"""Tests for module pups."""

import pytest

from uintahtools.pups import *

# Tests for normalize function


def test_normalize_1wrt5():
    assert normalize(1, varmax=5) == 0.2


def test_normalize_1wrt5_flip():
    assert normalize(1, varmax=5, flip=True) == 0.8


def test_normalize_1wrt5_varmin__2_varmax_2():
    assert normalize(1, varmax=2, varmin=-2) == 0.75


def test_normalize_6wrt5():
    assert normalize(6, varmax=5) == 1.2

# Tests for header function


def test_headers_for_px():
    assert header("p.x") == ["time", "patch", "matl", "partId", "x", "y", "z"]


def test_headers_for_pporepressure():
    assert header("p.porepressure") == [
        "time", "patch", "matl", "partId", "pw"]


def test_undefined_header():
    var = "p.undefined"
    assert header(var) == ["time", "patch", "matl", "partId", var]

# # Tests for the main body


@pytest.fixture()
def basics():
    puda = "/home/hilde/trunk/opt/StandAlone/puda"
    uda = "tests/test.uda"
    return timesteps_parse(cmd_run([puda, "-timesteps", uda]))


# # Tests for timesteps
def test_get_timestep(basics):
    timedict = basics
    time = 0.013
    assert timesteps_get(timedict, times=time) == ["130"]


def test_get_timestep_every_30000th_timestep(basics):
    timedict = basics
    f = 3
    assert timesteps_get(timedict, frequency=f) == [
        "1", "30000", "60001", "90001", "100001-"]


def test_get_timestep_negative_time(basics):
    timedict = basics
    time = -0.1
    assert timesteps_get(timedict, time) == ["1"]


def test_get_timestep_negative_time_in_list(basics):
    timedict = basics
    time = [-1, 1]
    assert timesteps_get(timedict, time) == ["1", "10000"]


def test_get_timestep_times(basics):
    timedict = basics
    time = [0.013, 0.1]
    assert timesteps_get(timedict, time) == ["130", "1000"]


def test_get_timestep_time_between_steps(basics):
    timedict = basics
    time = 0.01355
    assert timesteps_get(timedict, time) == ["136"]


def test_get_timestep_out_of_bounds(basics):
    timedict = basics
    time = 2.0
    assert timesteps_get(timedict, time) == ["10001"]

# Make the command to go with puda
# TODO, though YAGNI: Check for highs and lows in nested timeseries


@pytest.fixture()
def udas():
    puda = "/home/hilde/trunk/opt/StandAlone/puda"
    uda = "tests/test.uda"
    return (puda, uda)


def test_construct_cmd(udas):
    var = "p.x"
    puda, uda = udas
    assert cmd_make(var, uda) == [puda, "-partvar", var, uda]


def test_construct_cmd_with_time(udas):
    var = "p.x"
    puda, uda = udas
    time = 2000
    assert cmd_make(var, uda, timestep=time) == \
        [puda, "-partvar", var, "-timesteplow", str(time),
         "-timestephigh", str(time), uda]


def test_construct_cmd_with_time_range(udas):
    var = "p.x"
    puda, uda = udas
    time = [2000, 5000]
    assert cmd_make(var, uda, timestep=time) == \
        [puda, "-partvar", var, "-timesteplow", str(2000),
         "-timestephigh", str(5000), uda]

# Testing the dataframe_make


def test_dataframe_make_shape(udas, basics):
    var = "p.x"
    __, uda = udas
    timesteps = ["200", "500", "600", "700"]
    df = dataframe_assemble(var, timesteps, uda)
    rows = 240 * len(timesteps)
    columns = len(header(var))
    assert df.shape == (rows, columns)

# Test get particle IDs


def test_particle_ids_samples(udas):
    __, uda = udas
    assert get_particleIDs(uda) == {0: 0.0125,
                                    85899345920: 0.5125,
                                    167503724544: 0.9875}

# Testing the variablelist


def test_variableslist(udas):
    __, uda = udas
    vardict = {"p.x": "Point",
               "p.color": "double",
               "p.temperature": "double",
               "p.velocity": "Vector",
               "g.velocity": "Vector",
               "p.particleID": "long64",
               "p.stress": "Matrix3",
               "p.externalforce": "Vector",
               "g.internalforce": "Vector",
               "g.externalforce": "Vector",
               "g.mass": "double",
               "p.deformationMeasure": "Matrix3",
               "g.acceleration": "Vector",
               "p.porepressure": "double",
               "p.porepressure+": "double",
               "g.internalfluidforce": "Vector",
               "g.internaldragforce": "Vector",
               "p.fluidvelocity": "Vector",
               "g.fluidvelocity": "Vector",
               "g.fluidacceleration": "Vector"}
    assert variablelist(uda) == vardict
