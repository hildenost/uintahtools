"""Tests for module runups."""

import pytest

from uintahtools.runups import *

def test_initialization():
    suite = Suite("tests/")
    assert suite.files == ["tests/test.ups"]