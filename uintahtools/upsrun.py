"""Module for running all Uintah input files in a given folder.

Usually used in conjunction with the upschanger module to generate a
simulation test suite.

"""
import os
import sys

import click

class Suite:
    """Class to keep track of the entire simulation suite."""

    def __init__(self, folder):
        self.files = self.find_files(folder)

    def find_files(self, folder):
        return [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".ups")]

    def run(self):
        cmd = "~/trunk/dbg/StandAlone/sus damping/1D.ups"
        os.system(cmd)