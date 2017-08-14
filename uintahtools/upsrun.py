"""Module for running all Uintah input files in a given folder.

Usually used in conjunction with the upschanger module to generate a
simulation test suite.

"""
import os
import re
import sys
import subprocess
from pathlib import Path

from ruamel.yaml import YAML
from uintahtools import CONFIG

class Suite:
    """Class to keep track of the entire simulation suite."""

    def __init__(self, folder):
        self.files = self.find_files(folder)
        self.UINTAHPATH = self.get_uintahpath()
    
    def get_uintahpath(self):
        yaml = YAML()
        settings = yaml.load(Path(CONFIG))
        return settings['uintahpath']

    def find_files(self, folder):
        return [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".ups")]

    def logfile(self, ups):
        """Return a file handle to a log file corresponding to the supplied ups file."""
        # logfilename is name of ups-file with extension "ups" swapped with "log"
        logname = re.sub(r'\.ups$', ".log", os.path.basename(ups))
        return open(os.path.join(os.path.dirname(ups),logname), "w")

    def run(self):
        """Run all the files in directory, directing stdout to a specified logfile."""
        testsuite = {upsfile: self.logfile(upsfile) for upsfile in self.files}

        processes = [subprocess.Popen([self.UINTAHPATH, inputfile], stdout=logfile, stderr=logfile)
                        for inputfile, logfile in testsuite.items()]
        
        print()
        print("Pid   Input file")
        [print(p.pid, p.args[1]) for p in processes]
        print()
        print("Waiting for completion... Cancel all with ctrl+c")
        [p.wait() for p in processes]
        print("All simulations finished!")