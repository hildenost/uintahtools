"""Module for running all Uintah input files in a given folder.

Usually used in conjunction with the upschanger module to generate a
simulation test suite.

"""
import os
import re
import sys
import subprocess

class Suite:
    """Class to keep track of the entire simulation suite."""

    def __init__(self, folder):
        self.files = self.find_files(folder)

    def find_files(self, folder):
        return [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".ups")]

    def logfile(self, ups):
        """Return a file handle to a log file corresponding to the supplied ups file."""
        # logfilename is name of ups-file with extension "ups" swapped with "log"
        logname = re.sub(r'\.ups$', ".log", os.path.basename(ups))
        return open(os.path.join(os.path.dirname(ups),logname), "w")

    def run(self):
        # Generate logfiles for all files in folder and make a dict of inputfile: logfile
        testsuite = {upsfile: self.logfile(upsfile) for upsfile in self.files}

        processes = [subprocess.Popen([UINTAHPATH, inputfile], stdout=logfile)
                        for inputfile, logfile in testsuite.items()]
        
        
