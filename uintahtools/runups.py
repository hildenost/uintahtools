"""Module for running all Uintah input files in a given folder.

Usually used in conjunction with the upschanger module to generate a
simulation test suite.

Wishlist:
    [ ] Header: Print out the time the simulation started
    [ ] Ability to view `tail -f <name>.log while running
    [ ] Ability to kill the pid of choice
    [ ] Ability to get <time> out of <total_time> to see how far the
            simulation has run

"""
import cmd
import os
import re
import sys
import subprocess
from pathlib import Path

import colorama
from colorama import Fore
from ruamel.yaml import YAML
from uintahtools import CONFIG
from uintahtools import UPS

colorama.init(autoreset=True)

class Prompt(cmd.Cmd):
    """Command processor to survey the simulations."""

    prompt = Fore.LIGHTGREEN_EX + "uintahtools $ " + Fore.RESET

    ruler = '-'

    def __init__(self, processes):
        self.pids = set(p.pid for p, log in processes)
        self.processes = {str(p.pid): (p, log) for p, log in processes}
        super().__init__()

    def cmdloop(self, intro=None):
        print()
        print("Use of prompt:")
        print("\tkill [pid]|all\tkill process by id or all processes")
        print("\tprod [pid]\tget process progress as time out of total time")
        print()
        return cmd.Cmd.cmdloop(self, intro)

    def do_prod(self, pid):
        """prod [pid]\tget process progress as time out of total time"""
        print("Checking out the process")
        print(pid, self.processes[pid][1].name, self.processes[pid][0].args[1])
        
        # Get max time
        inputfile = self.processes[pid][0].args[1]
        ups = UPS(inputfile)
        max_time = float(ups.search_tag("maxTime").text)

        tail = subprocess.run(["tail", self.processes[pid][1].name],
                stdout=subprocess.PIPE,
                universal_newlines=True)
        result = re.findall(
            "Time=(?P<time>(0|[1-9]\d*)(\.\d+)?([eE][+\-]?\d*)?)",
            tail.stdout,
            re.MULTILINE
        )
        current_time = float(result[-1][0])
        
        percentage_done = current_time/max_time*100

        print("At time {current_time} of {max_time},"
            " simulation is {percentage:f}% done.".format(
                current_time=current_time,
                max_time=max_time,
                percentage=percentage_done
            ))
        
        count = current_time
        total = max_time
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[{bar}] {percents}%\n'.format(bar=bar, percents=percents))
        sys.stdout.flush()
    
    def do_kill(self, pid):
        """kill [pid]|all\tkill process by id or all processes"""
        # TODO: make use of Popen.kill() in stead
        if pid == "all":
            print("Killing all simulations in {pids}".format(pids=self.pids))
            for id in self.pids:
                print("Killing process number {id}".format(id=id))
                subprocess.run(["kill", str(id)])
            self.pids.clear()
            print("All processes killed.")
        elif pid.isdigit() and int(pid) in self.pids:
            print("Killing simulation with process id {pid}...".format(pid=pid))
            subprocess.run(["kill", pid])
            self.pids.remove(int(pid))
            print("Process killed.")
        else:
            print("{pid} not valid.".format(pid=pid, pids=self.pids))
        if self.pids:
            print("Running processes: {pids}".format(pids=self.pids))
        else:
            print("No processes left. Quitting prompt.")
            return True
    
    def do_EOF(self, line):
        """Make it possible to use ctrl+d to exit."""
        return True

    def do_exit(self, line):
        """exit\texit prompt"""
        return True

    def postloop(self):
        print("Quitting...")
        self.do_kill("all")
        print()


class Suite:
    """Class to keep track of the entire simulation suite."""

    def __init__(self, folder):
        self.files = self.find_files(folder)
        self.UINTAHPATH = self.get_uintahpath()
        self.logfiles = []
        self.testsuite = {}
    
    def get_uintahpath(self):
        yaml = YAML()
        settings = yaml.load(Path(CONFIG))
        return settings['uintahpath']

    def find_files(self, folder):
        return [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".ups")]

    def logfile(self, ups):
        """Return a file handle to a log file corresponding to the supplied ups file."""
        # logfilename is name of ups-file with extension "ups" swapped with "log"
        self.logfiles.append(re.sub(r'\.ups$', ".log", os.path.basename(ups)))
        return open(os.path.join(os.path.dirname(ups), self.logfiles[-1]), "w")

    def run(self):
        """Run all the files in directory, directing stdout to a specified logfile."""

        # TODO: Make sure the log file "deletes itself". No reason to save all output...
        self.testsuite = {upsfile: self.logfile(upsfile) for upsfile in self.files}

        processes = [(subprocess.Popen([self.UINTAHPATH, inputfile],
                            stdout=logfile,
                            stderr=subprocess.STDOUT,
                            universal_newlines=True
                            ), logfile) for inputfile, logfile in self.testsuite.items()]
        
        print()
        print("Pid   Input file")
        [print(p.pid, p.args[1]) for p, log in processes]
        print()
        print("Waiting for completion... Cancel all processes with ctrl+c")

        Prompt(processes).cmdloop()
        print("All simulations finished!")