"""Script for running all Uintah input files in a given folder.

Usually used in conjunction with the ups-changer script to generate a
simulation test suite.

"""

import click
import os
import sys

@click.command()
@click.argument("dir", default=".")
def run(dir):
    """Program that starts multiple instances of Uintah on all ups-files in dir."""
    click.echo("Running ...")

if __name__ == "__main__":
    run()