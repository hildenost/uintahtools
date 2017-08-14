"""Command line interface for UintahTools."""
import os
import readline

import click
from pathlib import Path
from ruamel.yaml import YAML

from uintahtools import UPS, Suite, CONFIG, CONFIGSTR

# Tab completion for user prompt
readline.set_completer_delims(" \t\n")
readline.parse_and_bind("tab: complete")

def initialize_config():
    """Initializes the configuration file if it is not set.

    Also prompts for the uintah executable if this is not set.
    """

    # Check settings. We need a path to the Uintah executable
    yaml = YAML()
    p = Path(CONFIG)
    if not p.exists():
        p.touch()
        yaml.dump(yaml.load(CONFIGSTR), p)

    settings = yaml.load(p)
    if not settings["uintahpath"]:
        while True:
            suggestion = os.path.expanduser(
                input("UINTAHPATH is not set. Where is the Uintah executable?\t")
            )
        
            # Verify that the input path is valid and existing before setting
            # Also expand the path so it is absolute before storing
            try:
                Path(suggestion).exists()
            except FileNotFoundError:
                print("Please enter a valid path.")
            else:
                settings["uintahpath"] = suggestion
                break
        yaml.dump(settings, p)

@click.group()
def cli():
    """Command line scripts that simplify working with Uintah ups input files."""
    initialize_config()


@cli.command("generate", short_help="generate simulation suite")
@click.argument("ups", type=click.File())
@click.argument("yaml", type=click.File())
@click.option("--run", is_flag=True, help="run all generated input files")
def generate(ups, yaml, run):
    """Generate simulation suite based on a UPS file with settings from a YAML file."""
    folder = UPS(ups, yaml).generate_ups()
    if run:
        folder_run(os.getcwd(), d=False)
    
@cli.command("run", short_help="run the simulation suite")
@click.argument("folder")
@click.option('-d', is_flag=True, help="use dbg executable in stead of opt")
def folder_run(folder, d):
    """Run all Uintah input files residing in FOLDER."""
    click.echo("Running all ups files in folder")
    if d:
        click.echo("Using the debug executable.")
    suite = Suite(folder)
    suite.run()
