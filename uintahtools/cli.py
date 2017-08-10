"""Command line interface for UintahTools."""

import click
import os

from uintahtools import UPS, Suite

@click.group()
def cli():
    """Command line scripts that simplify working with Uintah ups input files."""
    pass

@cli.command("generate", short_help="generate simulation suite")
@click.argument("ups", type=click.File())
@click.argument("yaml", type=click.File())
@click.option("--run", is_flag=True, help="run all generated input files")
def generate(ups, yaml, run):
    """Generate simulation suite based on a UPS file with settings from a YAML file."""
    folder = UPS(ups, yaml).generate_ups()
    if run:
        folder_run(os.getcwd())
    
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
