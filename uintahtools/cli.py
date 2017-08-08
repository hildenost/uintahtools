"""Command line interface for UintahTools."""

import click

from uintahtools import UPS

@click.group()
def cli():
    """Command line scripts that simplify working with Uintah ups input files."""
    click.echo("Hello, world!")

@cli.command("generate", short_help="generate simulation suite")
@click.argument("ups", type=click.File())
@click.argument("yaml", type=click.File())
def generate(ups, yaml):
    """Generate simulation suite based on a UPS file with settings from a YAML file."""
    inputups = UPS(ups, yaml)
    print(inputups)
    inputups.generate_ups()
    
@cli.command("run", short_help="run the simulation suite")
@click.argument("folder")
@click.option('-d', is_flag=True, help="use dbg executable in stead of opt")
def run(folder, d):
    """Run all Uintah input files residing in FOLDER."""
    click.echo("Running all ups files in folder", folder)
    if d:
        click.echo("Using the debug executable.")
