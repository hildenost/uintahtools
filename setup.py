from setuptools import setup, find_packages

setup(  name="uintahtools",
        version="0.1dev",
        description="A bundle of small scripts that simplify working with Uintah input files.",
        long_description="""\
        """,
        author="Hilde Aas Nøst",
        author_email="hilde.nost@gmail.com",
        packages=find_packages(exclude="tests"),
        install_requires=['lxml>=3.8', 'click>=6.7', 'pyyaml>=3.12'],
        entry_points="""
            [console_scripts]
            uintahtools=uintahtools.cli:cli
        """)