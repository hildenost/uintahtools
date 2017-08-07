"""Script for changing elements of Uintah input files (*.ups)

Manually editing Uintah input files quickly becomes a hassle when one
wishes to change, delete or add settings in order to run several simulations
with variational properties.

Typical elements to change when creating multiple simulations are:
- Title (should be descriptive)
- Filebase (name of output directory)
- Various simulation flags
- With or without gravity
- Material
    - Material parameters
    - Resolution (particles per cell)
- Loads (Pressure BCs)
- Resolution (cells per direction)

This script parses the input files as xml objects with the help of the Python
`lxml` module. Then it performs the changes to the objects as provided in the json file, before
writing the resulting object tree to a specified output file.

"""

import os
import re
import sys
import unicodedata

from lxml import etree
import yaml

# Input file from command line argument
inputups = sys.argv[1]

# Settings file, in YAML
settings_file = sys.argv[2]
with open(settings_file, "r") as stream:
    settings = yaml.load(stream)

# For properly aligning the resulting ups file
parser = etree.XMLParser(remove_blank_text=True)

tree = etree.parse(inputups, parser)

def slugify(text):
    """
    Removes non-word characters (alphanumerics) and converts spaces to hyphens.
    Also strips leading and trailing whitespace.
    """
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub('[^\w\s-]', '', text).strip()
    return re.sub('[-\s]+', '-', text)

def search_tag(tag, all=False):
    """Return the result of endless search of tag in tree.
    
    By default, the first result is returned, but if all is set
    to True, all results are returned.

    """
    if all:
        return tree.findall(".//"+tag)
    return tree.find(".//"+tag)

def add_subelement(parent, tagname, text=None):
    """Add subelement to parent, with optional text."""
    sub = etree.SubElement(parent, tagname)
    sub.text = text
    return sub

def update_tag(key, value):
    """Update tag with name of key to the value of value."""
    tag = search_tag(key)
    if tag is not None:
        if key == "title":
            # The title is used as basis for the output directory
            # and file name
            tag.text = str(value)

            # Name the output directory, defined by the `filebase` tag
            search_tag("filebase").text = slugify(value) + ".uda"

        elif key == "load_curve":
            # Change the load curve. The entire previous load curve
            # is replaced by the YAML provided one.
            
            # Loop through all time_point tags and delete
            [tag.remove(tp) for tp in search_tag("time_point", all=True)]

            # Add the new load curve
            for timepoint in value:
                test = etree.SubElement(tag, "time_point")
                [add_subelement(test, key, str(value)) for key, value in timepoint.items()]
        else:
            tag.text = str(value)
    else:
        # Tag is not found
        if key == "material_model":
            tag = search_tag("constitutive_model")
            tag.attrib['type'] = value
        else:
            print("WARNING: The tag", key,"is not a valid ups tag. Skipped it.")

def base_tags(tag):
    """Returns the base value of a given parameter (tag).

    In case of the tag being the load curve, the base value could either
    be a time point to be changed or a load to be changed. Only the `middle_time`
    keyword will be implemented as of now.

    """
    if tag == "load_curve":
        # With the load curve, the middle timepoint needs to be extracted,
        # as that is the value to change
        return tree.findall(".//load_curve/time_point/time")[1]
    return search_tag(tag)

def get_values(key):
    if key == "load_curve":
        return settings[key]["middle_time"]
    return settings[key]


def update_given(tag, value):
    tag.text = str(value)

def abbrev(string):
    return ''.join(word[0] for word in string.split('_'))

# Check if we're dealing with a combinatorical approach
if "combine" in settings:
    # If combine holds a value other than true, create a folder where all
    # related files are placed, including a copy of the original input file
    try:
        os.mkdir(settings["combine"])
    except FileExistsError:
        print("Directory", settings["combine"],"already exists. Skipping.")
    os.chdir(settings["combine"])
    tree.write(inputups, pretty_print=True, xml_declaration=True)

    # We are going to combine some parameters into multiple input files
    # Getting parameters to vary
    tags = {key: get_values(key) for key in settings if key != "combine"}
    # Getting the tags to vary
    defaults = {tag: base_tags(tag) for tag in tags}
    # Combining it all into a list of dicts
    combos = [{default: tag.text, other:value} for default, tag in defaults.items()
                                                    for other in defaults
                                                    if other != default
                                                    for value in tags[other]]
    for combo in combos:
        # Create a title from the combos by adding the abbreviated
        # names of the altered parameters plus their values.
        # long_name_parameter: 0.5 adds lnp05 to title.
        title = "-".join(abbrev(key)+str(v).strip() for key, v in combo.items())
        outputups = inputups[:-4] +"-"+ slugify(title) + ".ups"
        [update_given(defaults[tag], combo[update]) for tag in defaults
                                  for update in combo
                                  if tag == update]
        tree.write(outputups, pretty_print=True, xml_declaration=True)

else:
    [update_tag(key, value) for key, value in settings.items()]

    outputups = slugify(search_tag("title").text) + ".ups"

    tree.write(outputups, pretty_print=True, xml_declaration=True)
