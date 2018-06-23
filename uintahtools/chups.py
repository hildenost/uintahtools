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
import unicodedata

from lxml import etree
from ruamel.yaml import YAML, yaml_object

yaml = YAML()


@yaml_object(yaml)
class Material:
    """YAML class for the constitutive model.

    Usage:
    material_model: !material
      type: name_of_model
      materialparam1: some_value
      materialparam2: another_value
      ...
      materialparamn: last_value

    """
    yaml_tag = u"!material"

    def __init__(self, **kwargs):
        pass


class UPS:
    """Class containing the parsed UPS XML tree and related methods."""

    def __init__(self, ups, settings=None):
        self.tree = self.parse_ups(ups)
        self.settings = yaml.load(settings) if settings else None
        self.name = self.create_name()

    def parse_ups(self, ups):
        """Return the parsed XML tree."""
        # For proper aligning of the resulting ups file
        return etree.parse(ups, etree.XMLParser(remove_blank_text=True))

    def update_tag(self, key, value):
        """Update tag with name of key to the value of value."""
        tag = self.search_tag(key)
        if tag is not None:
            if key == "title":
                # The title is used as basis for the output directory
                # and file name
                tag.text = str(value)

                # Name the output directory, defined by the `filebase` tag
                self.search_tag("filebase").text = slugify(value) + ".uda"

            elif key == "load_curve":
                # Change the load curve. The entire previous load curve
                # is replaced by the YAML provided one.

                # Loop through all time_point tags and delete
                [tag.remove(tp)
                 for tp in self.search_tag("time_point", one=False)]

                # Add the new load curve
                for timepoint in value:
                    test = etree.SubElement(tag, "time_point")
                    [self.add_subelement(test, key, str(value))
                     for key, value in timepoint.items()]
            else:
                tag.text = str(value)
        else:
            # Tag is not found
            if key == "material_model":
                tag = self.search_tag("constitutive_model")
                # Resetting tag
                tag.clear()

                # Assigning new material model type
                tag.attrib['type'] = value.type

                # Assigning new child nodes with the material properties provided
                [self.add_subelement(tag, key, str(value)) for key, value in vars(
                    value).items() if key != "type"]
            else:
                print("WARNING: The tag", key,
                      "was not specified in input ups file. Skipped it.")

    def search_tag(self, tagname, one=True):
        """Return the result of endless search of tag in tree.

        By default, the first result is returned, but if all is set
        to True, all results are returned.

        """
        func = self.tree.find if one else self.tree.findall
        res = func(".//"+tagname)
        return res

    def base_tags(self, tag):
        """Returns the base value of a given parameter (tag).

        In case of the tag being the load curve, the base value could either
        be a time point to be changed or a load to be changed. Only the `middle_time`
        keyword will be implemented as of now, presuming it is the second time point.

        """
        return self.search_tag("load_curve/time_point/time")[1] if tag == "load_curve" else self.search_tag(tag)

    def create_name(self):
        """Set the name of the simulation suite family based on the input file title."""
        return self.search_tag("title").text

    def get_values(self, key):
        if key == "load_curve":
            return self.settings[key]["middle_time"]
        return self.settings[key]

    def update_given(self, tag, value):
        tag.text = str(value)

    def add_subelement(self, parent, tagname, text=None):
        """Add subelement to parent, with optional text."""
        sub = etree.SubElement(parent, tagname)
        sub.text = text
        return sub

    def generate_ups(self):
        """Generate the simulation suite based on all combinations from settings."""
        # Check if we're dealing with a combinatorical approach
        if "combine" in self.settings:
            # All generated input files are placed here, including a copy of the original input file
            changedirectory(self.settings["combine"])

            self.tree.write(self.name+".ups", pretty_print=True,
                            xml_declaration=True)

            # We are going to combine some parameters into multiple input files
            # Getting parameters to vary
            tags = {key: self.get_values(key)
                    for key in self.settings if key != "combine"}
            # Getting the tags to vary
            defaults = {tag: self.base_tags(tag) for tag in tags}
            # Expanding it all into a list of dicts
            combos = [{key: value} for key in tags for value in tags[key]]

            for combo in combos:
                # Create a title from the combos by adding the abbreviated
                # names of the altered parameters plus their values.
                # `long_name_parameter: 0.5` adds lnp05 to title.
                title = "-".join(abbrev(key)+str(v).strip()
                                 for key, v in combo.items())
                outputups = self.name + "-" + slugify(title) + ".ups"
                filebase = self.name + "-" + slugify(title) + ".uda"
                self.update_tag("filebase", filebase)
                self.update_tag("title", " ".join([self.name, title]))
                [self.update_given(defaults[tag], combo[update]) for tag in defaults
                 for update in combo
                 if tag == update]
                self.tree.write(outputups, pretty_print=True,
                                xml_declaration=True)
        else:
            [self.update_tag(key, value)
             for key, value in self.settings.items()]
            outputups = slugify(self.search_tag("title").text) + ".ups"
            self.tree.write(outputups, pretty_print=True, xml_declaration=True)
        return os.getcwd()


def abbrev(string):
    """Returns the first letter in each underscore-separated word."""
    return ''.join(word[0] for word in string.split('_'))


def changedirectory(folder):
    """Changes directory to folder. Creates the folder if it does not exist."""
    try:
        os.mkdir(folder)
    except FileExistsError:
        print("Directory", folder,
              "already exists. Skipping creation.")

    os.chdir(folder)


def slugify(text):
    """
    Removes non-word characters (alphanumerics) and converts spaces to hyphens.
    Also strips leading and trailing whitespace.
    """
    text = unicodedata.normalize('NFKD', text).encode(
        'ascii', 'ignore').decode('ascii')
    text = re.sub('[^\w\s-]', '', text).strip()
    return re.sub('[-\s]+', '-', text)
