import os

from pathlib import Path
from ruamel.yaml import YAML

from uintahtools import CONFIG, CONFIGSTR


class Settings:

    def __init__(self):
        self.yaml = YAML()
        self.path = Path(CONFIG)
        self.initialize_path()
        self.settings = self.initialize_settings()

    def __getitem__(self, key):
        return self.settings[key]

    def configure(self, key, override=True):
        if not self.isSet(key) or override:
            instruction = self.input_instruction(key)
            input_msg = key.upper() + " is not set. " + instruction + "\n"
            suggestion = os.path.expanduser(
                input(input_msg))

            self.settings.insert(1, key, [
                float(time) for time in suggestion.split()])
            self.yaml.dump(self.settings, self.path)

    def input_instruction(self, key):
        if key.find("time") > 0 or key.find("momentum") > 0:
            return "Enter list of times separated by space"
        else:
            return "Enter " + key + " to set"

    def isSet(self, key):
        return key in self.settings

    def initialize_settings(self):
        return self.yaml.load(self.path)

    def initialize_path(self):
        if not self.path.exists():
            self.path.touch()
            self.yaml.dump(self.yaml.load(CONFIGSTR), self.path)
