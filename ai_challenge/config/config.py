import ConfigParser
import os


class Config:
    """
    A container for any model and specific values.
    """

    def __init__(self, name):
        self.config = ConfigParser.ConfigParser()
        self.config.read(os.path.join(os.path.dirname(__file__), name))

    def get_as_type(self, type, section, key):
        return type(self.config.get(section, key))

    def get_int(self, section, key):
        return self.get_as_type(int, section, key)

    def get_float(self, section, key):
        return self.get_as_type(float, section, key)

    def get_str(self, section, key):
        return self.get_as_type(str, section, key)

    def get_bool(self, section, key):
        return self.get_as_type(bool, section, key)

    def get_num_section(self, section):
        chosen_section = self.config._sections[section].copy()
        chosen_section.pop('__name__')
        return {name: self.str_to_num(value) for name, value in chosen_section.items()}

    @staticmethod
    def str_to_num(str_val):
        try:
            return int(str_val)
        except ValueError:
            return float(str_val)

    def get_path(self, section, key):
        """
        This is done to get rid of different relative imports. Every path used in config should be
        relative to main repo directory and read with this method.
        """
        return os.path.join(os.path.dirname(__file__), self.get_as_type(str, section, key))
