import ConfigParser
import os

from ai_challenge.utils import get_config_dir


class Config:
    """
    A container for any specific values.
    """

    def __init__(self, name):
        """
        Initializes the config.
        :param name: type str, the name of the config.
        """
        self.config = ConfigParser.ConfigParser()
        self.config.read(os.path.join(os.path.join(get_config_dir(), name)))
        self.name = name

    def get_as_type(self, type, section, key):
        """
        Gets specified type from config.
        """
        return type(self.config.get(section, key))

    def get_int(self, section, key):
        """
        Gets int from this config.
        """
        return self.get_as_type(int, section, key)

    def get_float(self, section, key):
        """
        Gets float from this config.
        """
        return self.get_as_type(float, section, key)

    def get_str(self, section, key):
        """
        Gets string from this config.
        """
        return self.get_as_type(str, section, key)

    def get_bool(self, section, key):
        """
        Gets bool from this config. 
        """
        return self.get_as_type(bool, section, key)

    def get_section(self, section):
        """
        Gets the whole section of a config.
        :param section: type str, the name of section
        """
        chosen_section = self.config._sections[section].copy()
        chosen_section.pop('__name__')
        return dict(
            [(name, self.str_to_num(value)) if self.is_number(value) else (name, value)
             for name, value in chosen_section.items()])

    @staticmethod
    def str_to_num(str_val):
        """
        Converts string to int or float.
        :param str_val: type str, string to convert
        :return: type int or type float, converted string
        """
        try:
            return int(str_val)
        except ValueError:
            return float(str_val)

    @staticmethod
    def is_number(input_):
        """
        Checks whether the provided input is numerical.
        :param input_: type str, the value to check
        :return: type bool, true if the provided string represents a number
        """
        return input_.replace('.', '').isdigit() or input_.replace('-', '').isdigit()

    def copy_config(self, rewrite_dir):
        """
        Rewrites the config to specified directory to save its values.
        :param rewrite_dir: type str, the directory to which the config will be rewritten.
        """
        if not os.path.isdir(rewrite_dir):
            os.makedirs(rewrite_dir)
        with open(os.path.join(os.path.join(get_config_dir(), self.name))) as dest_handle, open(
                os.path.join(rewrite_dir, self.name), "w") as trg_handle:
            trg_handle.write(dest_handle.read())
