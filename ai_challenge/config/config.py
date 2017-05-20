import ConfigParser
import os

from ai_challenge.utils import get_config_dir


class Config:
    """
    A container for any model and specific values.
    """

    def __init__(self, name):
        self.config = ConfigParser.ConfigParser()
        self.config.read(os.path.join(os.path.join(get_config_dir(), name)))
        self.name = name

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

    def get_section(self, section):
        chosen_section = self.config._sections[section].copy()
        chosen_section.pop('__name__')
        return dict(
            [(name, self.str_to_num(value)) if self.is_number(value) else (name, value)
             for name, value in chosen_section.items()])

    @staticmethod
    def str_to_num(str_val):
        try:
            return int(str_val)
        except ValueError:
            return float(str_val)

    @staticmethod
    def is_number(input_):
        return input_.replace('.', '').isdigit() or input_.replace('-', '').isdigit()

    def copy_config(self, dest_dir):
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
        with open(os.path.join(os.path.join(get_config_dir(), self.name))) as dest_handle, open(
                os.path.join(dest_dir, self.name), "w") as trg_handle:
            trg_handle.write(dest_handle.read())
