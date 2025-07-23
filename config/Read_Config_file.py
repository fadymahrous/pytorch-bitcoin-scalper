from configparser import ConfigParser,SectionProxy
from os import path

class Read_Config_file:
    def __init__(self):
        config_file_location=path.join('config','config.ini')
        if not path.exists(config_file_location):
            raise Exception('Configuration file does not exist.')
        self.config=ConfigParser()
        self.config.read(config_file_location)

    def get_section(self,section:str)->SectionProxy:
        if section not in self.config:
            raise Exception(f"Section {section} not found in config file.")
        return self.config[section]
