# Based on code from: https://github.com/iamgroot42/property_inference

import json
import logging
from pathlib import Path
from typing import List
from datetime import datetime
from simple_parsing.helpers import Serializable


class Logger:
    def __init__(self, exp_name: str, path: Path):
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(exp_name)
        self.fileHandler = logging.FileHandler(f"{path}.log")
        # Print to log file
        self.fileHandler.setFormatter(self.formatter)
        self.logger.addHandler(self.fileHandler)
        # And console simultaneously
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(self.formatter)
        self.logger.addHandler(consoleHandler)
        
    def log(self, msg: str, level: int = logging.INFO):
        self.logger.log(level, msg)


class Result:
    def __init__(self, path: Path, name: str) -> None:
        self.name = name
        self.path = path
        self.start = datetime.now()
        self.dic = {'name': name, 'start time': str(self.start)}

    def save(self):
        self.save_t = datetime.now()
        self.dic['save time'] = str(self.save_t)
        save_p = self.path.joinpath(f"{self.name}.json")
        self.path.mkdir(parents=True, exist_ok=True)
        with save_p.open('w') as f:
            json.dump(self.dic, f)

    def not_empty_dic(self, dic: dict, key):
        if key not in dic:
            dic[key] = {}

    def convert_to_dict(self, dic: dict):
        for k in dic:
            if isinstance(dic[k], Serializable):
                dic[k] = dic[k].__dict__
            if isinstance(dic[k], dict):
                self.convert_to_dict(dic[k])

    def load(self):
        raise NotImplementedError("Implement method to model for logger")

    def check_rec(self, dic: dict, keys: List):
        if not keys == []:
            k = keys.pop(0)
            self.not_empty_dic(dic, k)
            self.check_rec(dic[k], keys)
