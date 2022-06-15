# Based on code from: https://github.com/iamgroot42/property_inference
# The 'Result' class and its children were written by Yifu Lu

import json
import os
import logging
from pathlib import Path
from typing import List
from datetime import datetime
from copy import deepcopy
from simple_parsing.helpers import Serializable

from bbeval.config import AttackerConfig
from bbeval.utils import get_log_save_path


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
            json.dump(self.dic, f, indent=4)

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


class AttackResult(Result):
    def __init__(self,
                 attack_config: AttackerConfig):
        # Infer path from data_config inside attack_config
        dataset_name = attack_config.dataset_config.name
        experiment_name = attack_config.experiment_name 
        save_path = get_log_save_path()
        path = Path(os.path.join(save_path, dataset_name, experiment_name))
        super().__init__(path, experiment_name)
        # Worthwhile to save the attack config
        attack_config_copy = deepcopy(attack_config)
        # Get rid of 'aux_model_configs' field, if present
        if attack_config_copy.aux_model_configs is not None:
            attack_config_copy.aux_model_configs = None
        self.dic["attack_config"] = attack_config_copy
        self.convert_to_dict(self.dic)
    
    def add_result(self, queries_used: int, result: dict):
        """
            Log misc. information
        """
        self.check_rec(self.dic, ['result'])
        self.dic['result'][queries_used] = result


class Logger:
    def __init__(self, attack_config: AttackerConfig):
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(attack_config.experiment_name)
        path = get_log_save_path()
        self.fileHandler = logging.FileHandler(f"{path}.log")
        # Print to log file
        self.fileHandler.setFormatter(self.formatter)
        self.logger.addHandler(self.fileHandler)
        # And console simultaneously
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(self.formatter)
        self.logger.addHandler(consoleHandler)
        # Also create a tracker for query-wise results
        self.result_logger = AttackResult(attack_config)
        
    def log(self, msg: str, level: int = logging.INFO):
        self.logger.log(level, msg)
    
    def add_result(self, queries_used: int, result: dict):
        self.result_logger.add_result(queries_used, result)
    
    def save(self):
        self.result_logger.save()
