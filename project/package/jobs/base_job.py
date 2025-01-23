from abc import ABC, abstractmethod


class JobBase:
    def __init__(self, config, mandatory_fields, optional_fields):
        self.config = config
        self.mandatory_fields = mandatory_fields
        self.optional_fields = optional_fields
        intsection_flag = set(self.mandatory_fields).intersection(set(self.optional_fields)) != set()
        if intsection_flag:
            raise RuntimeError(f"Internal Bug: Overlap in Mandatory Fields and Optional Fields")
        for key in self.mandatory_fields:
            assert key in self.config, f"Mandatory field {key} is missing in config"
        
    @abstractmethod
    def __call__(self, *args, **kwds):
        pass