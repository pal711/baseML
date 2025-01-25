from abc import abstractmethod
from typing import Iterable

class JobBase:
    """This abstract class is responsible for maintaining the same format in 
    __main__.py file.
    The reqired format in __main__.py is to create a job object and call the object to perform respective tasks.
    """
    def __init__(self, config: dict, mandatory_fields: Iterable, optional_fields: Iterable)-> None:
        """Every job is created from config file. The initializer takes care of basic tasks.
        Getting mandatory_fields and optional_fields as norm for all the jobs.
        Validating all the mandatory fields are present and No common key between mandatory and optional keys

        Args:
            config (dict): The job config parameters
            mandatory_fields (Iterable): Names of the mandatory fields for the job
            optional_fields (Iterable): Names of the optional fields for the job

        Raises:
            RuntimeError: if there is overlap between mandatory fields and optional fields
            AssertionError: Mandatory key is missing in config
        """
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