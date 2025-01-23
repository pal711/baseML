"""
There are different jobs we can perform. All of those allowed types are listed below.
"""
from package.jobs.trainer import Trainer

SUPPORTED_JOBS = {
    "train": Trainer    # used to train a model
}
                