"""
There are different jobs we can perform. All of those allowed types are listed below.
All the jobs should inherit package.jobs.base_job.JobBase
"""
from package.jobs.base_job import JobBase
from package.jobs.trainer import Trainer

SUPPORTED_JOBS = {
    "train": Trainer    # used to train a model
}

for jobname, jobcls in SUPPORTED_JOBS.items():
    # every supported job must inherit package.jobs.base_job.JobBase
    if not issubclass(jobcls, JobBase):
        error_msg = f"{jobcls.__name__} is not subclass of JobBase"
        raise AssertionError(error_msg)
                