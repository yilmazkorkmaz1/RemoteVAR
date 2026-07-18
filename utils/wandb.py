from accelerate.tracking import GeneralTracker, on_main_process
from typing import Optional
import wandb


class CustomWandbTracker(GeneralTracker):
    name = "wandb"
    requires_logging_directory = False

    def __init__(self, run_name: str, project: str = "RemoteVAR", **init_kwargs):
        # IMPORTANT: __init__ is executed on every process
        self.run_name = run_name
        self.project = project
        self.init_kwargs = init_kwargs
        self.run = None  # only created on main process

        self._init_wandb()  # safe: decorated

    @on_main_process
    def _init_wandb(self):
        # Only the main process will actually create a run
        self.run = wandb.init(name=self.run_name, project=self.project, **self.init_kwargs)

    @property
    def tracker(self):
        # Accelerate expects `tracker` to be an attribute-like property
        return None if self.run is None else self.run

    @on_main_process
    def store_init_configuration(self, values: dict):
        if self.run is not None:
            self.run.config.update(values, allow_val_change=True)

    @on_main_process
    def log(self, values: dict, step: Optional[int] = None):
        if self.run is not None:
            self.run.log(values, step=step)

    @on_main_process
    def finish(self):
        if self.run is not None:
            self.run.finish()
            self.run = None