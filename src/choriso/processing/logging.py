import builtins
import os

import wandb


# Redefine print function to a custom logger
def print(s):
    builtins.print(f"\t * {s}")


class Logger:
    def __init__(self, wandb_log=False, report_dir="data/report/", wandb_project=None):
        self.wandb = wandb_log
        self.report_dir = report_dir

        if wandb_log:
            if wandb_project:
                wandb.init(project=wandb_project)
            else:
                wandb.init(project="cjhif-processing")

        log_file = report_dir + "analysis_log.log"

        # Create report dir if not there
        if not os.path.isdir(self.report_dir):
            os.makedirs(self.report_dir)

        if os.path.isfile(log_file):
            os.remove(log_file)
        self.log_file = open(log_file, "a")

    def log_fig(self, fig=None, fig_name=None):
        if self.wandb:
            wandb.log({fig_name: wandb.Image(fig)})
        else:
            fig.savefig(self.report_dir + fig_name)

    def log(self, data: dict):
        """log generic data (numeric), text, etc to wandb if flag is True"""

        if type(data) == dict:
            if self.wandb:
                wandb.log(data)
            else:
                for k in data.keys():
                    self.log_file.write(f"{k}: {data[k]}\n")

        else:
            self.log_file.write(data)
