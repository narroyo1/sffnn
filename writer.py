"""
This module contains class Writer.
"""


import numpy as np

from torch.utils.tensorboard import SummaryWriter


class Writer:
    """
    This class logs training information that can be viewed with tensorboard.
    """

    def __init__(self, datasets_target_function, trainer_params, datasets_params):

        # self.z_samples_size = z_samples_size

        # layout = {"pressure_charts": {}}
        # tags = ["z_{}".format(i) for i in range(self.z_samples_size)]
        # layout["pressure_charts"]["z-lines"] = ["Multiline", tags]
        # layout["pressure_charts"]["outer"] = ["Multiline", ["up_outer", "down_outer"]]

        self.summwriter = SummaryWriter()
        # self.summwriter.add_custom_scalars(layout)
        self.summwriter.add_text("target function:", datasets_target_function)
        self.summwriter.add_text("training params:", trainer_params)
        self.summwriter.add_text("datasets params:", datasets_params)
        # self.summwriter.add_graph(trainer.model)

    def log_emd(self, emd, epoch):
        """
        This method logs the emd (Earth Mover's Distance) scalar.
        """
        self.summwriter.add_scalar("emd", emd, epoch)

    def log_pressure(self, pressureratio, upouterpressure, dnouterpressure, epoch):
        """
        This method logs total pressure and individual pressure differences as scalars.
        """
        self.summwriter.add_scalar(
            "total pressure", np.sum(np.abs(pressureratio)), epoch
        )
        self.summwriter.add_scalar(
            "outer pressure", upouterpressure + dnouterpressure, epoch
        )
        self.summwriter.add_scalar("up_outer", upouterpressure, epoch)
        self.summwriter.add_scalar("down_outer", dnouterpressure, epoch)

        for idx, press in enumerate(pressureratio):
            self.summwriter.add_scalar("z_{}".format(idx + 1), press, epoch)

    def log_goal1_error(self, goal1_error, epoch):
        """
        This method logs total pressure and individual pressure differences as scalars.
        """
        self.summwriter.add_scalar(
            "goal1 error", goal1_error, epoch
        )

    def log_plot(self, figure, epoch):
        """
        This method logs the datasets figure.
        """
        self.summwriter.add_figure("plot", figure, epoch, close=False)

    def log_weights(self, model, epoch):
        """
        This function logs the weights and gradients of every layer in a model to a tensorboard
        summary writer.
        """
        for tag, value in model.named_parameters():
            tag = tag.replace(".", "/")
            self.summwriter.add_histogram(
                tag + "/weights", value.data.cpu().numpy(), global_step=epoch
            )
            self.summwriter.add_histogram(
                tag + "/grad", value.grad.data.cpu().numpy(), global_step=epoch
            )
