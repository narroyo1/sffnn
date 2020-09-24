"""
This module contains class Plotter.
"""
# pylint: disable=bad-continuation

import os

import numpy as np

from matplotlib import pyplot

from IPython.display import clear_output


class Plotter:
    """
    This class creates plots to track the model progress.
    """

    def __init__(self, datasets, z_samples_size, **kwargs):
        self.x_dimensions = datasets.x_dimensions
        self.x_test = datasets.x_test
        self.y_test = datasets.y_test
        self.z_samples_size = z_samples_size
        self.figures = []
        self.labels = ["$z_{{{}}}$".format(i) for i in range(self.z_samples_size)]

        self.options = kwargs

    def start_frame(self, epoch):
        """
        Initializes the plot for the current epoch.
        """

        # To refresh the plots using IPython/Jupyter it is necessary to clear all
        # the plots and re-create them.
        clear_output(wait=True)

        width = 20
        height = 16 if len(self.x_dimensions) == 1 else 8
        self.figures = [pyplot.figure(i) for i in range(len(self.x_dimensions))]
        for dim, figure in enumerate(self.figures):
            figure.set_size_inches(width, height)
            title = f"epoch {epoch}"
            if len(self.figures) > 1:
                title += f' / X dimension "{self.x_dimensions[dim]}"'
            figure.suptitle(title)

    def plot_goal1(
        self, x_np, local_goal1_err, global_goal1_err, dimension,
    ):
        """
        This method plots goal 1 test results.
        """

        axes_goals = self.figures[dimension].add_subplot(2, 2, 1)
        axes_goals.set_ylim(0.0, 0.2)
        # for i in range(self.z_samples_size):
        #    axes_goals.plot(x_np, local_goal1_err[i], "o-", label=f"$z_{{{i}}}$")
        axes_goals.plot(x_np, local_goal1_err, "o--", label="goal 1 - local error")
        # axes_goals.plot(
        #    x_np, local_goal1_max_err, "o--", label="goal 1 - local max error"
        # )
        axes_goals.legend(loc="upper right")
        axes_goals.set_title("Training goals")
        axes_goals.set_xlabel(f"$X_{dimension}$")
        axes_goals.text(
            0.1,
            0.95,
            f"goal 1 - mean error {global_goal1_err:.4f}",
            transform=axes_goals.transAxes,
        )
        axes_goals.grid()

    def display_goal2(self, mon_incr):
        """
        This method displays the goal 2 test result.
        """

        self.figures[0].text(
            0.7,
            1.0,
            "goal 2 - monotonically increasing {}".format("yes" if mon_incr else "no"),
            {"color": "green" if mon_incr else "red"},
        )

    def plot_emd(self, x_np, local_emds, dimension):
        """
        This method plots EMD test results.
        """

        if len(self.x_dimensions) == 1:
            axes_emd = self.figures[dimension].add_subplot(2, 2, 2)
        else:
            axes_emd = self.figures[dimension].add_subplot(2, 2, 3)

        axes_emd.plot(x_np, local_emds, "o--", label="local emd")
        axes_emd.legend(loc="upper right")
        axes_emd.set_title("Earth Mover's Distance (EMD)")
        axes_emd.set_xlabel(f"$X_{dimension}$")
        axes_emd.text(
            0.1,
            0.95,
            f"mean emd {np.mean(local_emds):.4f}",
            transform=axes_emd.transAxes,
        )
        axes_emd.grid()

    def end_frame(self, epoch):
        """
        Finalizes the plot for the current epoch.
        """

        # Create a png with the plot and save it to a file.
        if not os.path.exists("plots"):
            os.makedirs("plots")
        for i, figure in enumerate(self.figures):
            figure.savefig(f"plots/img_{epoch:03}_{i}.png", bbox_inches="tight")

        pyplot.show()

    def plot_datasets_zlines(self, y_predict_mat, orderings):
        """
        This method plots the test dataset along with the zlines.
        """

        if len(self.x_dimensions) != 1:
            return

        axe = self.figures[0].add_subplot(2, 2, 3)
        # Filter the z-sample lines so that they are not as dense.
        zline_skip = self.options.get("zline_skip", 1)

        x_skipped = self.x_test[::zline_skip]
        y_predict_mat_skipped = y_predict_mat[:, ::zline_skip]

        x_tiled = np.tile(
            x_skipped, (y_predict_mat_skipped.shape[0], x_skipped.shape[1])
        )
        # Reshape y_predict_mat_skipped to be flat.
        y_predict_mat_flat = y_predict_mat_skipped.flatten()

        # Add the scatter plots.
        for dimension in range(len(self.x_dimensions)):

            # Get the positions for the rightmost elements in the z-lines to be used
            # with the z-sample labels.
            y_label_pos = y_predict_mat[:, orderings[dimension][-1]]
            x_label_pos = self.x_test[orderings[dimension][-1]]

            axe.scatter(
                self.x_test[:, dimension],
                self.y_test,
                marker="o",
                s=self.options.get("test_s", 0.5),
            )

            axe.scatter(
                x_tiled[:, dimension],
                y_predict_mat_flat,
                marker="o",
                s=self.options.get("zline_s", 0.1),
            )

            for j, label in enumerate(self.labels):
                axe.annotate(
                    label, (x_label_pos[dimension], y_label_pos[j]), fontsize=8
                )

            legend = [r"test dataset ($y \sim Y_{x \in X}$)"]
            legend.append(r"zlines ($f(x \in X, z \in z-samples)$)")
            # Print the legend.
            axe.legend(
                legend, loc="upper right",
            )
            axe.set_title("test dataset & z-lines")
            axe.set_xlabel(f"$X_{dimension}$")
            axe.grid()

    def plot_datasets_preds(self, y_pred_d):
        """
        This method plots the test dataset along with random samples.
        """

        if len(self.x_dimensions) == 1:
            axes = [
                self.figures[i].add_subplot(2, 2, 4)
                for i in range(len(self.x_dimensions))
            ]
        else:
            axes = [
                self.figures[i].add_subplot(1, 2, 2)
                for i in range(len(self.x_dimensions))
            ]
        # Add the scatter plots.
        for dimension in range(len(self.x_dimensions)):
            axes[dimension].scatter(
                self.x_test[:, dimension],
                self.y_test,
                marker="o",
                s=self.options.get("test_s", 0.9),
            )

            axes[dimension].scatter(
                self.x_test[:, dimension],
                y_pred_d,
                marker="x",
                s=self.options.get("train_s", 0.5),
            )

            legend = [r"test dataset ($y \sim Y_{x \in X}$)"]
            legend.append(r"random preds ($f(x \in X, z \sim Z)$)")
            # Print the legend.
            axes[dimension].legend(
                legend, loc="upper right",
            )
            axes[dimension].set_title("test dataset & random preds")
            axes[dimension].set_xlabel(f"$X_{dimension}$")
            axes[dimension].grid()
