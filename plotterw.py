"""
This module contains class PlotterWindowed.
"""
# pylint: disable=bad-continuation

# import os

import multiprocessing as mp
import numpy as np

from mpl_toolkits import mplot3d
from matplotlib import pyplot


class ProcessPlotter:
    """
    This class implements a plotting process that receives updates over a pipe and
    updates a group of windows.
    """

    def __init__(self, x_dimensions, labels, **kwargs):
        self.figures = []
        self.pipe = None
        self.x_dimensions = x_dimensions

        num_dimensions = len(self.x_dimensions)

        self.line_emd = []
        self.text_emd = []
        self.axes_emd = []

        self.line_goal1 = []
        self.text_goal1 = []
        self.text_goal2 = []
        self.axes_goals = []

        self.axes_plot_preds = []
        self.scatter_preds = [None] * num_dimensions
        self.x_tests_preds = [None] * num_dimensions

        self.axes_plot_zlines = None
        self.scatter_zlines = None
        self.x_tests_zlines = None

        self.annotations = []
        self.labels = labels
        self.options = kwargs

    def terminate(self):
        """
        Called when the process receives a poison pill.
        """
        pyplot.close("all")

    def call_back(self):
        """
        This method will be called on a timed interval to check if there is new data in the pipe.
        """
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                data_type, dimension, data = command
                if data_type == "start":
                    epoch = data
                    title = f"epoch {epoch}"
                    # if len(self.figures) > 1:
                    #    title += f' / X dimension "{self.x_dimensions[dim]}"'
                    for figure in self.figures:
                        figure.suptitle(title)
                elif data_type == "end":
                    for figure in self.figures:
                        # figure.subplots_adjust(top=0.970, bottom=0.045, left=0.045, right=990)
                        figure.canvas.draw()
                elif data_type == "emd":
                    x_np, local_emds = data
                    if x_np is not None:
                        (line_emd,) = self.axes_emd[dimension].plot(
                            x_np, local_emds, "o--", label="local emd"
                        )
                        self.line_emd.append(line_emd)
                        self.axes_emd[dimension].legend(loc="upper right")
                        text_emd = self.axes_emd[dimension].text(
                            0.1,
                            0.95,
                            f"mean emd {np.mean(local_emds):.4f}",
                            transform=self.axes_emd[dimension].transAxes,
                        )
                        self.text_emd.append(text_emd)
                    else:
                        self.line_emd[dimension].set_ydata(local_emds)
                        self.text_emd[dimension].set_text(
                            f"mean emd {np.mean(local_emds):.4f}"
                        )
                        self.axes_emd[dimension].relim()
                        self.axes_emd[dimension].autoscale_view(True, True, True)

                elif data_type == "goal1":
                    (
                        x_np,
                        local_goal1_err,
                        global_goal1_err,
                        local_goal1_err_zsample,
                    ) = data
                    if x_np is not None:
                        (line_goal1,) = self.axes_goals[dimension].plot(
                            x_np, local_goal1_err, "o--", label="goal 1 - local error"
                        )
                        self.line_goal1.append(line_goal1)
                        self.axes_goals[dimension].legend(loc="upper right")
                        text_goal1 = self.axes_goals[dimension].text(
                            0.1,
                            0.95,
                            f"goal 1 - mean error {global_goal1_err:.4f}",
                            transform=self.axes_goals[dimension].transAxes,
                        )
                        self.text_goal1.append(text_goal1)
                        # text_goal2 = self.axes_goals[dimension].text(
                        #    0.1,
                        #    0.90,
                        #    "goal 2 - monotonically increasing {}".format(
                        #        "yes" if mon_incr else "no"
                        #    ),
                        #    {"color": "green" if mon_incr else "red"},
                        #    transform=self.axes_goals[dimension].transAxes,
                        # )
                        # self.text_goal2.append(text_goal2)
                    else:
                        self.line_goal1[dimension].set_ydata(local_goal1_err)
                        self.text_goal1[dimension].set_text(
                            f"goal 1 - mean error {global_goal1_err:.4f}"
                        )
                        # self.text_goal2[dimension].set_text(
                        #    "goal 2 - monotonically increasing {}".format(
                        #        "yes" if mon_incr else "no"
                        #    )
                        # )
                        # self.text_goal2[dimension].set_color(
                        #    "green" if mon_incr else "red"
                        # )
                        self.axes_goals[dimension].relim()
                        self.axes_goals[dimension].autoscale_view(True, True, True)

                elif data_type == "preds test":
                    continue
                    x_test, y_test = data
                    self.axes_plot_preds[dimension].scatter(
                        x_test,
                        y_test[:, 0],
                        y_test[:, 1],
                        marker="o",
                        s=self.options.get("test_s", 0.9),
                    )

                elif data_type == "z-lines test":
                    if len(self.x_dimensions) == 1:
                        x_test, y_test = data
                        self.axes_plot_zlines.scatter(
                            x_test,
                            y_test[:, 0],
                            y_test[:, 1],
                            marker="o",
                            s=self.options.get("test_s", 0.9),
                        )

                elif data_type == "z-lines":
                    if len(self.x_dimensions) == 1:
                        x_test, y_test = data
                        if x_test is not None:
                            self.scatter_zlines = self.axes_plot_zlines.scatter(
                                x_test,
                                y_test[:, 0],
                                y_test[:, 1],
                                marker="o",
                                s=self.options.get("zline_s", 0.1),
                            )

                            self.x_tests_zlines = x_test
                            legend = [r"test dataset ($y \sim Y_{x \in X}$)"]
                            legend.append(r"zlines ($f(x \in X, z \in z-samples)$)")
                            # Print the legend.
                            self.axes_plot_zlines.legend(
                                legend, loc="upper right",
                            )
                        else:
                            self.scatter_zlines._offsets3d = (
                                self.x_tests_zlines,
                                y_test[:, 0],
                                y_test[:, 1],
                            )
                            continue
                            xmin = X[:, 0].min()
                            xmax = X[:, 0].max()
                            ymin = X[:, 1].min()
                            ymax = X[:, 1].max()
                            self.axes_plot_zlines.set_xlim(
                                xmin - 0.1 * (xmax - xmin), xmax + 0.1 * (xmax - xmin)
                            )
                            self.axes_plot_zlines.set_ylim(
                                ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin)
                            )

                elif data_type == "preds":
                    # continue
                    x_test, y_test = data
                    if x_test is not None:
                        self.scatter_preds[dimension] = self.axes_plot_preds[
                            dimension
                        ].scatter(
                            x_test,
                            y_test[:, 0],
                            y_test[:, 1],
                            marker="x",
                            s=self.options.get("train_s", 0.5),
                        )

                        self.x_tests_preds[dimension] = x_test
                        legend = [r"test dataset ($y \sim Y_{x \in X}$)"]
                        legend.append(r"random preds ($f(x \in X, z \sim Z)$)")
                        # Print the legend.
                        self.axes_plot_preds[dimension].legend(
                            legend, loc="upper right",
                        )
                    else:
                        self.scatter_preds[dimension]._offsets3d = (
                            self.x_tests_preds[dimension],
                            y_test[:, 0],
                            y_test[:, 1],
                        )
                        # X = np.c_[self.x_tests_preds[dimension], y_test]
                        # X = np.c_[y_test[:, 0], y_test[:, 1]]
                        # self.scatter_preds[dimension].set_offsets(X)
                        xmin = self.x_tests_preds[dimension].min()
                        xmax = self.x_tests_preds[dimension].max()
                        # ymin = X[:, 1].min()
                        # ymax = X[:, 1].max()
                        self.axes_plot_preds[dimension].set_xlim(
                            xmin - 0.1 * (xmax - xmin), xmax + 0.1 * (xmax - xmin)
                        )
                        self.axes_plot_preds[dimension].set_ylim(
                            ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin)
                        )

                elif data_type == "z-lines pos":
                    if len(self.x_dimensions) == 1:
                        x_label_pos, y_label_pos = data

                        if not self.annotations:
                            for j, label in enumerate(self.labels):
                                annotation = self.axes_plot_zlines.annotate(
                                    label,
                                    (x_label_pos[dimension], y_label_pos[j]),
                                    fontsize=8,
                                )
                                self.annotations.append(annotation)
                        else:
                            for j, label in enumerate(self.labels):
                                self.annotations[j].set_position(
                                    (x_label_pos[dimension], y_label_pos[j])
                                )

                else:
                    assert False, data_type
        return True

    def __call__(self, pipe):
        print("starting plotter...")

        total_dimensions = len(self.x_dimensions)
        figure_dimensions = min(total_dimensions, 4)

        for _ in range(((total_dimensions - 1) // 4) + 1):
            figure = pyplot.figure()
            self.figures.append(figure)

        goal_rows = figure_dimensions * 2
        plot_rows = figure_dimensions if total_dimensions > 1 else 2
        columns = 2

        for dimension in range(total_dimensions):
            figure_idx = dimension // 4
            figure_dimension = dimension % figure_dimensions
            ############################################################################
            position = 1 if total_dimensions == 1 else figure_dimension * 4 + 1
            axes_goals = self.figures[figure_idx].add_subplot(
                goal_rows, columns, position
            )
            # axes_goals.set_ylim(0.0, 0.2)
            axes_goals.set_title("Training goals")
            axes_goals.set_xlabel(f"$X_{dimension}$")
            axes_goals.grid()
            self.axes_goals.append(axes_goals)
            ############################################################################

            ############################################################################
            position = 2 if total_dimensions == 1 else figure_dimension * 4 + 3
            axes_emd = self.figures[figure_idx].add_subplot(
                goal_rows, columns, position
            )
            axes_emd.set_title("Earth Mover's Distance (EMD)")
            axes_emd.set_xlabel(f"$X_{dimension}$")
            axes_emd.grid()
            self.axes_emd.append(axes_emd)
            ############################################################################

            ############################################################################
            position = 3 if total_dimensions == 1 else figure_dimension * 2 + 2
            axes_plot_preds = self.figures[figure_idx].add_subplot(
                plot_rows, columns, position, projection="3d"
            )
            axes_plot_preds.set_title("test dataset & random preds")
            axes_plot_preds.set_xlabel(f"$X_{dimension}$")
            axes_plot_preds.grid()
            self.axes_plot_preds.append(axes_plot_preds)
            ############################################################################

            ############################################################################
            if total_dimensions == 1:
                self.axes_plot_zlines = self.figures[0].add_subplot(
                    plot_rows,
                    columns,
                    4,
                    projection="3d"
                    # 1,
                    # 1,
                    # 1,
                    # projection="3d",
                )
                self.axes_plot_zlines.set_title("test dataset & z-lines")
                self.axes_plot_zlines.set_xlabel(f"$X_{dimension}$")
                self.axes_plot_zlines.grid()
            ############################################################################

        self.pipe = pipe
        # figure.tight_layout()
        for figure in self.figures:
            figure.subplots_adjust(
                top=0.970,
                bottom=0.045,
                left=0.045,
                right=0.990,
                hspace=0.400,
                wspace=0.100,
            )

        timer = self.figures[0].canvas.new_timer(interval=1000)
        timer.add_callback(self.call_back)
        timer.start()

        print("...done")
        pyplot.show()


class Plotter:
    """
    This class creates plots to track the model progress.
    """

    def __init__(self, datasets, z_samples, **kwargs):
        self.x_dimension_names = datasets.x_dimension_names
        self.y_dimension_name = datasets.y_dimension_name
        self.x_test = datasets.x_test
        self.y_test = datasets.y_test
        self.options = kwargs
        """
        self.z_samples_size = z_samples_size
        self.figures = []
        """

        self.first = True
        self.plot_pipe, plotter_pipe = mp.Pipe()
        # labels = ["$z_{{{}}}$".format(i) for i in range(z_samples_size)]
        self.plotter = ProcessPlotter(
            datasets.x_dimension_names, z_samples.labels, **kwargs
        )
        self.plot_process = mp.Process(
            target=self.plotter, args=(plotter_pipe,), daemon=True
        )
        self.plot_process.start()

    def start_frame(self, epoch):
        """
        Initializes the plot of the current epoch.
        """
        self.plot_pipe.send(("start", None, (epoch),))

    def plot_goal1(
        self,
        x_np,
        local_goal1_err,
        global_goal1_err,
        dimension,
        local_goal1_err_zsample,
    ):
        """
        This method sends the goal 1 and goal 2 information to the plotter process.
        """
        if not self.first:
            x_np = None
        self.plot_pipe.send(
            (
                "goal1",
                dimension,
                (
                    x_np,
                    local_goal1_err,
                    global_goal1_err.data.cpu().numpy(),
                    local_goal1_err_zsample,
                ),
            )
        )

    def plot_emd(self, x_np, local_emds, dimension):
        """
        This method sends the emd information to the plotter process.
        """
        return
        if not self.first:
            x_np = None
        self.plot_pipe.send(("emd", dimension, (x_np, local_emds)))

    def end_frame(self, epoch):
        """
        This method is called when the frame is finished being drawn.
        """

        # Create a png with the plot and save it to a file.
        # if not os.path.exists("plots"):
        #    os.makedirs("plots")
        # self.figure.savefig(f"plots/img_{epoch:03}_{i}.png", bbox_inches="tight")

        self.first = False
        self.plot_pipe.send(("end", None, None))

    def plot_datasets_zlines(self, y_predict_mat, orderings):
        # Filter the z-sample lines so that they are not as dense.
        zline_skip = self.options.get("zline_skip", 1)

        x_skipped = self.x_test[::zline_skip]
        y_predict_mat_skipped = y_predict_mat[:, ::zline_skip]

        x_tiled = np.tile(
            x_skipped, (y_predict_mat_skipped.shape[0], x_skipped.shape[1])
        )
        # Reshape y_predict_mat_skipped to be flat.
        # y_predict_mat_flat = y_predict_mat_skipped.flatten()
        shape = y_predict_mat_skipped.shape
        y_predict_mat_flat = y_predict_mat_skipped.reshape(
            (shape[0] * shape[1], shape[2])
        )

        # Add the scatter plots.
        for dimension in range(len(self.x_dimension_names)):

            x_np = None
            if self.first:
                self.plot_pipe.send(
                    (
                        "z-lines test",
                        dimension,
                        (self.x_test[:, dimension], self.y_test),
                    )
                )
                x_np = x_tiled[:, dimension]

            self.plot_pipe.send(("z-lines", dimension, (x_np, y_predict_mat_flat)))
            return

            # Get the positions for the rightmost elements in the z-lines to be used
            # with the z-sample labels.
            y_label_pos = y_predict_mat[:, orderings[dimension][-1]]
            x_label_pos = self.x_test[orderings[dimension][-1]]
            self.plot_pipe.send(("z-lines pos", dimension, (x_label_pos, y_label_pos)))

    def plot_datasets_preds(self, y_pred_d):

        # Add the scatter plots.
        for dimension in range(len(self.x_dimension_names)):
            x_np = None
            # x_np = self.x_test[:, dimension]
            if self.first:
                x_np = self.x_test[:, dimension]
                self.plot_pipe.send(
                    ("preds test", dimension, (self.x_test[:, dimension], self.y_test))
                )

            self.plot_pipe.send(("preds", dimension, (x_np, y_pred_d)))
