"""
Callback utilities for training visualization and monitoring.

This module provides custom Keras callbacks for real-time training
progress visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class TrainingPlot(tf.keras.callbacks.Callback):
    """Callback for real-time training progress visualization.

    Plots training and validation loss/accuracy at regular intervals
    during training.

    Args:
        path_save_file: Path to save the plot image.
        plot_interval: Interval (in epochs) between plot updates.
    """

    def __init__(self, path_save_file: str, plot_interval: int = 5) -> None:
        super().__init__()
        self.path_save_file = path_save_file
        self.plot_interval = plot_interval
        self.acc: list = []
        self.loss: list = []
        self.val_acc: list = []
        self.val_loss: list = []
        self.logs: list = []

    def on_train_begin(self, logs: dict | None = None) -> None:
        """Initialize tracking lists at training start."""
        self.acc = []
        self.loss = []
        self.val_acc = []
        self.val_loss = []
        self.logs = []

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Update and plot metrics at epoch end.

        Args:
            epoch: Current epoch number.
            logs: Dictionary containing training metrics.
        """
        if logs is None:
            logs = {}

        acc = logs.get("acc") or logs.get("accuracy")
        loss = logs.get("loss")
        val_acc = logs.get("val_acc") or logs.get("val_accuracy")
        val_loss = logs.get("val_loss")

        self.logs.append(logs)
        self.acc.append(acc)
        self.loss.append(loss)
        self.val_acc.append(val_acc)
        self.val_loss.append(val_loss)

        if epoch > 0 and epoch % self.plot_interval == 0:
            self._create_plot(epoch)

    def _create_plot(self, epoch: int) -> None:
        """Create and save the training progress plot.

        Args:
            epoch: Current epoch number for labeling.
        """
        x_epoch = np.arange(0, len(self.loss))

        plt.style.use("seaborn-v0_8-whitegrid")

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = "tab:red"
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color=color)
        ax1.plot(x_epoch, self.loss, color=color, label="Training Loss")
        ax1.plot(
            x_epoch,
            self.val_loss,
            color=color,
            linestyle="dashed",
            label="Validation Loss",
        )
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("Accuracy", color=color)
        ax2.plot(x_epoch, self.acc, color=color, label="Training Accuracy")
        ax2.plot(
            x_epoch,
            self.val_acc,
            color=color,
            linestyle="dashed",
            label="Validation Accuracy",
        )
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.legend(loc="upper right")

        plt.title(f"Training Progress - Epoch {epoch}")
        fig.tight_layout()

        plt.savefig(self.path_save_file)
        plt.close()
