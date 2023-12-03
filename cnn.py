from lsx_sum_grad import DecoyMNIST, log_to_wandb
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
import os
import argparse
import pickle
from time import time
import wandb


def _make_cnn(input_shape, n_classes):
    model = Sequential(
        [
            layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                input_shape=input_shape,
            ),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(16, activation="relu", name="second_to_top"),
            layers.Dropout(0.2),
            layers.Dense(n_classes, name="top"),
        ]
    )

    return model


class CNNGradientClipped(keras.Model):
    """CNN with gradient clipping."""

    def __init__(self, model):
        super(CNNGradientClipped, self).__init__()
        self.model = model

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit`.
        (x, z), y = data

        with tf.GradientTape() as tape:
            s, g, z_true = self((x, z), training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile`)
            loss = self.compiled_loss(y, s, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # clip the gradients
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)

        # Update weights
        self.optimizer.apply_gradients(zip(clipped_gradients, trainable_vars))

        # Compute sum of non-target gradients and log it
        sum_grads = self.compute_non_target_gradients((x, z))
        tf.py_function(log_to_wandb, [sum_grads], Tout=tf.float32)

        # Update the metrics.
        # Metrics are configured in `compile`.
        self.compiled_metrics.update_state(y, s)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs):
        """Performs inference.

        It returns the logits of the predictor, the logits of the critic,
        and the input gradient explanations of the predictor.  The critic
        is fed the input-times-gradient explanations of the predictor.
        """
        x, z = inputs
        batch_size = tf.shape(x)[0]

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            s = self.model(x)  # scores in [-inf, +inf]
            p = tf.nn.softmax(s, axis=-1)  # probabilities in [0, 1]
            logp = tf.math.log(p)  # log-probabilities in [-inf, 0]
            sumlogp = tf.reduce_sum(logp, axis=-1)
        g = tape.gradient(sumlogp, x)  # gradient of sumlogp wrt x

        s = tf.reshape(s, (batch_size, -1))
        g = tf.reshape(g, (batch_size, -1))
        g = tf.reshape(g, tf.shape(x))
        g = tf.clip_by_value(tf.abs(g), 0, 1)
        return s, g, z

    def compute_non_target_gradients(self, inputs):
        s, g, z_true = self(inputs)

        # Reshape z_true to match the shape of pg
        z_true = tf.reshape(z_true, tf.shape(g))

        # Convert Z to the correct dtype if necessary
        z_true = tf.cast(z_true, dtype=tf.float32)

        # Only select gradients where Z is False (i.e., non-target areas)
        non_target_grads = tf.abs(g) * (1 - z_true)

        # Sum of absolute gradients in non-target areas
        sum_non_target_grads = tf.reduce_sum(non_target_grads)

        return sum_non_target_grads


def main():
    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)
    parser.add_argument(
        "-E",
        "--max-epochs",
        type=int,
        default=10,
        help="Number of epochs per iteration",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    wandb.init(project="LSX", entity="zhihanhu99")
    wandb.config.update(args)
    wandb_callback = wandb.keras.WandbCallback()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    dataset = DecoyMNIST(rng=rng)

    ce_loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model = _make_cnn(dataset.X_tr[0].shape, dataset.n_classes)
    model_clipped = CNNGradientClipped(model)
    model_clipped.compile(optimizer="sgd", loss=ce_loss, metrics=["accuracy"])

    combined_inputs = (dataset.X_tr, dataset.Z_tr)
    combined_validation = (dataset.X_ts, dataset.Z_ts)

    # Train the model
    model_clipped.fit(
        combined_inputs,
        dataset.Y_tr,
        epochs=args.max_epochs,
        # batch_size=args.batch_size,
        verbose=args.verbose,
        validation_data=(combined_validation, dataset.Y_ts),
        callbacks=[
            wandb_callback,
        ],
    )

    wandb.finish()


if __name__ == "__main__":
    main()
