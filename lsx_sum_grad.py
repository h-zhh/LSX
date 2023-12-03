import argparse
import os
import pickle
from time import time

import numpy as np
import tensorflow as tf
from sklearn.utils import check_random_state
from tensorflow import keras
from tensorflow.keras import Model, Sequential, layers
import wandb
import matplotlib.pyplot as plt


def load(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)


def dump(path, what):
    with open(path, "wb") as fp:
        pickle.dump(what, fp)


class Dataset:
    def __init__(self, tr_data, ts_data, n_classes, **kwargs):
        subsample = kwargs.pop("subsample", None)
        rng = check_random_state(kwargs.pop("rng", None))

        self.X_tr, self.Z_tr, self.Y_tr = tr_data
        self.X_ts, self.Z_ts, self.Y_ts = ts_data
        self.n_classes = n_classes

        if subsample is not None:
            n_train = len(self.X_tr)
            if subsample > 1:
                n_sampled = subsample
            elif subsample > 0:
                n_sampled = int(n_train * subsample)
            else:
                raise ValueError("subsample must be > 0")

            sampled = rng.permutation(n_train)[:n_sampled]
            self.X_tr = self.X_tr[sampled]
            self.Y_tr = self.Y_tr[sampled]
            self.Z_tr = self.Z_tr[sampled]

        self.flat_Y_tr = np.argmax(self.Y_tr, axis=-1)
        self.flat_Y_ts = np.argmax(self.Y_ts, axis=-1)

        self.x_shape = self.X_tr[0].shape
        self.z_shape = self.Z_tr[0].shape

        self.n_train = len(self.X_tr)
        self.n_test = len(self.X_ts)
        self.n_inputs = int(np.prod(self.x_shape))


class DecoyMNIST(Dataset):
    def __init__(self, **kwargs):
        with np.load(os.path.join("data", "decoy-mnist.npz")) as data:
            X_tr = data["arr_1"]  # (60k, 784) [0-255]
            Y_tr = data["arr_2"]  # (60k,) [0-10]
            Z_tr = data["arr_3"]  # (60k, 784) bool
            X_ts = data["arr_5"]  # (10k, 784) [0-255]
            Y_ts = data["arr_6"]  # (10k,) [0-10]
            Z_ts = data["arr_7"]  # (10k, 784) bool

        X_tr = (X_tr.reshape(-1, 28, 28, 1) / 255 - 0.5).astype(np.float32)
        X_ts = (X_ts.reshape(-1, 28, 28, 1) / 255 - 0.5).astype(np.float32)
        Y_tr = keras.utils.to_categorical(Y_tr)
        Y_ts = keras.utils.to_categorical(Y_ts)
        Z_tr = (~Z_tr).astype(np.float32)
        Z_ts = (~Z_ts).astype(np.float32)

        super().__init__(
            tr_data=(X_tr, Z_tr, Y_tr),
            ts_data=(X_ts, Z_ts, Y_ts),
            n_classes=10,
            **kwargs,
        )


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


class InputGradientModel(keras.Model):
    def __init__(
        self,
        submodel,
    ):
        super(InputGradientModel, self).__init__()
        self.submodel = submodel

    def call(self, x):
        """Performs inference.

        It calls the submodel and returns both the logits of the submodel
        and the input gradient w.r.t. the output of the softmax.
        """
        batch_size = tf.shape(x)[0]

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            s = self.submodel(x)  # scores in [-inf, +inf]
            p = tf.nn.softmax(s, axis=-1)  # probabilities in [0, 1]
            logp = tf.math.log(p)  # log-probabilities in [-inf, 0]
            sumlogp = tf.reduce_sum(logp, axis=-1)
        g = tape.gradient(sumlogp, x)  # gradient of sumlogp wrt x

        s = tf.reshape(s, (batch_size, -1))
        g = tf.reshape(g, (batch_size, -1))
        return s, g


def log_to_wandb(sum_grads):
    wandb.log({"sum_non_target_grads": sum_grads})
    return sum_grads


class LSXPipeline(keras.Model):
    """Learning-from-Self-Explaining Pipeline.

    It glues together the predictor and the critic.
    """

    def __init__(self, predictor, critic):
        super(LSXPipeline, self).__init__()
        self.predictor = predictor
        self.critic = critic

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit`.
        (x, z), y = data

        with tf.GradientTape() as tape:
            ps, cs, pg, z_true = self((x, z), training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile`)
            loss = self.compiled_loss(y, [ps, cs], regularization_losses=self.losses)

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
        self.compiled_metrics.update_state(y, [ps, cs])

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
        ps, pg = self.predictor(x)  # logits and input scores
        pg = tf.reshape(pg, tf.shape(x))
        pg = tf.clip_by_value(tf.abs(pg), 0, 1)
        cs = self.critic(tf.math.multiply(x, pg))  # logits
        return ps, cs, pg, z  #  logits, logits, input gradients, ground truth labels

    def compute_non_target_gradients(self, inputs):
        ps, cs, pg, z_true = self(inputs)

        # Reshape z_true to match the shape of pg
        z_true = tf.reshape(z_true, tf.shape(pg))

        # Convert Z to the correct dtype if necessary
        z_true = tf.cast(z_true, dtype=tf.float32)

        # Only select gradients where Z is False (i.e., non-target areas)
        non_target_grads = tf.abs(pg) * (1 - z_true)

        # Sum of absolute gradients in non-target areas
        sum_non_target_grads = tf.reduce_sum(non_target_grads)

        return sum_non_target_grads


DATASETS = {
    "decoy-mnist": DecoyMNIST,
}


MODELS = {
    "cnn": _make_cnn,
}


def _get_basename(args):
    fields = [
        (None, args.dataset),
        (None, args.predictor),
        (None, args.critic),
        (None, args.seed),
        ("S", args.subsample),
        ("E", args.max_epochs),
        ("B", args.batch_size),
        ("A", args.alpha),
    ]

    basename = "__".join(
        [(name + "=" + str(value) if name else str(value)) for name, value in fields]
    )
    return basename


def main():
    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)
    parser.add_argument("dataset", choices=sorted(DATASETS.keys()), help="Dataset")
    parser.add_argument(
        "predictor",
        choices=sorted(MODELS.keys()),
        help="Model to be used for the predictor",
    )
    parser.add_argument(
        "critic", choices=sorted(MODELS.keys()), help="Model to be used for the critic"
    )
    parser.add_argument(
        "-S",
        "--subsample",
        type=float,
        default=None,
        help="Amount of training data used for training",
    )
    parser.add_argument(
        "-E",
        "--max-epochs",
        type=int,
        default=10,
        help="Number of epochs per iteration",
    )
    parser.add_argument(
        "-B", "--batch-size", type=int, default=32, help="Batch size used for training"
    )
    parser.add_argument(
        "-A",
        "--alpha",
        type=float,
        default=0.1,
        help="Weight of predictor loss vs critic loss, in [0, 1]",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    wandb.init(project="LSX", entity="zhihanhu99")  # change entity to your own username
    wandb.config.update(args)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    basename = _get_basename(args)

    dataset = DATASETS[args.dataset](subsample=args.subsample, rng=rng)
    ce_loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    # Build the predictor:  takes as input MNIST images
    predictor = _make_cnn(
        dataset.X_tr[0].shape,
        dataset.n_classes,
    )
    ig_predictor = InputGradientModel(predictor)

    # Build the critic:  takes as input saliency maps
    critic = _make_cnn(
        dataset.X_tr[0].shape,
        dataset.n_classes,
    )

    wandb_callback = wandb.keras.WandbCallback()

    # Build the LSX pipeline
    pipeline = LSXPipeline(ig_predictor, critic)
    pipeline.compile(
        optimizer="sgd",
        loss={
            "output_1": ce_loss,  # loss on predictor's logits
            "output_2": ce_loss,  # loss on critic's logits
        },
        loss_weights={
            "output_1": 1.0,  # weight of predictor's loss
            "output_2": args.alpha,  # weight of critic's loss
        },
        metrics={
            "output_1": ["accuracy"],
            "output_2": ["accuracy"],
        },
    )

    combined_inputs = (dataset.X_tr, dataset.Z_tr)
    combined_validation = (dataset.X_ts, dataset.Z_ts)

    pipeline.fit(
        combined_inputs,
        dataset.Y_tr,
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
        validation_data=(combined_validation, dataset.Y_ts),
        callbacks=[
            wandb_callback,
        ],
    )

    # pipeline.save(f"{basename}_model/", save_format="tf")

    wandb.finish()


if __name__ == "__main__":
    main()
