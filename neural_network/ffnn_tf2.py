# Feed-Forward Neural Network
# Copyright (C) 2017-2020 Abien Fred Agarap
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Feed-Forward Neural Network"""
import tensorflow as tf

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


class DNN(tf.keras.Model):
    """
    A feed-forward neural network that optimizes
    softmax cross entropy using Adam optimizer.
    """

    def __init__(self, layers=[512, 512], initializer="glorot_uniform", **kwargs):
        """
        Constructs a feed-forward network classifier.

        Parameters
        ----------
        layers : list
            The list of network units.
        initializer : str
            The initializer to use for the hidden layers.
        """
        super().__init__()
        self.num_layers = len(layers)
        self.hidden_layers = []
        self.num_classes = kwargs.get("num_classes")
        self.activation = kwargs.get("activation")

        for index in range(self.num_layers):
            layer = tf.keras.layers.Dense(
                units=layers[index],
                activation=self.activation,
                kernel_initializer=initializer,
            )
            self.hidden_layers.append(layer)

        self.output_layer = tf.keras.layers.Dense(units=self.num_classes)
        self.criterion = tf.losses.CategoricalCrossentropy()
        self.optimizer = tf.optimizers.Adam(learning_rate=kwargs.get("learning_rate"))

    @tf.function
    def call(self, features):
        """
        Defines the forward pass by the model.

        Parameter
        ---------
        features : object
            The input features.
        Returns
        -------
        output : object
            The model output.
        """
        activations = []
        for index in range(self.num_layers):
            if index == 0:
                activations.append(self.hidden_layers[index](features))
            else:
                activations.append(self.hidden_layers[index](activations[index - 1]))
        output = self.output_layer(activations[-1])
        return output

    def fit(self, data_loader, epochs, show_every: int = 2):
        """
        Trains the feedforward network model.

        Parameters
        ----------
        data_loader : tf.data.Dataset
            The data loader object that consists of the data pipeline.
        epochs : int
            The number of epochs to train the model.
        show_every : int
            Print training epoch every `show_every` interval.
        """
        for epoch in range(epochs):
            epoch_loss = self.epoch_train(self, data_loader)
            self.train_loss.append(epoch_loss)
            if (epoch + 1) % show_every == 0:
                print(
                    f"epoch {epoch + 1}/{epochs} : mean loss = {self.train_loss[-1]:.6f}"
                )

    @staticmethod
    def epoch_train(model, data_loader):
        """
        Trains a model for one epoch.

        Parameters
        ----------
        model : tf.keras.Model
            The model to train.
        data_loader : tf.data.Dataset
            The data loader object that consists of the data pipeline.

        Returns
        -------
        epoch_loss : float
            The training epoch loss.
        """
        epoch_loss = []
        for batch_features, batch_labels in data_loader:
            with tf.GradientTape() as tape:
                outputs = model(batch_features)
                train_loss = model.criterion(batch_labels, outputs)
                epoch_loss.append(train_loss)
            gradients = tape.gradient(train_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        epoch_loss = tf.reduce_mean(epoch_loss)
        return epoch_loss
