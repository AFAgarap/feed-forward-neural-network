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
"""PyTorch implementation of a feed-forward neural network"""
import torch

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


class DNN(torch.nn.Module):
    def __init__(
        self,
        model_device: torch.device = torch.device("cpu"),
        units: list or tuple = [(784, 500), (500, 500), (500, 10)],
        learning_rate: float = 1e-4,
    ):
        """
        Constructs a feed-forward neural network classifier.

        Parameters
        ----------
        model_device: torch.device
            The device to use for model computations.
        units: list or tuple
            An iterable that consists of the number of units in each hidden layer.
        learning_rate: float
            The learning rate to use for optimization.
        """
        super().__init__()
        self.model_device = model_device
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_features=in_features, out_features=out_features)
                for in_features, out_features in units
            ]
        )
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.model_device)
        self.train_loss = []

    def forward(self, features):
        """
        Defines the forward pass by the model.

        Parameter
        ---------
        features : torch.Tensor
            The input features.

        Returns
        -------
        logits : torch.Tensor
            The model output.
        """
        activations = {}
        for index, layer in enumerate(self.layers):
            if index == 0:
                activations[index] = torch.relu(layer(features))
            elif index == 0:
                activations[index] = torch.relu(layer(activations[index - 1]))
        logits = activations[len(activations) - 1]
        return logits

    def fit(self, data_loader, epochs):
        """
        Trains the neural network model.

        Parameters
        ----------
        data_loader : torch.utils.dataloader.DataLoader
            The data loader object that consists of the data pipeline.
        epochs : int
            The number of epochs to train the model.
        """
        self.to(self.model_device)
        for epoch in range(epochs):
            epoch_loss = self.epoch_train(self, data_loader)
            if "cuda" in self.model_device.type:
                torch.cuda.empty_cache()
            self.train_loss.append(epoch_loss)
            print(f"epoch {epoch + 1}/{epochs} : mean loss = {self.train_loss[-1]:.6f}")

    def predict(self, features, return_likelihoods=False):
        """
        Returns model classifications

        Parameters
        ----------
        features: torch.Tensor
            The input features to classify.
        return_likelihoods: bool
            Whether to return classes with likelihoods or not.

        Returns
        -------
        predictions: torch.Tensor
            The class likelihood output by the model.
        classes: torch.Tensor
            The class prediction by the model.
        """
        outputs = self.forward(features)
        predictions, classes = torch.max(outputs.data, dim=1)
        return (predictions, classes) if return_likelihoods else classes

    def epoch_train(
        self, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader
    ) -> float:
        """
        Trains a model for one epoch.

        Parameters
        ----------
        model : torch.nn.Module
            The model to train.
        data_loader : torch.utils.dataloader.DataLoader
            The data loader object that consists of the data pipeline.

        Returns
        -------
        epoch_loss : float
            The epoch loss.
        """
        epoch_loss = 0
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.view(batch_features.shape[0], -1)
            batch_features = batch_features.to(model.model_device)
            batch_labels = batch_labels.to(model.model_device)
            model.optimizer.zero_grad()
            outputs = model(batch_features)
            train_loss = model.criterion(outputs, batch_labels)
            train_loss.backward()
            model.optimizer.step()
            epoch_loss += train_loss.item()
        return epoch_loss
