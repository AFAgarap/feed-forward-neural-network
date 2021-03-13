# Feed-Forward Neural Network
# Copyright (C) 2017-2021 Abien Fred Agarap
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
# from pt_datasets import load_dataset, create_dataloader
"""Training module for WDBC classifier"""
from pt_datasets import load_dataset, create_dataloader

__author__ = "Abien Fred Agarap"
__version__ = "2.0.0"


train_data, test_data = load_dataset("wdbc")
train_loader = create_dataloader(train_data, batch_size=32, num_workers=4)
