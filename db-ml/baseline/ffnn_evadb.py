# coding=utf-8
# Copyright 2018-2023 EvaDB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import OrderedDict

import pandas as pd
import numpy as np
import torch
from evadb.functions.decorators.decorators import forward, setup
from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.functions.abstract.pytorch_abstract_function import (
    PytorchAbstractClassifierFunction,
)
from evadb.utils.generic_utils import try_to_import_torch, try_to_import_torchvision


class FFNN_EVADB(AbstractFunction):
    def as_numpy(self, val) -> np.ndarray:
        """
        Given a tensor in GPU, detach and get the numpy output
        Arguments:
             val (Tensor): tensor to be converted
        Returns:
            np.ndarray: numpy array representation
        """
        return val.detach().cpu().numpy()
    
    @property
    def name(self) -> str:
        return "FFNN_EVADB"
    
    @setup(cacheable=True, function_type="classification", batchable=True)
    def setup(self):
        import torch.nn as nn

        class FFNNModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(FFNNModel, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x

        self.model = FFNNModel(597540, 1024, 14588)
        self.model.eval()

    @property
    def labels(self):
        return list([str(num) for num in range(14588)])

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["val"],
                column_types=[NdArrayType.FLOAT32],
                column_shapes=[(None, 597540)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["label"],
                column_types=[
                    NdArrayType.STR
                ],
                column_shapes=[(None,)],
            )
        ],
    )
    def forward(self, data) -> pd.DataFrame:
        outcome = []
        data = np.ravel(data.to_numpy())
        list_data = [arr for arr in data]
        list_data = np.array(list_data).astype(np.float32)
        predictions = self.model(torch.tensor(list_data))
        for prediction in predictions:
            label = self.as_numpy(prediction.data.argmax())
            outcome.append({"label": str(label)})

        return pd.DataFrame(outcome, columns=["label"])
