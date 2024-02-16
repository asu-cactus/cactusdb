from collections import OrderedDict
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import utils
import tensorflow as tf
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
        # convert the string back to list of int
        list_hidden_layer_sizes = np.load("evadb_ffnn_reg.npy")
        self.model = FFNNPyTorch(list_hidden_layer_sizes)


        # self.model = FFNNModel(597540, 1, 10)
        self.model.eval()
        self.timer_process = utils.Timer()
        self.timer_model_inference = utils.Timer()
        self.t_process = 0
        self.t_model_inference = 0

    @property
    def labels(self):
        return list([str(num) for num in range(10)])

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
                columns=["label", "t_process", "t_model_inference"],
                column_types=[
                    NdArrayType.STR,
                    NdArrayType.FLOAT32,
                    NdArrayType.FLOAT32,
                ],
                column_shapes=[(None,), (None,), (None,)],
            )
        ],
    )
    def forward(self, data) -> pd.DataFrame:
        outcome = []
        self.timer_process.tic()
        data = np.ravel(data.to_numpy())
        list_data = [arr for arr in data]
        list_data = np.array(list_data).astype(np.float32)
        self.t_process += self.timer_process.toc()
        self.timer_model_inference.tic()
        predictions = self.model(torch.tensor(list_data))
        self.t_model_inference += self.timer_model_inference.toc()
        for prediction in predictions:
            label = self.as_numpy(prediction.data.argmax())
            outcome.append(
                {
                    "label": str(label),
                    "t_process": self.t_process,
                    "t_model_inference": self.t_model_inference,
                }
            )
        result_df = pd.DataFrame(
            outcome, columns=["label", "t_process", "t_model_inference"]
        )
        return result_df



class FFNNPyTorch(nn.Module):
    def __init__(self, list_hidden_layer_sizes):
        super(FFNNPyTorch, self).__init__()
        self.linears = []
        for i in range(len(list_hidden_layer_sizes) - 1):
            self.linears.append(
                nn.Linear(list_hidden_layer_sizes[i], list_hidden_layer_sizes[i + 1])
            )
        self.linears = nn.ModuleList(self.linears)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = self.linears[i](x)
            if i != (len(self.linears) - 1):
                x = self.relu(x)
            else:
                x = self.softmax(x)
        return x


def FFNNTensorFlow(list_hidden_layer_sizes):
    model = tf.keras.Sequential()

    # Adding input layer
    model.add(tf.keras.layers.InputLayer(input_shape=(list_hidden_layer_sizes[0],)))

    # Adding hidden layers
    for units in list_hidden_layer_sizes[1:-1]:
        model.add(tf.keras.layers.Dense(units, activation="relu"))

    # Adding output layer
    model.add(tf.keras.layers.Dense(list_hidden_layer_sizes[-1], activation="softmax"))

    return model
