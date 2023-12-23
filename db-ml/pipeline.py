import utils
import pandas as pd
import numpy as np
import connectorx as cx
from tqdm.auto import tqdm
from abc import ABC, abstractmethod


class Pipeline(object):
    """A convenient class to measure the running time of a program"""

    def __init__(self, name, num_loop=10):
        self.name = name
        self.num_loop = num_loop
        pass

    @abstractmethod
    def data_loading_impl(self):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def data_processing_impl(self, data):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def model_inference_impl(self, data):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def run_customized_pipeline(self):
        raise NotImplementedError("Not implemented")

    def run_pipeline(self):
        timer_end_end = utils.Timer()
        timer_data_loading = utils.Timer()
        timer_data_processing = utils.Timer()
        timer_model_inference = utils.Timer()
        t_end_end = 0
        t_data_loading = 0
        t_data_processing = 0
        t_model_inference = 0

        data = None
        timer_end_end.tic()
        for _ in tqdm(range(self.num_loop)):
            timer_data_loading.tic()
            data = self.data_loading_impl()
            t_data_loading += timer_data_loading.toc()

            timer_data_processing.tic()
            data = self.data_processing_impl(data)
            t_data_processing += timer_data_processing.toc()

            timer_model_inference.tic()
            data = self.model_inference_impl(data)
            t_model_inference += timer_model_inference.toc()

        t_end_end += timer_end_end.toc() / self.num_loop
        t_data_loading /= self.num_loop
        t_data_processing /= self.num_loop
        t_model_inference /= self.num_loop

        result_df = pd.DataFrame(
            {
                "config_name": self.name,
                "t_data_load": t_data_loading,
                "t_data_process": t_data_processing,
                "t_model": t_model_inference,
                "t_end_end": t_end_end,
            }, index=[0]
        )
        return result_df


class TwoTowerModelPipeline(Pipeline):
    def __init__(self, num_loop=10):
        super(TwoTowerModelPipeline, self).__init__("two-tower-model", num_loop)
        # self.model = None  # TODO
        # self.model.eval()
        self.postgres_conn = utils.get_postgres_connection_config()

    def data_loading_impl(self):
        # TODO
        # data = cx.read_sql(self.postgres_conn, "")
        data = None
        return data

    def data_processing_impl(self, data):
        # TODO
        # data['COL1'] = label_encoder.transform(data['COL1'])
        return data

    def model_inference_impl(self, data):
        # TODO
        # result = self.model(data)
        return data
