import sys
import logging

sys.path.append("../")
import tensorflow as tf
import model.data as data
import model.pipelining as pipe
import utils.request_classes as rq
import services.segmenting_service as seg
import services.training_service as train


class Controller:
    """Controller class for the API. Used to control the pipeline and dataset objects.

    Attributes:
        pipeline: PipelineModel object used to segment emails
        dataset: EmailDataset object used to store the emails to segment
    """

    pipeline: pipe.PipelineModel
    dataset: data.EmailDataset
    line_dataset: data.EmailLineDataset

    def __init__(self):
        self.pipeline = pipe.PipelineModel("default")

    def set_pipeline_from_json(self, pipeline_name: rq.PipelineName):
        self.pipeline.from_json(pipeline_name)

    def get_pipeline_to_json(self):
        return self.pipeline.to_json()

    def set_dataset_from_json(self, thread_list: rq.ThreadList):
        self.dataset = data.EmailDataset.from_json(thread_list)

    def set_dataset_from_csv(self, csv_path: str):
        self.dataset = data.EmailDataset.from_csv(csv_path)

    def get_dataset_to_json(self):
        return self.dataset.to_json()

    def set_line_dataset_from_csv(self, csv_path: str):
        self.line_dataset = data.EmailLineDataset.from_csv(csv_path)

    def segment(self):
        """Used to segment all EmailThread objects in the dataset
        pipeline must be a valid PipelineModel object
        """
        seg.segment(self.dataset, self.pipeline)

    def train_classifier(self, epochs: int = 1):
        """Used to train the classifier on the dataset
        pipeline must be a valid PipelineModel object
        """
        train.train_classifier(self.dataset, self.pipeline, epochs)

    def train_encoder(self, epochs: int = 1):
        """Used to train the encoder on the dataset
        pipeline must be a valid PipelineModel object
        """
        train.train_encoder(self.line_dataset, self.pipeline.encoder.encoder, epochs)
