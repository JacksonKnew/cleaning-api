import sys

sys.path.append("../")
import os
from tqdm import tqdm
import tensorflow as tf
import model.data as data
import model.pipelining as pipe


class Controller:
    pipeline: pipe.PipelineModel
    dataset: data.EmailDataset

    def __init__(self):
        self.pipeline = pipe.PipelineModel("default")

    def set_pipeline_from_json(self, pipeline_name: str):
        self.pipeline.from_json(pipeline_name)

    def get_pipeline_to_json(self):
        return self.pipeline.to_json()

    def set_dataset_from_json(self, thread_list: list):
        self.dataset = data.EmailDataset.from_json(thread_list)

    def get_dataset_to_json(self):
        return self.dataset.to_json()

    def segment(self):
        """Used to segment all EmailThread objects in the dataset
        pipeline must be a valid PipelineModel object
        """
        print("Segmenting emails...")
        for batch, seq_order in tqdm(
            zip(
                self.dataset.get_tf_dataset(),
                self.dataset._chunks(self.dataset.seq_order, self.dataset.batch_size),
            )
        ):
            pred = self.pipeline(batch)
            cat_pred = tf.argmax(pred[:, :, :7], axis=-1)
            frag_pred = pred[:, :, -1]
            concat_cat_pred = {seq: [] for seq in seq_order}
            concat_frag_pred = {seq: [] for seq in seq_order}
            for i, seq in enumerate(seq_order):
                concat_cat_pred[seq].append(cat_pred[i])
                concat_frag_pred[seq].append(frag_pred[i])
            for key in concat_cat_pred.keys():
                cat_pred, frag_pred = self.dataset._flatten_list(
                    concat_cat_pred[key]
                ), self.dataset._flatten_list(concat_frag_pred[key])
                self.dataset.threads[key].segment(cat_pred, frag_pred)
        self.dataset.is_labeled = True
