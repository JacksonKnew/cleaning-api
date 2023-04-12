import logging
import tensorflow as tf
import model.data as data
import model.pipelining as pipe
from utils.data_manipulation import flatten_list, batch_list


def segment(dataset: data.EmailDataset, pipeline: pipe.PipelineModel) -> None:
    """Used to segment all EmailThread objects in the dataset
    pipeline must be a valid PipelineModel object
    """
    logging.info("Segmenting emails...")
    for batch, seq_order in zip(
        dataset.get_tf_dataset(),
        batch_list(dataset.seq_order, dataset.batch_size),
    ):
        pred = pipeline(batch)
        cat_pred = tf.argmax(pred[:, :, :7], axis=-1)
        frag_pred = pred[:, :, -1]
        concat_cat_pred = {seq: [] for seq in seq_order}
        concat_frag_pred = {seq: [] for seq in seq_order}
        for i, seq in enumerate(seq_order):
            concat_cat_pred[seq].append(cat_pred[i])
            concat_frag_pred[seq].append(frag_pred[i])
        for key in concat_cat_pred.keys():
            cat_pred, frag_pred = flatten_list(concat_cat_pred[key]), flatten_list(
                concat_frag_pred[key]
            )
            dataset.threads[key].segment(cat_pred, frag_pred)
    dataset.is_labeled = True
    logging.info("Segmentation complete")
