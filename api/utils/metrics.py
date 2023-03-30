import tensorflow as tf
from keras import backend as K

def balanced_recall(y_true, y_pred):
    """This function calculates the balanced recall metric
    recall = TP / (TP + FN)
    """
    recall_by_class = 0
    _classes = 0
    # iterate over each predicted class to get class-specific metric
    for i in range(7):
        y_pred_class = y_pred[:, i]
        y_true_class = y_true[:, i]
        true_positives = K.sum(K.round(K.clip(y_true_class * y_pred_class, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true_class, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred_class, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        recall_by_class = recall_by_class + recall
        _classes += tf.cond(tf.logical_or(tf.cast(possible_positives, tf.bool), tf.cast(predicted_positives, tf.bool)), lambda : 1, lambda: 0)
    return recall_by_class / K.cast(_classes, tf.float32)

def balanced_precision(y_true, y_pred):
    """This function calculates the balanced precision metric
    precision = TP / (TP + FP)
    """
    precision_by_class = 0
    _classes = 0
    # iterate over each predicted class to get class-specific metric
    for i in range(7):
        y_pred_class = y_pred[:, i]
        y_true_class = y_true[:, i]
        possible_positives = K.sum(K.round(K.clip(y_true_class, 0, 1)))
        true_positives = K.sum(K.round(K.clip(y_true_class * y_pred_class, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred_class, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        precision_by_class = precision_by_class + precision
        _classes += tf.cond(tf.logical_or(tf.cast(possible_positives, tf.bool), tf.cast(predicted_positives, tf.bool)), lambda : 1, lambda: 0)
    # return average balanced metric for each class
    return precision_by_class / K.cast(_classes, tf.float32)

def balanced_f1_score(y_true, y_pred):
    """This function calculates the F1 score metric"""
    precision = balanced_precision(y_true, y_pred)
    recall = balanced_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def seq_balanced_recall(y_true, y_pred):
    """This function calculates the balanced recall metric on a sequence of lines
    recall = TP / (TP + FN)
    """
    # iterate over each predicted class to get class-specific metric
    _classes = 0
    recall_by_class = 0
    for i in range(1, 7):
        y_pred_class = y_pred[:, :, i]
        y_true_class = y_true[:, :, i]
        true_positives = K.sum(K.round(K.clip(y_true_class * y_pred_class, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true_class, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred_class, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        recall_by_class = recall_by_class + recall
        _classes += tf.cond(tf.logical_or(tf.cast(possible_positives, tf.bool), tf.cast(predicted_positives, tf.bool)), lambda : 1, lambda: 0)
    return recall_by_class / K.cast(_classes, tf.float32)

def seq_balanced_precision(y_true, y_pred):
    """This function calculates the balanced precision metric
    precision = TP / (TP + FP)
    """
    precision_by_class = 0
    _classes = 0
    # iterate over each predicted class to get class-specific metric
    for i in range(1, 7):
        y_pred_class = y_pred[:, :, i]
        y_true_class = y_true[:, :, i]
        possible_positives = K.sum(K.round(K.clip(y_true_class, 0, 1)))
        true_positives = K.sum(K.round(K.clip(y_true_class * y_pred_class, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred_class, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        precision_by_class = precision_by_class + precision
        _classes += tf.cond(tf.logical_or(tf.cast(possible_positives, tf.bool), tf.cast(predicted_positives, tf.bool)), lambda : 1, lambda: 0)
    # return average balanced metric for each class
    return precision_by_class / K.cast(_classes, tf.float32)

def seq_balanced_f1_score(y_true, y_pred):
    """This function calculates the F1 score metric"""
    precision = seq_balanced_precision(y_true, y_pred)
    recall = seq_balanced_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def seq_fragment_precision(y_true, y_pred):
    y_pred_class = y_pred[:, :, -1]
    y_true_class = y_true[:, :, -1]
    true_positives = K.sum(K.round(K.clip(y_true_class * y_pred_class, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred_class, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def seq_fragment_recall(y_true, y_pred):
    y_pred_class = y_pred[:, :, -1]
    y_true_class = y_true[:, :, -1]
    true_positives = K.sum(K.round(K.clip(y_true_class * y_pred_class, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true_class, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def seq_fragment_f1_score(y_true, y_pred):
    precision = seq_fragment_precision(y_true, y_pred)
    recall = seq_fragment_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


if __name__ == "__main__":
    y_true = tf.constant([[[0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0]]], dtype=tf.float32)

    y_pred = tf.constant([[[0, 0.97, 0.3, 0, 0, 0, 0, 0.25],
                        [0, 0.5, 0, 0.95, 0, 0, 0, 0.75],
                        [0.50, 0.25, 0.25, 0, 0, 0, 0, 0.2]]], dtype=tf.float32)

    print(seq_balanced_f1_score(y_true, y_pred))
    print(seq_fragment_f1_score(y_true, y_pred))