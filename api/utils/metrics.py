"""This module defines usefull metrics and losses for model training
"""
import tensorflow as tf
from keras import backend as K

### Losses ###################################################################


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes including padding

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # trim y to number of categories in weights
        n_classes = len(weights)
        y_pred = y_pred[:, :, :n_classes]
        y_true = y_true[:, :, :n_classes]
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = K.cast(y_true, tf.float32) * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss


def multifactor_loss(class_weights=[1, 1, 1, 1, 1, 1, 1], lambd=1):
    """
    A loss function that combines the weighted categorical crossentropy and the binary crossentropy for the model

    Variables:
        class_weights: numpy array of shape (C,) where C is the number of classes including padding

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    categorical_loss = weighted_categorical_crossentropy(class_weights)
    fragment_loss = tf.keras.losses.BinaryCrossentropy()

    def loss(y_true, y_pred):
        return categorical_loss(y_true, y_pred) + lambd * fragment_loss(y_true, y_pred)

    return loss


### Metrics ###################################################################

## Single line metrics


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
        _classes += tf.cond(
            tf.logical_or(
                tf.cast(possible_positives, tf.bool),
                tf.cast(predicted_positives, tf.bool),
            ),
            lambda: 1,
            lambda: 0,
        )
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
        _classes += tf.cond(
            tf.logical_or(
                tf.cast(possible_positives, tf.bool),
                tf.cast(predicted_positives, tf.bool),
            ),
            lambda: 1,
            lambda: 0,
        )
    # return average balanced metric for each class
    return precision_by_class / K.cast(_classes, tf.float32)


def balanced_f1_score(y_true, y_pred):
    """This function calculates the F1 score metric"""
    precision = balanced_precision(y_true, y_pred)
    recall = balanced_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


## Sequence metrics
# Categorization


def seq_balanced_recall(y_true, y_pred, class_weights=[0, 1, 1, 1, 1, 1, 1]):
    """This function calculates the balanced recall metric on a sequence of lines
    recall = TP / (TP + FN)
    """
    # iterate over each predicted class to get class-specific metric
    _classes = 0
    recall_by_class = 0
    for i in range(len(class_weights)):
        y_pred_class = y_pred[:, :, i]
        y_true_class = y_true[:, :, i]
        true_positives = K.sum(K.round(K.clip(y_true_class * y_pred_class, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true_class, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred_class, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon()) * class_weights[i]
        recall_by_class = recall_by_class + recall
        _classes += (
            tf.cond(
                tf.logical_or(
                    tf.cast(possible_positives, tf.bool),
                    tf.cast(predicted_positives, tf.bool),
                ),
                lambda: 1,
                lambda: 0,
            )
            * class_weights[i]
        )
    return recall_by_class / K.cast(_classes, tf.float32)


def seq_balanced_precision(y_true, y_pred, class_weights=[0, 1, 1, 1, 1, 1, 1]):
    """This function calculates the balanced precision metric on a sequence of lines
    precision = TP / (TP + FP)
    """
    precision_by_class = 0
    _classes = 0
    # iterate over each predicted class to get class-specific metric
    for i in range(len(class_weights)):
        y_pred_class = y_pred[:, :, i + 1]
        y_true_class = y_true[:, :, i + 1]
        possible_positives = K.sum(K.round(K.clip(y_true_class, 0, 1)))
        true_positives = K.sum(K.round(K.clip(y_true_class * y_pred_class, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred_class, 0, 1)))
        precision = (
            true_positives / (predicted_positives + K.epsilon()) * class_weights[i]
        )
        precision_by_class = precision_by_class + precision
        _classes += (
            tf.cond(
                tf.logical_or(
                    tf.cast(possible_positives, tf.bool),
                    tf.cast(predicted_positives, tf.bool),
                ),
                lambda: 1,
                lambda: 0,
            )
            * class_weights[i]
        )
    # return average balanced metric for each class
    return precision_by_class / K.cast(_classes, tf.float32)


def seq_balanced_f1_score(y_true, y_pred, class_weights=[0, 1, 1, 1, 1, 1, 1]):
    """This function calculates the F1 score metric on a sequence of lines"""
    precision = seq_balanced_precision(y_true, y_pred, class_weights)
    recall = seq_balanced_recall(y_true, y_pred, class_weights)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# Fragmentation


def seq_fragment_precision(y_true, y_pred):
    """This function calculates the precision metric for the fragmentation part on a sequence of lines"""
    y_pred_class = y_pred[:, :, -1]
    y_true_class = y_true[:, :, -1]
    true_positives = K.sum(K.round(K.clip(y_true_class * y_pred_class, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred_class, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def seq_fragment_recall(y_true, y_pred):
    """This function calculates the recall metric for the fragmentation part on a sequence of lines"""
    y_pred_class = y_pred[:, :, -1]
    y_true_class = y_true[:, :, -1]
    true_positives = K.sum(K.round(K.clip(y_true_class * y_pred_class, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true_class, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def seq_fragment_f1_score(y_true, y_pred):
    """This function calculates the f1 score metric for the fragmentation part on a sequence of lines"""
    precision = seq_fragment_precision(y_true, y_pred)
    recall = seq_fragment_recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


### Test #########################################################################################


def test_multifactor_loss():
    y_true = tf.constant(
        [
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 0, 0, 0],
            ]
        ],
        dtype=tf.float32,
    )

    y_pred = tf.constant(
        [
            [
                [0, 0.97, 0.3, 0, 0, 0, 0, 0.25],
                [0, 0.5, 0, 0.95, 0, 0, 0, 0.75],
                [0.25, 0.5, 0.25, 0, 0, 0, 0, 0.2],
            ]
        ],
        dtype=tf.float32,
    )

    return multifactor_loss()(y_true, y_pred)


def test_seq_balanced_f1_score():
    y_true = tf.constant(
        [
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 0, 0, 0],
            ]
        ],
        dtype=tf.float32,
    )

    y_pred = tf.constant(
        [
            [
                [0, 0.97, 0.3, 0, 0, 0, 0, 0.25],
                [0, 0.5, 0, 0.95, 0, 0, 0, 0.75],
                [0.25, 0.5, 0.25, 0, 0, 0, 0, 0.2],
            ]
        ],
        dtype=tf.float32,
    )

    return seq_balanced_f1_score(y_true, y_pred)


def test_seq_fragment_f1_score():
    y_true = tf.constant(
        [
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 0, 0, 0],
            ]
        ],
        dtype=tf.float32,
    )

    y_pred = tf.constant(
        [
            [
                [0, 0.97, 0.3, 0, 0, 0, 0, 0.25],
                [0, 0.5, 0, 0.95, 0, 0, 0, 0.75],
                [0.25, 0.5, 0.25, 0, 0, 0, 0, 0.2],
            ]
        ],
        dtype=tf.float32,
    )

    return seq_fragment_f1_score(y_true, y_pred)


if __name__ == "__main__":
    print(test_multifactor_loss().numpy())
    print(test_seq_balanced_f1_score().numpy())
    print(test_seq_fragment_f1_score().numpy())
