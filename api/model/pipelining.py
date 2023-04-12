import json
import logging
import re
import tensorflow as tf
from typing import List, Generator, Any
import model.request_classes as rq
from keras.layers import Bidirectional, GRU, Dropout, Dense, Concatenate
from transformers import TFAutoModel, AutoTokenizer

from config import MODELS, FEATURE_REGEX


class ExtractorModel:
    """Extracts features from text.
    Give a features_list with the features you want to extract from the following list:
    - phone_number
    - url
    - punctuation
    - horizontal_separator
    - hashtag
    - pipe
    - email
    - capitalized
    - full_caps
    """

    def __init__(self, features_list: List[str]) -> None:
        self.regex_list = [FEATURE_REGEX[feature] for feature in features_list]

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        feats = []
        for sequence in inputs.numpy():
            feats.append([])
            for line in sequence:
                feats[-1].append(
                    [
                        self.extract_feature(line.decode("utf-8"), regex)
                        for regex in self.regex_list
                    ]
                )
        return tf.convert_to_tensor(feats, dtype=tf.float32)

    @staticmethod
    def extract_feature(text: str, regex: str) -> int:
        return len(re.findall(regex, text))


class EncoderModel:
    """Encodes text into embeddings. Uses transformers models.
    specify model_name_or_path with the name of the model you want to use from hugging face.
    """

    def __init__(self, model_name_or_path: str) -> None:
        # loads transformers model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = TFAutoModel.from_pretrained(model_name_or_path)

    def __call__(self, inputs: tf.Tensor, normalize: bool = True) -> tf.Tensor:
        tokenized = [
            self.tokenizer(
                [line.decode("utf-8") for line in lines],
                padding=True,
                truncation=True,
                return_tensors="tf",
            )
            for lines in inputs.numpy()
        ]
        # runs model on inputs
        embeddings = [
            self.mean_pooling(self.model(tokens), tokens["attention_mask"])
            for tokens in tokenized
        ]

        # normalizes the embeddings if wanted
        if normalize:
            embeddings = [self.normalize(embedding) for embedding in embeddings]

        return tf.convert_to_tensor(embeddings, tf.float32)

    def mean_pooling(
        self, model_output: tf.Tensor, attention_mask: tf.Tensor
    ) -> tf.Tensor:
        """Pool the model output to get one fixed sized sentence vector"""
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = tf.cast(
            tf.broadcast_to(
                tf.expand_dims(attention_mask, -1), tf.shape(token_embeddings)
            ),
            tf.float32,
        )
        return tf.math.reduce_sum(
            token_embeddings * input_mask_expanded, axis=1
        ) / tf.clip_by_value(
            tf.math.reduce_sum(input_mask_expanded, axis=1), 1e-9, tf.float32.max
        )

    def normalize(self, embeddings: tf.Tensor) -> tf.Tensor:
        """Normalizes the embeddings. Uses L2 norm."""
        embeddings, _ = tf.linalg.normalize(embeddings, 2, axis=1)
        return embeddings


class FeatureCreator:
    """Creates features from text. combines the encoder and the extractor.
    Give the name of the encoder model you want to use and the list of features you want to extract.
    Refer to the documentation of EncoderModel and ExtractorModel for more information."""

    def __init__(self, encoder_name: str, features_list: List[str]) -> None:
        self.encoder = EncoderModel(encoder_name)
        self.extractor = ExtractorModel(features_list)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        encoded = self.encoder(inputs)
        extracted = self.extractor(inputs)
        return tf.concat([encoded, extracted], axis=-1)


class ClassifierModel:
    """Classifies sequences of embeddings. Uses a RNN model.
    Specify classifier_name with the name of the model you want to use from the weights folder."""

    def __init__(
        self, classifier_name: str, n_encoded: int, n_extracted: int, seq_len: int = 64
    ) -> None:

        n_features = n_encoded + n_extracted

        i = tf.keras.Input(shape=(seq_len, n_features))

        i = tf.keras.layers.Reshape(
            (seq_len, n_features, 1), input_shape=(seq_len, n_features)
        )(i)

        first_half = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, n_extracted)))(i)
        first_half = tf.keras.layers.Reshape(
            (seq_len, n_encoded), input_shape=(seq_len, n_encoded, 1)
        )(first_half)

        second_half = tf.keras.layers.Cropping2D(cropping=((0, 0), (n_encoded, 0)))(i)
        second_half = tf.keras.layers.Reshape(
            (seq_len, n_extracted), input_shape=(seq_len, n_extracted, 1)
        )(second_half)

        second_half = Dense(128, activation="relu")(second_half)

        x = Concatenate(axis=-1)([first_half, second_half])

        x = Bidirectional(GRU(256, return_sequences=True))(x)
        x = Dropout(0.25)(x)
        x = Bidirectional(GRU(128, return_sequences=True))(x)
        x = Dropout(0.25)(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.2)(x)
        section_x = Dense(7, activation="softmax")(x)
        fragment_x = Dense(1, activation="sigmoid")(x)
        x = Concatenate(axis=-1)([section_x, fragment_x])

        self.classifier = tf.keras.Model(inputs=i, outputs=x)
        self.classifier.load_weights(f"./weights/{classifier_name}.h5")

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.classifier(inputs)


class PipelineModel:
    """Combines the encoder, the extractor and the classifier.
    Specify model_name with the name of the model you want to use from the MODELS dictionary (or models.json file)."""

    def __init__(self, model_name: str) -> None:
        logging.info("Creating model pipeline...")
        self.parameters = MODELS[model_name]
        self.encoder = FeatureCreator(
            self.parameters["encoder"], self.parameters["features_list"]
        )
        self.classifier = ClassifierModel(
            self.parameters["classifier"],
            self.parameters["encoder_dim"],
            len(self.parameters["features_list"]),
        )
        logging.info("Model pipeline created.")

    def from_json(self, payload: rq.PipelineName):
        self.__init__(dict(payload)["name"])

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        with tf.device("GPU"):
            result = self.classifier(self.encoder(inputs))
        return result

    def to_dict(self) -> dict:
        return self.parameters

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
