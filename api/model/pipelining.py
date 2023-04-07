import json
import re
import tensorflow as tf
from keras.layers import Bidirectional, GRU, Dropout, Dense, Concatenate
from transformers import TFAutoModel, AutoTokenizer

with open("./models.json", "r") as f:
    MODELS = json.load(f)


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

    def __init__(self, features_list):
        self.features_dict = {
            "phone_number": self.extract_phone_number,
            "url": self.extract_url,
            "punctuation": self.extract_punctuation,
            "horizontal_separator": self.extract_horizontal_separators,
            "hashtag": self.extract_hashtags,
            "pipe": self.extract_pipes,
            "email": self.extract_email,
            "capitalized": self.extract_capitalized,
            "full_caps": self.extract_full_caps,
        }
        self.features_list = [self.features_dict[feature] for feature in features_list]

    def __call__(self, inputs):
        feats = []
        for sequence in inputs.numpy():
            feats.append([])
            for line in sequence:
                feats[-1].append(
                    [feature(line.decode("utf-8")) for feature in self.features_list]
                )
        return tf.convert_to_tensor(feats, dtype=tf.float32)

    @staticmethod
    def extract_phone_number(text):
        phone_regex = re.compile(
            "^\\+?\\d{1,4}?[-.\\s]?\\(?\\d{1,3}?\\)?[-.\\s]?\\d{1,4}[-.\\s]?\\d{1,4}[-.\\s]?\\d{1,9}$"
        )
        return len(re.findall(phone_regex, text))

    @staticmethod
    def extract_url(text):
        url_regex = re.compile(
            "^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$"
        )
        return len(re.findall(url_regex, text))

    @staticmethod
    def extract_punctuation(text):
        punct_regex = re.compile("[!?]")
        return len(re.findall(punct_regex, text))

    @staticmethod
    def extract_horizontal_separators(text):
        special_regex = re.compile("[-=~]")
        return len(re.findall(special_regex, text))

    @staticmethod
    def extract_hashtags(text):
        hashtag_regex = re.compile("[#]")
        return len(re.findall(hashtag_regex, text))

    @staticmethod
    def extract_pipes(text):
        pipes_regex = re.compile("[|]")
        return len(re.findall(pipes_regex, text))

    @staticmethod
    def extract_email(text):
        email_regex = re.compile("[-\w\.]+@([-\w]+\.)+[-\w]{2,4}")
        return len(re.findall(email_regex, text))

    @staticmethod
    def extract_capitalized(text):
        capitalized_regex = re.compile("[A-Z][a-z]*")
        return len(re.findall(capitalized_regex, text))

    @staticmethod
    def extract_full_caps(text):
        full_caps_regex = re.compile("[A-Z]+")
        return len(re.findall(full_caps_regex, text))


class EncoderModel:
    """Encodes text into embeddings. Uses transformers models.
    specify model_name_or_path with the name of the model you want to use from hugging face.
    """

    def __init__(self, model_name_or_path, **kwargs):
        # loads transformers model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = TFAutoModel.from_pretrained(model_name_or_path, **kwargs)

    def __call__(self, inputs, normalize=True):
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

    def mean_pooling(self, model_output, attention_mask):
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

    def normalize(self, embeddings):
        """Normalizes the embeddings. Uses L2 norm."""
        embeddings, _ = tf.linalg.normalize(embeddings, 2, axis=1)
        return embeddings


class FeatureCreator:
    """Creates features from text. combines the encoder and the extractor.
    Give the name of the encoder model you want to use and the list of features you want to extract.
    Refer to the documentation of EncoderModel and ExtractorModel for more information."""

    def __init__(self, encoder_name, features_list):
        self.encoder = EncoderModel(encoder_name)
        self.extractor = ExtractorModel(features_list)

    def __call__(self, inputs):
        encoded = self.encoder(inputs)
        extracted = self.extractor(inputs)
        return tf.concat([encoded, extracted], axis=-1)


class ClassifierModel:
    """Classifies sequences of embeddings. Uses a RNN model.
    Specify classifier_name with the name of the model you want to use from the weights folder."""

    def __init__(self, classifier_name, n_extracted):

        n_features = 384 + n_extracted

        i = tf.keras.Input(shape=(64, n_features))

        i = tf.keras.layers.Reshape((64, n_features, 1), input_shape=(64, n_features))(
            i
        )

        first_half = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, n_extracted)))(i)
        first_half = tf.keras.layers.Reshape((64, 384), input_shape=(64, 384, 1))(
            first_half
        )

        second_half = tf.keras.layers.Cropping2D(cropping=((0, 0), (384, 0)))(i)
        second_half = tf.keras.layers.Reshape(
            (64, n_extracted), input_shape=(64, n_extracted, 1)
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

    def __call__(self, inputs):
        return self.classifier(inputs)


class PipelineModel:
    """Combines the encoder, the extractor and the classifier.
    Specify model_name with the name of the model you want to use from the MODELS dictionary (or models.json file)."""

    def __init__(self, model_name):
        print("Creating model pipeline...")
        self.parameters = MODELS[model_name]
        self.encoder = FeatureCreator(
            self.parameters["encoder"], self.parameters["features_list"]
        )
        self.classifier = ClassifierModel(
            self.parameters["classifier"], len(self.parameters["features_list"])
        )
        print("Done!")

    def from_json(self, payload):
        self.__init__(dict(payload)["name"])

    def __call__(self, inputs):
        with tf.device("GPU"):
            result = self.classifier(self.encoder(inputs))
        return result

    def to_dict(self):
        return self.parameters

    def to_json(self):
        return json.dumps(self.to_dict())
