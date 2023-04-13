import logging
import tensorflow as tf
import model.data as data
import model.pipelining as pipe
from transformers import AutoTokenizer


def train_classifier(train_dataset: data.EmailDataset, pipeline: pipe.PipelineModel, epochs:int=1) -> None:
    """Used to train the classifier on the dataset
    pipeline must be a valid PipelineModel object
    """

    def _get_generator(tf_dataset: tf.data.Dataset, feature_creator: pipe.FeatureCreator):
        def gen():
            for text, label in tf_dataset:
                yield feature_creator(text) , label
        return gen

    logging.info("Training classifier...")
    tf_dataset = train_dataset.get_tf_dataset()
    feature_creator = pipeline.encoder
    classifier = pipeline.classifier

    feature_generator = _get_generator(tf_dataset, feature_creator)
    classifier.classifier.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    with tf.device("GPU"): # type: ignore
        classifier.classifier.fit(feature_generator(), epochs=epochs)
    logging.info("Training complete")


def train_encoder(train_dataset: data.EmailLineDataset, encoder: pipe.EncoderModel, epochs:int=1) -> None:
    """Used to train the encoder on the dataset
    pipeline must be a valid PipelineModel object
    """

    def _get_generator(tf_dataset: tf.data.Dataset, tokenizer: AutoTokenizer):
        def gen():
            for text, label in tf_dataset:
                lines = [str(line) for line in text.numpy()]
                tokens = tokenizer(lines, padding=True, truncation=True, return_tensors='tf') # type: ignore
                yield { "input_ids": tf.convert_to_tensor(tokens["input_ids"], dtype=tf.int32), "token_type_ids": tf.convert_to_tensor(tokens["token_type_ids"], dtype=tf.int32), "attention_mask": tf.convert_to_tensor(tokens["attention_mask"], dtype=tf.int32)}, label
        return gen
    
    def _get_dataset(tf_dataset: tf.data.Dataset, tokenizer: AutoTokenizer):
        return tf.data.Dataset.from_generator(
            _get_generator(tf_dataset, tokenizer), 
            output_types=({ "input_ids": tf.int32, "token_type_ids": tf.int32, "attention_mask": tf.int32 }, tf.int32)
        )

    logging.info("Training encoder...")
    tf_dataset = train_dataset.get_tf_dataset()
    tokenizer = encoder.tokenizer
    encoder_model = encoder.model

    feature_generator = _get_dataset(tf_dataset, tokenizer)

    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    token_type_ids = tf.keras.layers.Input(shape=(None, ), dtype=tf.int32, name="token_type_ids")
    attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
    
    x = encoder_model([input_ids, token_type_ids, attention_mask]) # type: ignore
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(7, activation='softmax')(x)
    clf = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=x)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    clf.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy")
    with tf.device("GPU"): # type: ignore
        clf.fit(feature_generator, epochs=epochs)
    logging.info("Training complete")



    
    