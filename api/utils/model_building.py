import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
from keras.models import Sequential
from sentence_transformers import SentenceTransformer
from keras.layers import Bidirectional, GRU, Dropout, Dense, Concatenate

### Encoders ############################################

def build_MiniLM():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def build_fine_tuned_small_bert():
    preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/2")
    i = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    x = preprocessor(i)
    x = encoder(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x['pooled_output'])
    x = tf.keras.layers.Dense(7, activation='softmax', name="output")(x)
    model = tf.keras.Model(i, x)
    model.load_weights("../api/models/fine_tuned_small_bert.h5")
    return tf.keras.Model(model.layers[0].input, model.layers[2].output["pooled_output"])

def build_multi_bert():
    preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3")
    encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4")
    i = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    x = preprocessor(i)
    x = encoder(x)["pooled_output"]
    model = tf.keras.Model(i, x)
    return model

### Classifiers #########################################

def build_small_bert_classifier_fragments():
    i = tf.keras.Input(shape=(64, 128))
    x = Bidirectional(GRU(128, return_sequences=True))(i)
    x = Dropout(0.25)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    section_x = Dense(7, activation="softmax")(x)
    fragment_x = Dense(1, activation="sigmoid")(x)
    x = Concatenate(axis=-1)([section_x, fragment_x])

    clf = tf.keras.Model(inputs = i, outputs=x)
    clf.load_weights("../api/models/fine_tuned_small_bert.h5")
    return clf

def build_MiniLM_classifier():
    clf = Sequential()
    clf.add(Bidirectional(GRU(128, return_sequences=True)))
    clf.add(Dropout(0.25))
    clf.add(Dense(64, activation="relu"))
    clf.add(Dropout(0.2))
    clf.add(Dense(7, activation="softmax"))

    clf.build(input_shape=(None, 64, 384))

    clf.load_weights("./models/MiniLM_classifier.h5")
    return clf

def build_MiniLM_classifier_fragments():
    i = tf.keras.Input(shape=(64, 384))
    x = Bidirectional(GRU(128, return_sequences=True))(i)
    x = Dropout(0.25)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    section_x = Dense(7, activation="softmax")(x)
    fragment_x = Dense(1, activation="sigmoid")(x)
    x = Concatenate(axis=-1)([section_x, fragment_x])

    clf = tf.keras.Model(inputs = i, outputs=x)

    clf.load_weights("../api/models/MiniLM_classifier_fragments.h5")
    return clf

def build_MiniLM_classifier_features_fragments():
    i = tf.keras.Input(shape=(64, 391))

    i = tf.keras.layers.Reshape((64, 391, 1), input_shape=(64, 391))(i)

    first_half = tf.keras.layers.Cropping2D(cropping=((0,0), (0,7)))(i)
    first_half = tf.keras.layers.Reshape((64, 384), input_shape=(64, 384, 1))(first_half)

    second_half = tf.keras.layers.Cropping2D(cropping=((0,0), (384,0)))(i)
    second_half = tf.keras.layers.Reshape((64, 7), input_shape=(64, 7, 1))(second_half)

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

    clf = tf.keras.Model(inputs = i, outputs=x)
    clf.load_weights("../api/models/MiniLM_classifier_features_fragments.h5")
    return clf

def build_multi_MiniLM_classifier():
    i = tf.keras.Input(shape=(64, 393))

    i = tf.keras.layers.Reshape((64, 393, 1), input_shape=(64, 393))(i)

    first_half = tf.keras.layers.Cropping2D(cropping=((0,0), (0,9)))(i)
    first_half = tf.keras.layers.Reshape((64, 384), input_shape=(64, 384, 1))(first_half)

    second_half = tf.keras.layers.Cropping2D(cropping=((0,0), (384,0)))(i)
    second_half = tf.keras.layers.Reshape((64, 9), input_shape=(64, 9, 1))(second_half)

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

    clf = tf.keras.Model(inputs = i, outputs=x)
    clf.load_weights("../api/models/multi_miniLM_classifier_check.h5")
    return clf 

if __name__ == "__main__":
    build_multi_bert().summary()
    build_fine_tuned_small_bert().summary()
