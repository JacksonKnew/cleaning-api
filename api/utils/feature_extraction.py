import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sentence_transformers import SentenceTransformer

tqdm.pandas()

encoder = tf.keras.models.load_model("./fine_tuned_multi_bert.h5")

def create_encoding(df, enc_model="sentence-transformers/all-MiniLM-L6-v2"):
    print("Starting encoding of inputs".ljust(56, "-"))
    df["Encoding"] = df["Text"].progress_apply(encoder.encode)
    return df

def extract_phone_numbers(text):
    phone_regex = re.compile("^\\+?\\d{1,4}?[-.\\s]?\\(?\\d{1,3}?\\)?[-.\\s]?\\d{1,4}[-.\\s]?\\d{1,4}[-.\\s]?\\d{1,9}$")
    return re.findall(phone_regex, text)

def extract_url(text):
    url_regex = re.compile("^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$")
    return re.findall(url_regex, text)

def extract_punctuation(text):
    punct_regex = re.compile("[!?]")
    return re.findall(punct_regex, text)

def extract_special(text):
    special_regex = re.compile("[*_\-=#~|]")
    return re.findall(special_regex, text)

def extract_features(text):
    feats = []
    for line in text:
        feats.append([
            len(extract_url(line)),
            len(extract_phone_numbers(line)),
            len(extract_punctuation(line)),
            len(extract_special(line)),
        ])
    return np.array(feats)

def create_extracted(df):
    print("Starting feature extraction".ljust(56, "-"))
    df["Extracted"] = df["Text"].progress_apply(extract_features)
    return df

def create_features(df, encoding=None, extracted=None):
    n_inputs = df.shape[0]
    if encoding:
        encode = np.array([enc for enc in df[encoding].values])
    if extracted:
        extract = np.array([ext for ext in df[extracted].values])
    if (encoding and extracted):
        features = np.concatenate((encode, extract), axis=2)
    else:
        features = df[encoding or extracted].values
    return np.asarray([feat for feat in features]).astype('float32')