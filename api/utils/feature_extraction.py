import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import utils.tf_model_creation as mc
from transformers import AutoTokenizer

tqdm.pandas()

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
encoder = mc.TFSentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def encode_text(text):
    lines = [str(line) for line in text]
    tokens = tokenizer(lines, padding=True, truncation=True, return_tensors='tf')
    return encoder(tokens)

def create_encoding(df, enc_model="sentence-transformers/all-MiniLM-L6-v2"):
    print("Starting encoding of inputs".ljust(56, "-"))
    df["Encoding"] = df["Text"].progress_apply(encode_text)
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

def extract_horizontal_separators(text):
    special_regex = re.compile("[-=~]")
    return re.findall(special_regex, text)

def extract_hashtags(text):
    hashtag_regex = re.compile("[#]")
    return re.findall(hashtag_regex, text)

def extract_pipes(text):
    pipes_regex = re.compile("[|]")
    return re.findall(pipes_regex, text)

def extract_email(text):
    email_regex = re.compile("[-\w\.]+@([-\w]+\.)+[-\w]{2,4}")
    return re.findall(email_regex, text)

def extract_capitalized(text):
    capitalized_regex = re.compile("[A-Z][a-z]*")
    return re.findall(capitalized_regex, text)

def extract_full_caps(text):
    full_caps_regex = re.compile("[A-Z]+")
    return re.findall(full_caps_regex, text)

def extract_features(text):
    feats = []
    for line in text:
        feats.append([
            len(extract_url(line)),
            len(extract_phone_numbers(line)),
            len(extract_punctuation(line)),
            len(extract_horizontal_separators(line)),
            len(extract_hashtags(line)),
            len(extract_pipes(line)),
            len(extract_email(line)),
            len(extract_capitalized(line)),
            len(extract_full_caps(line)),
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
        features = np.concatenate((encode, extract), axis=-1)
    else:
        features = df[encoding or extracted].values
    return np.asarray([feat for feat in features]).astype('float32')