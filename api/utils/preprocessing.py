import pandas as pd
import regex as re
import numpy as np
from tqdm import tqdm

tqdm.pandas()

def payload2df(payload):
    df = pd.DataFrame(data=dict(payload))
    return df

def fix_formating(text):
    text=str(text)
    text = text.replace(u'\\xa333', u' ')
    text = text.replace(u'\\u2019', u'\'')
    text = text.replace("\r\n\t\t", "")
    text = text.replace(u' B7; ', '')
    text = text.replace(u'\\xb4', u'\'')
    text = text.replace("&#43;", "+")
    text = text.replace(u'\\xa0', u' ')
    text = text.replace(u'\\xa0', u' ')
    text = text.replace(u'f\\xfcr', u'\'s')
    text = text.replace(u'\\xa', u' x')
    text = text.replace(u'_x000D_', u'')
    text = text.replace(u'x000D', u'\n')
    text = text.replace(u'.à', u' a')
    text = text.replace(u' ', u'')
    text = text.replace(u'‎', u'')
    text = text.replace(u'­', '')
    text = text.replace(u'﻿', u'')
    text = text.replace('&nbsp;', u'')
    text = text.replace('&#43;', '')
    text = text.replace('&lt;', '<')
    text = text.replace('&quot;', '"')
    text = text.replace('&gt;', '>')
    text = text.replace('ï»¿', '')
    text = text.replace('...', '.')
    text = text.replace('..', '.')
    text = text.replace(' .', '. ')
    text = text.replace('\r\n', '\n')
    text = text.replace('\xa0', ' ').replace('：', ': ').replace('\u200b', '').replace('\u2026', '...').replace('’', "'")
    text = text.replace('...', '.')
    text = text.replace('..', '.')
    text = re.sub(r':\s+', ': ', text)
    text = text.replace(' .', '. ')
    text = re.sub(r':\s?\.', ':', text)

    return text.strip('\n').strip().strip('\n')

def clean(text, remove_html=True, email_forward=True):
    text=str(text)
    if remove_html:
        text = re.sub(r"(<|\[)https?:\/\/.*(\.).*(>|\])", "", text, 0, re.M)
        text = re.sub(r"(?:[^\r\n\t\f\v]*{[^{}]*})+", '', text,0, re.MULTILINE)
        text = re.sub(r"(?:[^\r\n\t\f\v]*{[^{}]*})+", '', text, 0, re.MULTILINE)
        text = re.sub(r"[^\r\n\t\f\v]*\s*(\}|\{)\s*|@import.*", "", text, 0, re.MULTILINE)
        text = re.sub(r"\/\*[^*]*\*+([^/*][^*]*\*+)*\/", "", text, 0, re.MULTILINE)
    text= fix_formating(text).strip()
    text=re.sub(r"^(\s*\|\s+)+", "", text)
    text=re.sub(r"\[cid:.*\]", "", text, 0, re.MULTILINE)

    if email_forward:
        text = re.sub(r"^>+[ ]*", "", text, 0, re.MULTILINE)
    return text

def email2list(email):
    email = clean(email)
    email_list = email.split("\n")
    for i in range(len(email_list)-1, -1, -1):
        if not email_list[i].strip():
            email_list.pop(i)
    return email_list

def split(L, N):
    for i in range(0, len(L), N):
        yield L[i:i+N]

def list2sequences(email_list, seq_len=64, padding=""):
    inp_len = len(email_list)
    left = inp_len % seq_len
    inputs = [part for part in split(email_list, seq_len)]
    if left != 0:
        pad = [padding] * (seq_len - left)
        inputs[-1] += pad
    ret = np.array(inputs)
    return ret

def df2sequences(df, training=True, line_df=True, seq_len=64):
    """This function takes a dataframe containing emails and separates them into sequences for the model 
    """
    assert not(training and not line_df), "If training dataframe must be line by line"

    if line_df:
        df["Text"] = df["Text"].apply(str)
        if training:
            df["Section"] = df["Section"].apply(int)
            df = df.groupby("Email").agg({"Text": list,"Section": list, "FragmentChanges": list})
        else:
            df = df.groupby("Email").agg({"Text": list})
    else:
        print("Starting email cleaning".ljust(56, "-"))
        df["Text"] = df["Text"].progress_apply(email2list)
        df.index.name = "Email"

    print("Starting sequence separation".ljust(56, "-"))
    df["Text"] = df["Text"].progress_apply(list2sequences, seq_len=seq_len)
    df = df.explode(column="Text")
    df = df.reset_index()
    df["Part"] = (df.groupby("Email")["Email"].rank(method="first") - 1).astype(int)
    if training:
        print("Starting label separation".ljust(56, "-"))
        df["Section"] = df.progress_apply(lambda row: list2sequences(row["Section"], padding=0, seq_len=seq_len)[row["Part"]], axis=1)
        df["FragmentChanges"] = df.progress_apply(lambda row: list2sequences(row["FragmentChanges"], padding=0, seq_len=seq_len)[row["Part"]], axis=1)
        return df[["Email", "Text", "Part", "Section", "FragmentChanges"]]
    return df[["Email", "Text", "Part"]]

