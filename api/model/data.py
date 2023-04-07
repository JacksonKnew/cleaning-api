"""This file defines the usefull data types for the api
"""
from typing import List
import tensorflow as tf
import pandas as pd
import json
import re
from tqdm import tqdm


class EmailMessage:
    """This class represents a single message within a thread of messages

    attributes:
    - lines: list of lines
    - sections: category associated to each line, listed.
        this attribute is created using the set_sections method
    """

    lines: List[str]
    sections: List[int]

    sections_dict = {
        "full": [i for i in range(1, 6)],
        "header": [1],
        "disclaimer": [2],
        "greetings": [3],
        "body": [4],
        "signature": [5],
        "caution": [6],
    }

    def __init__(self, lines: List[str]):
        self.lines = lines

    def set_sections(self, sections_list: List[int]):
        self.sections = sections_list

    def get(self, section):
        """Get the text associated to a given section of the message
        raises an exception if set_sections has not been called yet
        """
        if self.sections:
            return "\n".join(
                [
                    line
                    for line, sec in zip(self.lines, self.sections)
                    if sec in self.sections_dict[section]
                ]
            ).strip("\n")
        else:
            raise Exception("Sections not set")

    def to_dict(self):
        return {
            "full": self.get("full"),
            "header": self.get("header"),
            "disclaimer": self.get("disclaimer"),
            "greetings": self.get("greetings"),
            "body": self.get("body"),
            "signature": self.get("signature"),
            "caution": self.get("caution"),
        }

    def to_json(self):
        return json.dumps(self.to_dict())


class EmailThread:
    """This class represents a thread of messages. A source string will create one EmailThread object

    attributes:
    - source: source text given at instantiation
    - lines: cleaned up lines of the email stored in a list
    - messages: list of EmailMessage objects. created when segment is called with the output of a model
    """

    source: str
    lines: List[str]
    messages: List[EmailMessage]

    def __init__(self, text: str):
        self.source = text
        text = self.clean(text)
        self.lines = self.email2list(text)
        self.messages = []

    @classmethod
    def from_lines(
        cls, lines: List[str], labels: List[int] = None, fragments: List[int] = None
    ):
        obj = cls("")
        obj.source = "\n".join(lines)
        obj.lines = lines
        if labels and fragments:
            obj.segment(labels, fragments)
        return obj

    def get_sequences(self, seq_len=64):
        """turn the list of lines into a list of lists of 64 lines (called sequences)"""
        return self.list2sequences(self.lines, seq_len=seq_len)

    def get_label_sequences(self, seq_len=64):
        """Get a list of sequences of labels

        TODO: implement this method"""
        pass

    def segment(self, cat_pred, frag_pred):
        """segment the thread into messages using the output of a model

        Arguments:
            cat_pred {list} -- list of category predictions
            frag_pred {list} -- list of fragment predictions
        """
        message = []
        sections = []
        for line, cat, frag in zip(self.lines, cat_pred, frag_pred):
            if frag >= 0.5:
                if message:
                    self.messages.append(EmailMessage(message))
                    self.messages[-1].set_sections(sections)
                message = []
                sections = []
            else:
                message.append(line)
                sections.append(cat)
        if message:
            self.messages.append(EmailMessage(message))
            self.messages[-1].set_sections(sections)
        return self

    @staticmethod
    def fix_formating(text):
        """fixes common formatting errors"""
        text = str(text)
        text = text.replace("\\xa333", " ")
        text = text.replace("\\u2019", "'")
        text = text.replace("\r\n\t\t", "")
        text = text.replace(" B7; ", "")
        text = text.replace("\\xb4", "'")
        text = text.replace("&#43;", "+")
        text = text.replace("\\xa0", " ")
        text = text.replace("\\xa0", " ")
        text = text.replace("f\\xfcr", "'s")
        text = text.replace("\\xa", " x")
        text = text.replace("_x000D_", "")
        text = text.replace("x000D", "\n")
        text = text.replace(".à", " a")
        text = text.replace(" ", "")
        text = text.replace("‎", "")
        text = text.replace("­", "")
        text = text.replace("﻿", "")
        text = text.replace("&nbsp;", "")
        text = text.replace("&#43;", "")
        text = text.replace("&lt;", "<")
        text = text.replace("&quot;", '"')
        text = text.replace("&gt;", ">")
        text = text.replace("ï»¿", "")
        text = text.replace("...", ".")
        text = text.replace("..", ".")
        text = text.replace(" .", ". ")
        text = text.replace("\r\n", "\n")
        text = text.replace("\xa0", " ")
        text = text.replace("：", ": ")
        text = text.replace("\u200b", "")
        text = text.replace("\u2026", "...")
        text = text.replace("’", "'")
        text = text.replace("...", ".")
        text = text.replace("..", ".")
        text = re.sub(r":\s+", ": ", text)
        text = text.replace(" .", ". ")
        text = re.sub(r":\s?\.", ":", text)

        return text.strip("\n").strip().strip("\n")

    @classmethod
    def clean(cls, text, remove_html=True, email_forward=True):
        """fixes formatting, html and other issues in a single character string"""
        text = str(text)
        if remove_html:
            text = re.sub(r"(<|\[)https?:\/\/.*(\.).*(>|\])", "", text, 0, re.M)
            text = re.sub(r"(?:[^\r\n\t\f\v]*{[^{}]*})+", "", text, 0, re.MULTILINE)
            text = re.sub(r"(?:[^\r\n\t\f\v]*{[^{}]*})+", "", text, 0, re.MULTILINE)
            text = re.sub(
                r"[^\r\n\t\f\v]*\s*(\}|\{)\s*|@import.*", "", text, 0, re.MULTILINE
            )
            text = re.sub(r"\/\*[^*]*\*+([^/*][^*]*\*+)*\/", "", text, 0, re.MULTILINE)
        text = cls.fix_formating(text).strip()
        text = re.sub(r"^(\s*\|\s+)+", "", text)
        text = re.sub(r"\[cid:.*\]", "", text, 0, re.MULTILINE)

        if email_forward:
            text = re.sub(r"^>+[ ]*", "", text, 0, re.MULTILINE)
        return text

    @staticmethod
    def email2list(email):
        """transforms a character string to a list of lines (deletes blank lines)"""
        email_list = email.split("\n")
        for i in range(len(email_list) - 1, -1, -1):
            if not email_list[i].strip():
                email_list.pop(i)
        return email_list

    @staticmethod
    def split(L, N):
        """batches list L into N size chunks. Returns a generator"""
        for i in range(0, len(L), N):
            yield L[i : i + N]

    @classmethod
    def list2sequences(cls, email_list, seq_len=64, padding=""):
        """Creates sequences of specified length with padding for the last one"""
        inp_len = len(email_list)
        left = inp_len % seq_len
        inputs = [part for part in cls.split(email_list, seq_len)]
        if left != 0:
            pad = [padding] * (seq_len - left)
            inputs[-1] += pad
        return inputs

    def to_dict(self):
        return {
            "source": self.source,
            "messages": [message.to_dict() for message in self.messages],
        }

    def to_json(self):
        return json.dumps(self.to_dict())


class EmailDataset:
    """This class is used to feed a list of emails to a pipeline and segment each email

    attributes:
    - threads: list of EmailThread objects
    - batch_size: batch_size for models
    - seq_order: list of integers used to know which emails are split in multiple sequences
    - dataset: tf.data.Datset object created to feed the pipeline
    """

    threads: List[EmailThread]
    seq_order: List[int]
    batch_size: int
    dataset: tf.data.Dataset
    is_labeled: bool = False

    def __init__(
        self,
        threads: List[str],
        batch_size=16,
    ):
        self.threads = [EmailThread(str(thread)) for thread in threads]
        self.build_dataset(batch_size)
        self.is_labeled = False

    @classmethod
    def from_json(cls, json_str):
        return cls(dict(json_str)["threads"])

    @classmethod
    def from_csv(cls, csv_file):
        """Create labeles dataset from csv file

        Expected Columns in csv file:
        - Email: email number to group lines by
        - Text: text of the line of the email
        - Label: label of the line of the email
        - Fragment: fragment changes equal to 1 when the line corresponds to a new fragment
        """
        df = pd.read_csv(csv_file)
        df["Text"] = df["Text"].astype(str)
        df = df.groupby("Email").agg(
            {
                "Email": "first",
                "Text": list,
                "Label": list,
                "Fragment": list,
            }
        )
        threads = df["Text"].tolist()
        labels = df["Label"].tolist()
        fragments = df["Fragment"].tolist()
        obj = cls([])
        obj.threads = [
            EmailThread.from_lines(thread).segment(label, fragment)
            for thread, label, fragment in zip(threads, labels, fragments)
        ]
        obj.is_labeled = True
        obj.build_dataset()
        return obj

    def get_tf_dataset(self):
        return self.dataset

    def to_csv(self, csv_file):
        """Save dataset to csv file.

        TODO: implement this method"""
        pass

    def build_dataset(self, batch_size=16):
        sequences = [thread.get_sequences() for thread in self.threads]
        self.batch_size = batch_size
        self.seq_order = [i for i, seqs in enumerate(sequences) for seq in seqs]
        if self.is_labeled:
            lab_sequences = [thread.get_label_sequences() for thread in self.threads]
            self.dataset = tf.data.Dataset.from_tensor_slices(
                (self._flatten_list(sequences), self._flatten_list(lab_sequences))
            ).batch(self.batch_size)
        else:
            self.dataset = tf.data.Dataset.from_tensor_slices(
                self._flatten_list(sequences)
            ).batch(self.batch_size)
        return self

    @staticmethod
    def _flatten_list(L):
        return [x for l in L for x in l]

    @staticmethod
    def _chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def to_dict(self):
        return {"threads": [thread.to_dict() for thread in self.threads]}

    def to_json(self):
        return json.dumps(self.to_dict())
