"""This file defines the usefull data types for the api
"""
from typing import List, Generator, Any
import tensorflow as tf
import pandas as pd
import json
import re
import utils.request_classes as rq
from utils.data_manipulation import flatten_list, batch_list

from config import PREPROCESSING, SECTIONS


class EmailMessage:
    """This class represents a single message within a thread of messages

    attributes:
    - lines: list of lines
    - sections: category associated to each line, listed.
        this attribute is created using the set_sections method
    """

    lines: List[str]
    sections: List[int]

    def __init__(self, lines: List[str]) -> None:
        self.lines = lines

    def set_sections(self, sections_list: List[int]) -> None:
        self.sections = sections_list

    def get(self, section: str) -> str:
        """Get the text associated to a given section of the message
        raises an exception if set_sections has not been called yet
        """
        if self.sections:
            return "\n".join(
                [
                    line
                    for line, sec in zip(self.lines, self.sections)
                    if sec in SECTIONS[section]
                ]
            ).strip("\n")
        else:
            raise Exception("Sections not set")

    def to_dict(self) -> dict:
        return {section: self.get(section) for section in SECTIONS.keys()}

    def to_json(self) -> str:
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

    def __init__(self, text: str) -> None:
        self.source = text
        text = self.fix_formating(text)
        self.lines = self.email2list(text)
        self.messages = []

    @classmethod
    def from_lines(
        cls,
        lines: List[str],
        labels: List[int] = [],
        fragments: List[float] = [],
    ) -> "EmailThread":
        obj = cls("")
        obj.source = "\n".join(lines)
        obj.lines = lines
        if labels and fragments:
            obj.segment(labels, fragments)
        return obj

    def get_sequences(self, seq_len: int = 64) -> List[List[str]]:
        """turn the list of lines into a list of lists of 64 lines (called sequences)"""
        return self.list2sequences(self.lines, seq_len=seq_len)

    def get_label_sequences(self, seq_len: int = 64) -> List[List[int]]:
        """Get a list of sequences of labels"""
        return self.list2sequences(
            flatten_list([m.sections for m in self.messages]),
            padding=0,
            seq_len=seq_len,
        )

    def segment(self, cat_pred: List[int], frag_pred: List[float]) -> "EmailThread":
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
            message.append(line)
            sections.append(cat)
        if message:
            self.messages.append(EmailMessage(message))
            self.messages[-1].set_sections(sections)
        return self

    @staticmethod
    def fix_formating(text: str) -> str:
        """fixes common formatting errors"""
        text = str(text)
        for replacement in PREPROCESSING["text_replacements"]:
            text = text.replace(replacement["pattern"], replacement["replacement"])
        for replacement in PREPROCESSING["regex_replacements"]:
            text = re.sub(
                replacement["pattern"],
                replacement["replacement"],
                text,
                0,
                re.MULTILINE,
            )
        return text.strip("\n").strip().strip("\n")

    @staticmethod
    def email2list(email: str) -> List[str]:
        """transforms a character string to a list of lines (deletes blank lines)"""
        email_list = email.split("\n")
        for i in range(len(email_list) - 1, -1, -1):
            if not email_list[i].strip():
                email_list.pop(i)
        return email_list

    @staticmethod
    def list2sequences(
        email_list: list, seq_len: int = 64, padding: Any = ""
    ) -> List[list]:
        """Creates sequences of specified length with padding for the last one"""
        inp_len = len(email_list)
        left = inp_len % seq_len
        inputs = [part for part in batch_list(email_list, seq_len)]
        if left != 0:
            pad = [padding] * (seq_len - left)
            inputs[-1] += pad
        return inputs

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "messages": [message.to_dict() for message in self.messages],
        }

    def to_json(self) -> str:
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
    def from_json(cls, json_str: rq.ThreadList) -> "EmailDataset":
        return cls(dict(json_str)["threads"])

    @classmethod
    def from_csv(cls, csv_file: str) -> "EmailDataset":
        """Create labeled dataset from csv file

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
                "Text": list,
                "Section": list,
                "FragmentChange": list,
            }
        )
        threads = df["Text"].tolist()
        labels = df["Section"].tolist()
        fragments = df["FragmentChange"].tolist()
        obj = cls([])
        obj.threads = [
            EmailThread.from_lines(thread).segment(label, fragment)
            for thread, label, fragment in zip(threads, labels, fragments)
        ]
        obj.is_labeled = True
        obj.build_dataset()
        return obj

    def get_tf_dataset(self) -> tf.data.Dataset:
        return self.dataset

    def to_csv(self, csv_file: str) -> None:
        """Save dataset to csv file.

        TODO: implement this method"""
        raise NotImplementedError

    def build_dataset(self, batch_size: int = 16) -> "EmailDataset":
        sequences = [thread.get_sequences() for thread in self.threads]
        self.batch_size = batch_size
        self.seq_order = [i for i, seqs in enumerate(sequences) for _ in seqs]
        if self.is_labeled:
            lab_sequences = [thread.get_label_sequences() for thread in self.threads]
            for i, seqs in enumerate(lab_sequences):
                if len(seqs) != len(sequences[i]):
                    print(seqs)
                    print(sequences[i])
            self.dataset = tf.data.Dataset.from_tensor_slices(
                (flatten_list(sequences), flatten_list(lab_sequences))
            ).batch(self.batch_size)
        else:
            self.dataset = tf.data.Dataset.from_tensor_slices(
                flatten_list(sequences)
            ).batch(self.batch_size)
        return self

    def to_dict(self) -> dict:
        return {"threads": [thread.to_dict() for thread in self.threads]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class EmailLineDataset:
    dataset: tf.data.Dataset

    def __init__(self):
        pass

    @classmethod
    def from_csv(cls, csv_file: str) -> "EmailLineDataset":
        """Create labeled dataset from csv file

        Expected Columns in csv file:
        - Email: email number to group lines by
        - Text: text of the line of the email
        - Label: label of the line of the email
        - Fragment: fragment changes equal to 1 when the line corresponds to a new fragment
        """
        df = pd.read_csv(csv_file)
        df["Text"] = df["Text"].astype(str)

        threads = df["Text"].tolist()
        labels = df["Section"].tolist()
        obj = cls()
        obj.dataset = tf.data.Dataset.from_tensor_slices((threads, labels)).batch(3)
        return obj

    def get_tf_dataset(self) -> tf.data.Dataset:
        return self.dataset
