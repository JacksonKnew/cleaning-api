"""This file defines the usefull classes for the api
"""
from typing import List
import utils.preprocessing as pp


class EmailMessage:
    lines: List(str)
    sections: List(int)

    def read(self, text):
        text = pp.clean(text)
        self.lines = pp.email2list(text)

    def get(self, section):
        pass


class EmailThread:
    text: str
    messages: List(EmailMessage)

    def detect():
        pass
