from typing import List, Optional
from pydantic import BaseModel


class PipelineName(BaseModel):
    name: str


class ThreadList(BaseModel):
    threads: List[str]


class LabeledDataset(BaseModel):
    threads: List[str]
    labels: List[int]
    metrics: List[str]


class TrainRequest(BaseModel):
    csv_path: str
    epochs: int
