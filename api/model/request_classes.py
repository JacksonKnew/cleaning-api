from typing import List
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
    model_name: str
    metrics: List[str]
    loss_weights: List[float]
    epochs: int
