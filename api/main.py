from fastapi import FastAPI
import json
from typing import List, Optional
from pydantic import BaseModel
import pandas as pd
import numpy as np
import tensorflow as tf
import utils.preprocessing as pp
import utils.feature_extraction as fc
import utils.result_handling as rh

app = FastAPI()

model = tf.keras.models.load_model("./model")

@app.get("/")
def welcome():
    greetings = {"Hello": "world"}
    return json.dumps(greetings)

class Email(BaseModel):
    Text: List[str]

@app.post("/segment")
def segmentation(email: Email):
    email_df = pp.payload2df(email)
    email_df = pp.df2sequences(email_df, training=False, line_df=False)
    email_df = fc.create_encoding(email_df)
    feats = fc.create_features(email_df, encoding="Encoding")
    print("Starting model prediction".ljust(56, "-"))
    pred = model.predict(feats)
    result = rh.pred2dic(pred, email_df)
    return json.dumps(result)

