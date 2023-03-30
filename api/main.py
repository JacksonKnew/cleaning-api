import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from fastapi import FastAPI
import json
from typing import List, Optional
from pydantic import BaseModel
import utils.preprocessing as pp
import utils.feature_extraction as fc
import utils.result_handling as rh
import utils.model_building as mb
import utils.cosmetics as cos

app = FastAPI()

model = mb.build_multi_MiniLM_classifier()

@app.get("/")
def welcome():
    greetings = {"Hello": "world"}
    return json.dumps(greetings)

class Email(BaseModel):
    Text: List[str]

@app.post("/segment")
def segmentation(email: Email):
    email_df = pp.payload2df(email)
    cos.received_request("POST", "/segment", email_df.shape[0], 56)
    email_df = pp.df2sequences(email_df, training=False, line_df=False)
    email_df = fc.create_encoding(email_df)
    email_df = fc.create_extracted(email_df)
    feats = fc.create_features(email_df, encoding="Encoding", extracted="Extracted")
    print("Starting model prediction".ljust(56, "-"))
    pred = model.predict(feats)
    result = rh.pred2fragdic(pred, email_df)
    return json.dumps(result)

