import numpy as np
import pandas as pd
from tqdm import tqdm

def pred2dic(pred, df):
    current_email=-1
    pred = np.argmax(pred, axis=-1)
    labels = {
        2: "caution",
        3: "greetings",
        4: "content",
        5: "signature",
        6: "disclaimer"
    }
    texts = {
        "caution": [],
        "greetings": [],
        "content": [],
        "signature": [],
        "disclaimer": []
    }
    print("Recreating text sequences".ljust(56, "-"))
    for i, row in tqdm(df.iterrows()):
        if row["Email"] != current_email:
            for key in texts.keys():
                texts[key].append([])
            current_email = row["Email"]
        for j, line in enumerate(row["Text"]):
            if line and pred[i,j] in labels.keys():
                texts[labels[pred[i,j]]][-1].append(line)
    print("json creation".ljust(56, "-"))
    for key in tqdm(texts.keys()):
        for i in range(len(texts[key])):
            texts[key][i] = "\n".join(texts[key][i]).strip("\n")
    return texts

