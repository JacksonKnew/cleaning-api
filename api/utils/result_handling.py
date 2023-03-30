import numpy as np
import pandas as pd
from tqdm import tqdm

def flatten_list(L):
    return [x for l in L for x in l]

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

def pred2fragdic(pred, df):
    current_email=-1
    sect = np.argmax(pred[:, :, :7], axis=-1)
    frag = np.round(pred[:, :, -1])
    labels = {
        1: "header",
        2: "caution",
        3: "greetings",
        4: "content",
        5: "signature",
        6: "disclaimer"
    }
    texts = []
    df["Pred"] = [s for s in sect]
    df["Frag"] = [f for f in frag]
    emails = df.groupby("Email").agg({
        "Text": list,
        "Pred": list,
        "Frag": list,
    })
    for col in emails.columns:
        emails[col] = emails[col].apply(flatten_list)
    for i, row in emails.iterrows():
        new_text = [[]]
        new_pred = [[]]
        for j, frag_split in enumerate(row["Frag"]):
            if frag_split:
                new_text.append([])
                new_pred.append([])
            new_text[-1].append(row["Text"][j])
            new_pred[-1].append(row["Pred"][j])
        emails.loc[i, "Text"] = new_text
        emails.loc[i, "Pred"] = new_pred
    for i, row in emails.iterrows():
        all_sections = []
        full_text = "\n".join(flatten_list(row["Text"])).strip("\n")
        thread_length = len(row["Text"])
        for text, section in zip(row["Text"], row["Pred"]):
            sect_dic = {
                "full_text": [],
                "header": [],
                "caution": [],
                "greetings": [],
                "content": [],
                "signature": [],
                "disclaimer": []
            }
            for line, sect in zip(text, section):
                sect_dic["full_text"].append(line)
                if line and sect in labels.keys():
                    sect_dic[labels[sect]].append(line)
            for key in sect_dic.keys():
                sect_dic[key] = "\n".join(sect_dic[key]).strip("\n")
            all_sections.append(sect_dic)
        texts.append({
            "full_text": full_text,
            "length": thread_length,
            "sections": all_sections
        })
    return {"Text": texts}


