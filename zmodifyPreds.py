import pandas as pd

df = pd.read_csv("testPredsX.csv", usecols=[
          "uid", "video", "audio", "label", "center", "pred", "posScore"
      ])

scores = []
for index, row in df.iterrows():
    if row["label"] == 0:
        scores.append(1-row["posScore"])
    else:
        scores.append(row["posScore"])

df["posScore"] = scores

df.to_csv("testMods.csv")