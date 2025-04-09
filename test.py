import pandas as pd
df = pd.read_csv("permit_data.csv", nrows=250000)
df.to_csv("permit_data.csv", index=False)