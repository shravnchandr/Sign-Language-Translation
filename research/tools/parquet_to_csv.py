import pandas

df = pandas.read_parquet(r"Kaggle_Data/train_landmark_files/16069/100015657.parquet")
df.to_csv("100015657.csv")