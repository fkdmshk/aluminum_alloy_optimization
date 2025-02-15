# api_app.py

from fastapi import FastAPI
import pandas as pd

app = FastAPI()

# 1) トップページなど、最低1つのエンドポイント
@app.get("/")
def home():
    return {"message": "Hello from FastAPI!"}

# 2) CSV 読み込みサンプル (任意)
data_path = r"C:\Users\FMV\Desktop\アルミ合金最適化\optimal_materials.csv"

@app.get("/materials")
def get_materials():
    df = pd.read_csv(data_path)
    # 必要なカラムだけ返す、または df 全体を返すなど
    return df.head(10).to_dict(orient="records")

