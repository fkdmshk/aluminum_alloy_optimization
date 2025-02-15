from fastapi import FastAPI, Query
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import shap
import pickle
import os
import numpy as np
import seaborn as sns  # Heatmap描画用に追加

app = FastAPI()

# 学習データのロード
data_path = r"C:\Users\FMV\Desktop\アルミ合金最適化\optimal_materials.csv"
data = pd.read_csv(data_path)

# Material_Name列を生成（存在しない場合）
if 'Material_Name' not in data.columns:
    if 'index' in data.columns:
        data['Material_Name'] = data['index']
    else:
        data['Material_Name'] = "Unknown"

# Streamlitアプリのタイトル
st.title("Aluminum Alloy Optimization App")

# サイドバーに注意喚起を表示
st.sidebar.header("Important Factors Influencing Properties")
st.sidebar.subheader("1. Element Composition")
st.sidebar.write("""
- **Mg (Magnesium):** Increases strength and corrosion resistance.
- **Si (Silicon):** Improves wear resistance and thermal stability.
- **Cu (Copper):** Enhances strength and reduces elongation.
""")

st.sidebar.subheader("2. Manufacturing Process")
st.sidebar.write("""
- Casting: Results in good machinability but lower mechanical strength.
- Forging: Provides high strength and excellent fatigue resistance.
""")

st.sidebar.subheader("3. Heat Treatment (Refining)")
st.sidebar.write("""
- T6: Enhances both strength and hardness.
- O: Provides maximum elongation but lower strength.
""")

st.sidebar.subheader("4. Processing Methods")
st.sidebar.write("""
- Rolling: Improves tensile strength and elongation in one direction.
- Extrusion: Enhances properties along the extrusion direction.
""")

# 入力パラメータの設定
st.sidebar.header("Input Parameters")
tairyoku_threshold = st.sidebar.slider("Tairyoku Threshold", min_value=0, max_value=500, value=300)
nobi_threshold = st.sidebar.slider("Nobi Threshold", min_value=0, max_value=100, value=10)

# 条件を満たす最適材料のフィルタリング
optimal_materials = data[
    (data['Pred_Tairyoku'] >= tairyoku_threshold) & (data['Pred_Nobi'] >= nobi_threshold)
]

# 材料名でグループ化し、要因を表示
grouped_materials = optimal_materials.groupby('Material_Name').agg({
    'Refining': lambda x: ', '.join(map(str, x.unique())),
    'Manufacturing': lambda x: ', '.join(map(str, x.unique())),
    'Surface': lambda x: ', '.join(map(str, x.unique())),
    'Shiobori': lambda x: ', '.join(map(str, x.unique()))
}).reset_index()

st.header("Optimal Materials (Details by Material Name)")
selected_material = st.selectbox(
    "Select a Material to Highlight on the Scatter Plot:",
    grouped_materials['Material_Name']
)

st.dataframe(grouped_materials)

# 散布図の描画（背景色変更と注釈追加）
st.header("Scatter Plot of Predicted Tairyoku and Nobi")

fig, ax = plt.subplots(figsize=(10, 6))

# 背景色を変更
ax.set_facecolor('#f0f8ff')  # 淡い青色

# データプロット
scatter_all = ax.scatter(data['Pred_Tairyoku'], data['Pred_Nobi'], alpha=0.5, label='All Materials')
scatter_optimal = ax.scatter(optimal_materials['Pred_Tairyoku'], optimal_materials['Pred_Nobi'], 
                              color='red', alpha=0.7, label='Optimal Materials')

# 閾値線の追加
ax.axvline(x=tairyoku_threshold, color='blue', linestyle='--', linewidth=2, label='Tairyoku Threshold')
ax.axhline(y=nobi_threshold, color='green', linestyle='--', linewidth=2, label='Nobi Threshold')

# ユーザーが選択した材料を注釈
highlighted_points = optimal_materials[optimal_materials['Material_Name'] == selected_material]
if not highlighted_points.empty:
    for _, point in highlighted_points.iterrows():
        ax.annotate(f"Material: {point['Material_Name']}\n(Tairyoku={point['Pred_Tairyoku']}, Nobi={point['Pred_Nobi']})",
                    xy=(point['Pred_Tairyoku'], point['Pred_Nobi']),
                    xytext=(point['Pred_Tairyoku'] + 30, point['Pred_Nobi'] + 5),
                    arrowprops=dict(facecolor='black', arrowstyle='->'),
                    fontsize=10, color='darkred', bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='white'))

ax.set_xlabel("Predicted Tairyoku", fontsize=12)
ax.set_ylabel("Predicted Nobi", fontsize=12)
ax.set_title("Scatter Plot of Predicted Tairyoku and Nobi", fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True)

st.pyplot(fig)

# ヒートマップの表示
st.header("Heatmap of Material Properties")
st.write("This heatmap shows the correlation between material properties.")

numeric_data = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
st.pyplot(plt)

# 引張試験の動画
st.header("JIS Aluminum Material Testing")
st.subheader("Example of Strength and Elongation Tests/YouTube")
st.video("https://youtu.be/b0FiWTuDyGg")

# SHAP解析結果の表示
st.header("SHAP Analysis")
model_path = r"C:\Users\FMV\Desktop\アルミ合金最適化\rf_nobi.pkl"
X_test_path = r"C:\Users\FMV\Desktop\アルミ合金最適化\X_test_t.csv"

try:
    with open(model_path, "rb") as f:
        rf_nobi = pickle.load(f)
    X_test_t = pd.read_csv(X_test_path)
    explainer = shap.TreeExplainer(rf_nobi)
    shap_values = explainer.shap_values(X_test_t)
    st.subheader("SHAP Summary Plot (Bar)")
    plt.figure()
    shap.summary_plot(shap_values, X_test_t, plot_type="bar", show=False)
    st.pyplot(plt)
except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")

# モデル評価の結果を表示
st.header("Model Evaluation")
mse_nobi = mean_squared_error(data['Tairyoku'], data['Pred_Tairyoku'])
r2_nobi = r2_score(data['Tairyoku'], data['Pred_Tairyoku'])
st.text(f"Tairyoku: MSE = {mse_nobi:.2f}, R2 = {r2_nobi:.2f}")

# 材料力学に関する追加動画
st.header("材料力学　基本")
st.subheader("引張試験編/超初心者のための材料力学その3〜引張り試験〜/YouTube")
st.video("https://youtu.be/euSlVokRMK0")