# =========================================
# 🌲 FOREST COVER TYPE - STREAMLIT APP
# OPTIMIZED VERSION (PART 1)
# =========================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Forest Cover Type Prediction via XGBoost",
    page_icon="📊🌲",
    layout="wide"
)

st.title("🌲 Forest Cover Type Prediction via XGBoost")

# =========================================
# CACHE MANAGEMENT
# =========================================

if st.button("🧹 Clear Streamlit Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("✅ Cache cleared! Reload the app to start fresh.")

# =========================================
# DATA LOADING
# =========================================

@st.cache_data(show_spinner=False)
def load_data():

    data = fetch_covtype()

    X = data.data.astype(np.float32)
    y = (data.target - 1).astype(np.int8)

    df = pd.DataFrame(X)
    df["Cover_Type"] = y + 1

    return df


@st.cache_data(show_spinner=False)
def get_split(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


@st.cache_data(show_spinner=False)
def get_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# =========================================
# LOAD DATASET
# =========================================

df = load_data()

feature_df = df.drop(columns=["Cover_Type"])

X = feature_df.values.astype(np.float32)

y = (df["Cover_Type"].values - 1).astype(np.int8)

# =========================================
# SUMMARY
# =========================================

with st.expander("🧠 Summary", expanded=False):
    st.markdown("""
Forest type classification based on terrain, soil, hydrology, and ecological conditions.
""")

# =========================================
# DATASET PREVIEW
# =========================================

with st.expander("📋 Dataset Preview (First 50 rows)", expanded=False):

    st.dataframe(
        df.head(50),
        use_container_width=True
    )

    st.write("📊 Showing first 50 rows")

# =========================================
# DOWNLOAD BUTTON
# =========================================

csv = get_csv(df)

st.download_button(
    label="💾 Download Full Dataset",
    data=csv,
    file_name="forest_cover_dataset.csv",
    mime="text/csv"
)

st.markdown("---")

# =========================================
# DATASET EXPLANATION
# =========================================

with st.expander("📘 Dataset Explanation", expanded=False):

    st.markdown("""
# 🌲 FOREST COVER TYPE DATASET

Dataset with geographical and ecological Variables.
""")

    with st.expander("🎯 Objective", expanded=False):
        st.markdown("Predict forest cover type.")

    with st.expander("📌 Variables 0–2 ⛰️ Basic Topography", expanded=False):
        st.markdown("""
0: Elevation ➜ Height above sea level (meters)  
1: Aspect ➜ Slope direction (0 to 360 grades)  
2: Slope ➜ Terrain inclination (0 to 90 grades)
""")

    with st.expander("📌 Variables 3–5 🛣️ Distances & Infrastructure", expanded=False):
        st.markdown("""
3: Horizontal_Distance_To_Hydrology ➜ Distance to nearest surface water  
4: Vertical_Distance_To_Hydrology ➜ Vertical distance to water source  
5: Horizontal_Distance_To_Roadways ➜ Distance to nearest road
""")

    with st.expander("📌 Variables 6–8 ☀️ Solar Exposure", expanded=False):
        st.markdown("""
6: Hillshade_9am ➜ Solar radiation at 9 AM  
7: Hillshade_Noon ➜ Solar radiation at noon  
8: Hillshade_3pm ➜ Solar radiation at 3 PM
""")

    with st.expander("📌 VARIABLE 9 🔥 Fire History", expanded=False):
        st.markdown("""
9: Horizontal_Distance_To_Fire_Points ➜ Distance to nearest fire ignition point
""")

    with st.expander("📌 Variables 10–13 🌲 Wilderness Areas", expanded=False):
        st.markdown("""
10: Wilderness_Area_1 ➜ Rawah Wilderness Area  
11: Wilderness_Area_2 ➜ Neota Wilderness Area  
12: Wilderness_Area_3 ➜ Comanche Peak Wilderness Area  
13: Wilderness_Area_4 ➜ Cache la Poudre Wilderness Area
""")

    with st.expander("📌 Variables 14–53 🌱 SOIL TYPES", expanded=False):
        st.markdown("""
14: Soil_Type1 ➜ Rocky shallow soil  
15: Soil_Type2 ➜ Volcanic soil  
16: Soil_Type3 ➜ Alluvial soil  
17: Soil_Type4 ➜ Shallow rocky soil  
18: Soil_Type5 ➜ Moist valley soil  
19: Soil_Type6 ➜ Organic soil  
20: Soil_Type7 ➜ Extremely rocky soil  
21: Soil_Type8 ➜ Mixed rock soil  
22: Soil_Type9 ➜ Poor drainage soil  
23: Soil_Type10 ➜ Mountain-valley soil  
24: Soil_Type11 ➜ Eroded soil  
25: Soil_Type12 ➜ Compact soil  
26: Soil_Type13 ➜ Fertile mineral soil  
27: Soil_Type14 ➜ Dry soil  
28: Soil_Type15 ➜ Deep fertile soil  
29: Soil_Type16 ➜ Rock-organic mix  
30: Soil_Type17 ➜ Cold mountain soil  
31: Soil_Type18 ➜ Eroded soil  
32: Soil_Type19 ➜ Wet soil  
33: Soil_Type20 ➜ Transition soil  
34: Soil_Type21 ➜ Compact soil  
35: Soil_Type22 ➜ Glacial soil  
36: Soil_Type23 ➜ Sedimentary soil  
37: Soil_Type24 ➜ Rocky soil  
38: Soil_Type25 ➜ Stable deep soil  
39: Soil_Type26 ➜ Fast draining soil  
40: Soil_Type27 ➜ Wet lowland soil  
41: Soil_Type28 ➜ Poor soil  
42: Soil_Type29 ➜ Transition soil  
43: Soil_Type30 ➜ Mixed soil  
44: Soil_Type31 ➜ Complex soil  
45: Soil_Type32 ➜ Mixed geology soil  
46: Soil_Type33 ➜ Heterogeneous soil  
47: Soil_Type34 ➜ Variable structure  
48: Soil_Type35 ➜ Variable moisture  
49: Soil_Type36 ➜ Mineral soil  
50: Soil_Type37 ➜ Sloped soil  
51: Soil_Type38 ➜ High mountain rocky soil  
52: Soil_Type39 ➜ Deep glacial soil  
53: Soil_Type40 ➜ Alpine humid soil
""")

    with st.expander("📌 VARIABLE 54 🎯 Target Variable", expanded=False):
        st.markdown("""
Cover_Type ➜ forest type (1 to 7)

1: Spruce/Fir  
2: Lodgepole Pine  
3: Ponderosa Pine  
4: Cottonwood/Willow  
5: Aspen  
6: Douglas-fir  
7: Krummholz
""")
        
# =========================================
# 🌲 PART 2 - METRICS + DISTRIBUTION + TRAINING
# =========================================

# =========================================
# DATASET METRICS
# =========================================

with st.expander("📊 Dataset Metrics", expanded=False):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", f"{len(df):,}")

    with col2:
        st.metric("Features", f"{df.shape[1] - 1}")

    with col3:
        st.metric("Classes", "7")

# =========================================
# CLASS DISTRIBUTION
# =========================================

with st.expander("📈 Class Distribution", expanded=False):

    unique, counts = np.unique(y, return_counts=True)

    dist_df = pd.DataFrame({
        "Class": unique + 1,
        "Count": counts
    })

    st.dataframe(dist_df, use_container_width=True)

    fig = px.bar(
        dist_df,
        x="Class",
        y="Count",
        text="Count",
        title="Class Distribution"
    )

    fig.update_layout(
        height=450,
        xaxis_title="Class",
        yaxis_title="Count"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================
# TRAINING FUNCTION (OPTIMIZED XGBOOST)
# =========================================

@st.cache_resource(show_spinner=False)
def train_model(X_train, y_train, n_estimators, max_depth, learning_rate):

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,

        subsample=0.8,
        colsample_bytree=0.8,

        eval_metric="mlogloss",

        # PERFORMANCE BOOST
        tree_method="hist",
        max_bin=64,
        n_jobs=-1,

        random_state=42,
        verbosity=0
    )

    model.fit(X_train, y_train)

    return model

# =========================================
# HYPERPARAMETERS UI
# =========================================

with st.expander("🎛️ XGBoost Model Hyperparameters", expanded=False):

    st.markdown("""
### 🧠 Hyperparameters Explanation

You can edit the first three parameters using the sliders below. The last three are fixed.

**Editable:**
- **n_estimators** → Number of trees in the model. More trees = potentially higher accuracy but slower training.
- **max_depth** → Maximum depth of each tree. Higher depth = more complex model, risk of overfitting.
- **learning_rate** → Step size for boosting. Smaller = slower but more stable, larger = faster but may be unstable.

**Fixed (not editable in the app):**
- **subsample = 0.8** → Fraction of samples used per tree to reduce overfitting.
- **colsample_bytree = 0.8** → Fraction of features used per tree to reduce correlation between trees.
- **eval_metric = 'mlogloss'** → Evaluation metric used during training (multi-class log loss).
""")

    n_estimators = st.slider(
        "n_estimators",
        min_value=50,
        max_value=500,
        value=150
    )

    max_depth = st.slider(
        "max_depth",
        min_value=2,
        max_value=15,
        value=6
    )

    learning_rate = st.slider(
        "learning_rate",
        min_value=0.01,
        max_value=1.0,
        value=0.1
    )

# =========================================
# SESSION STATE
# =========================================

if "results" not in st.session_state:
    st.session_state.results = None

# =========================================
# TRAIN BUTTON
# =========================================

train_button = st.button(
    "🚀 Train Model (optimized)"
)

if train_button:

    with st.spinner("⏳ Training model... please wait"):

        X_train, X_test, y_train, y_test = get_split(X, y)

        model = train_model(
            X_train,
            y_train,
            n_estimators,
            max_depth,
            learning_rate
        )

        y_pred = model.predict(X_test)

        st.session_state.results = {
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred
        }

    st.success("Model trained successfully")

# =========================================
# 🌲 PART 3 - EVALUATION METRICS
# =========================================

if st.session_state.results is not None:

    results = st.session_state.results

    y_test = results["y_test"]
    y_pred = results["y_pred"]

    # =========================================
    # 🎯 ACCURACY
    # =========================================

    with st.expander("🎯 Accuracy", expanded=False):

        acc = (y_test == y_pred).mean()

        st.metric(
            "Accuracy",
            f"{acc:.4f}"
        )

    # =========================================
    # 📄 CLASSIFICATION REPORT
    # =========================================

    with st.expander("📄 Classification Report", expanded=False):

        from sklearn.metrics import classification_report

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df, use_container_width=True)

        st.markdown("""
### 🧠 What does this mean?

- **Precision** → Of all the times the model predicted a class, how many were correct.
- **Recall** → Of all the times that class actually occurred, how many did the model detect.
- **F1-score** → Balance between precision and recall.

### 📌 Quick interpretation:
- ✔ If precision is low → the model makes many mistakes when predicting that class.
- ✔ If recall is low → the model misses real cases of that class.
- ✔ If F1-score is low → the model is not balanced for that class.
""")

    # =========================================
    # 🧩 CONFUSION MATRIX (PLOTLY)
    # =========================================

    with st.expander("🧩 Confusion Matrix", expanded=False):

        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_test, y_pred)

        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual {i+1}" for i in range(cm.shape[0])],
            columns=[f"Predicted {i+1}" for i in range(cm.shape[1])]
        )

        st.dataframe(cm_df, use_container_width=True)

        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Confusion Matrix"
        )

        fig.update_layout(height=500)

        st.plotly_chart(fig, use_container_width=True)

    # =========================================
    # 📌 FEATURE IMPORTANCE
    # =========================================

    with st.expander("📌 Feature Importance", expanded=False):

        model = results["model"]

        importances = model.feature_importances_

        top_idx = np.argsort(importances)[-10:][::-1]

        feature_names = df.drop(columns=["Cover_Type"]).columns

        imp_df = pd.DataFrame({
            "Feature": feature_names[top_idx],
            "Importance": importances[top_idx]
        })

        st.dataframe(imp_df, use_container_width=True)

        fig = px.bar(
            imp_df,
            x="Feature",
            y="Importance",
            text="Importance",
            title="Top 10 Feature Importances"
        )

        fig.update_layout(
            height=500,
            xaxis_tickangle=45
        )

        st.plotly_chart(fig, use_container_width=True)

    # =========================================
    # 🧪 SAMPLE PREDICTIONS
    # =========================================

    with st.expander("🧪 Sample Predictions (First 100)", expanded=False):

        X_test = results["X_test"]

        y_test = results["y_test"]

        y_pred_sample = y_pred[:100]
        y_test_sample = y_test[:100]

        sample_df = pd.DataFrame({
            "Predicted Class": y_pred_sample + 1,
            "Real Class": y_test_sample + 1,
            "Correct?": (y_pred_sample == y_test_sample)
        })

        st.dataframe(sample_df, use_container_width=True)

        accuracy_100 = (sample_df["Correct?"].mean()) * 100

        st.write(f"🎯 Accuracy on 100 samples: {accuracy_100:.2f}%")

# =========================================
# 🌲 PART 4 - FINAL OPTIMIZATION / CLEANUP
# =========================================

# =========================================
# OPTIONAL: FORCE GARBAGE CLEANUP
# =========================================

import gc

gc.collect()

# =========================================
# OPTIONAL INFO FOOTER
# =========================================

st.markdown("---")

st.markdown("""
### 🚀 Optimized Streamlit App Notes

- Uses **float32** to reduce memory usage.
- XGBoost uses **tree_method='hist'** for faster training.
- Parallel processing enabled with **n_jobs=-1**.
- Heavy computations cached using **st.cache_data / st.cache_resource**.
- Visualizations powered by **Plotly (faster than Matplotlib)**.
- Results stored in **session_state** to avoid recomputation.

👉 This version is optimized for full dataset (581,012 rows).
""")

# =========================================
# OPTIONAL: RESET RESULTS BUTTON
# =========================================

if st.button("🔄 Reset Results"):

    st.session_state.results = None

    st.success("Results cleared. You can retrain the model.")