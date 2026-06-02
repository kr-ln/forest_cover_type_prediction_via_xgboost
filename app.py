# =========================================
# 🌲 FOREST COVER TYPE - STREAMLIT APP
# =========================================

import streamlit as st
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode

# =========================================
# FUNCTIONS
# =========================================

@st.cache_data
def get_split(X, y):
    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

@st.cache_resource
def train_model(X_train, y_train, n_estimators, max_depth, learning_rate):

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss"
    )

    model.fit(X_train, y_train)

    return model

# =========================================
# CONFIG
# =========================================

st.set_page_config(
    page_title="Forest Cover Type Prediction via XGBoost",
    page_icon="📊🌲",
    layout="wide"
)

st.title("🌲 Forest Cover Type Prediction via XGBoost")

# =========================================
# CACHE MANAGEMENT BUTTON
# =========================================
if st.button("🧹 Clear Streamlit Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("✅ Cache cleared! Reload the app to start fresh.")

# =========================================
# LOAD DATASET
# =========================================

@st.cache_data
def load_data():

    data = fetch_covtype()

    X = data.data
    y = data.target - 1

    df = pd.DataFrame(X)
    df["Cover_Type"] = y + 1

    return df


if "df" not in st.session_state:
    st.session_state.df = None

# =========================================
# MAIN APP
# =========================================

df = load_data()
X = df.drop(columns=["Cover_Type"]).values
y = df["Cover_Type"].values - 1

# =========================================
# 🧠 Summary
# =========================================

# 🧠 Summary
with st.expander("🧠 Summary", expanded=False):
    st.markdown("""
Forest type classification based on terrain, soil, hydrology, and ecological conditions.
""")

# =========================================
# 📋 TABLE
# =========================================

with st.expander("📋 Dataset Preview (First 50 rows)", expanded=False):
    df_preview = df.head(50)
    st.dataframe(df_preview, use_container_width=True)
    st.write(f"📊 Showing first {len(df_preview)} rows")

# =========================================
# 💾 DOWNLOAD BUTTON (FULL DATASET)
# =========================================

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="💾 Download Full Dataset",
    data=csv,
    file_name="forest_cover_dataset.csv",
    mime="text/csv"
)

st.markdown("---")

# =========================================
# 📘 DATASET EXPLANATION
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
# 📊 METRICS
# =========================================

with st.expander("📊 Dataset Metrics", expanded=False):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", f"{len(df):,}")

    with col2:
        st.metric("Features", f"{df.shape[1]-1}")

    with col3:
        st.metric("Classes", "7")

# =========================================
# 📈 CLASS DISTRIBUTION
# =========================================

with st.expander("📈 Class Distribution", expanded=False):

    unique, counts = np.unique(y, return_counts=True)

    dist_df = pd.DataFrame({
        "Class": unique + 1,
        "Count": counts
    })

    st.dataframe(dist_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=dist_df, x="Class", y="Count", ax=ax)
    st.pyplot(fig)

# =========================================
# 🚀 MODEL TRAINING
# =========================================

with st.expander("🎛️ XGBoost Model Hyperparameters", expanded=False):

        # =========================================
    # 📖 Hyperparameters Explanation
    # =========================================
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

    n_estimators = st.slider("n_estimators", 50, 500, 150)
    max_depth = st.slider("max_depth", 2, 15, 6)
    learning_rate = st.slider("learning_rate", 0.01, 1.0, 0.1)

if st.button("Train Model (Make sure you have adjusted the hyperparameters in the section above)"):
    with st.spinner("⏳ Training the model... This may take a few seconds, please wait."):
        
        X_train, X_test, y_train, y_test = get_split(X, y)

        model = train_model(
            X_train,
            y_train,
            n_estimators,
            max_depth,
            learning_rate
        )

        y_pred = model.predict(X_test)

        st.success("Model trained successfully")

        # =========================================
        # 🎯 ACCURACY
        # =========================================

        with st.expander("🎯 Accuracy", expanded=False):
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{acc:.4f}")

        # =========================================
        # 📄 CLASSIFICATION REPORT
        # =========================================

        with st.expander("📄 Classification Report", expanded=False):
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()

            st.dataframe(report_df, use_container_width=True)

            # 🧠 Interpretation
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
        # 🧩 CONFUSION MATRIX
        # =========================================

        with st.expander("🧩 Confusion Matrix (Table)", expanded=False):
            cm = confusion_matrix(y_test, y_pred)

            # Convert to DataFrame for Excel-style visualization
            cm_df = pd.DataFrame(
                cm,
                index=[f"Actual {i+1}" for i in range(cm.shape[0])],
                columns=[f"Predicted {i+1}" for i in range(cm.shape[1])]
            )

            st.dataframe(cm_df, use_container_width=True)

            # =========================================
            # 🧠 INTERPRETATION
            # =========================================

            st.markdown("""
            ### 🧠 How to interpret the confusion matrix?

            - **Rows → actual classes**
            - **Columns → predicted classes**

            ### 📌 Reading the matrix:

            - Values on the **main diagonal** are correct predictions.
            - Values outside the diagonal are classification errors.

            ### 📊 Example:

            |            | Pred 1 | Pred 2 |
            |------------|--------|--------|
            | Actual 1   | 5000   | 100    |
            | Actual 2   | 80     | 900    |

            This means:

            - ✔ 5000 samples correctly classified as class 1  
            - ✔ 900 samples correctly classified as class 2  
            - ❌ 100 class 1 samples misclassified as class 2  
            - ❌ 80 class 2 samples misclassified as class 1  

            ### ⚠️ Why this is important:

            - Errors outside the diagonal show where the model is confusing classes.
            - It is especially important to analyze minority classes for bias detection.

            ### 🧠 Advantage of this view:

            - Easy to copy and export
            - Excel-style format
            - Better for reporting and debugging
            ---
            """)

        # =========================================
        # 📌 FEATURE IMPORTANCE
        # =========================================
        with st.expander("📌 Feature Importance", expanded=False):

            # Obtener importancias del modelo
            importances = model.feature_importances_

            # Tomar los índices de las 10 features más importantes
            top_idx = np.argsort(importances)[-10:][::-1]

            # 🔹 Reemplazar índices por nombres reales de columnas
            feature_names = df.drop(columns=["Cover_Type"]).columns
            
            imp_df = pd.DataFrame({
                "Feature": feature_names[top_idx],
                "Importance": importances[top_idx]
            })

            imp_df["Feature"] = imp_df["Feature"].astype(str)

            # Mostrar tabla de importancias
            st.dataframe(imp_df, use_container_width=True)

            # Graficar las importancias
            fig, ax = plt.subplots(figsize=(12, 5))

            ax.bar(
                imp_df["Feature"],       # ya convertido a texto
                imp_df["Importance"],
                color="skyblue"
            )

            ax.set_title("Top 10 Feature Importances")
            ax.set_xlabel("Feature (categórica)")
            ax.set_ylabel("Importance")

            plt.xticks(rotation=45, ha='right')  # etiquetas legibles
            plt.tight_layout()
            st.pyplot(fig)

        # =========================================
        # 🧪 SAMPLE PREDICTION
        # =========================================

        with st.expander("🧪 Sample Predictions (First 100 test examples)", expanded=False):

            # Limit to first 100 examples or all if less than 100
            n_samples = min(100, len(X_test))
            X_sample = X_test[:n_samples]
            y_sample = y_test[:n_samples]

            # Make predictions
            y_pred_sample = model.predict(X_sample)

            # Prepare DataFrame to show
            sample_df = pd.DataFrame({
                "Predicted Class": y_pred_sample + 1,  # convert back to 1-7
                "Real Class": y_sample + 1
            })

            # ✅ Add a column "Correct?" to indicate if prediction was right
            sample_df["Correct?"] = sample_df["Predicted Class"] == sample_df["Real Class"]

            # Display the table
            st.dataframe(sample_df, use_container_width=True)

            # Calculate accuracy percentage for these 100 samples
            accuracy_percent = sample_df["Correct?"].mean() * 100

            # Show it below the table
            st.write(f"🎯 Accuracy on these {n_samples} examples: {accuracy_percent:.2f}%")
