# 🌲 Forest Cover Type Prediction via XGBoost

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square\&logo=python\&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-v2.0-FF6F20?style=flat-square\&logo=xgboost\&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square\&logo=scikit-learn\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square\&logo=pandas\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square\&logo=streamlit\&logoColor=white)

An end-to-end Machine Learning classification framework designed to predict dominant forest cover types across heavily imbalanced environments. Utilizing **581,012 spatial samples** from the **US Forest Service (USFS)**, this project eliminates the need for expensive physical site surveys by mapping geographical, topographic, and geological data directly into ecological forest categories.

The repository includes:

* **Original Python Pipeline (`main.py`)**
* **Interactive Streamlit Application (`app.py`) in the next link: https://forestcovertypepredictionviaxgboost.streamlit.app/**

Both implementations use the same Forest Cover Type dataset and XGBoost classification framework, while the Streamlit version adds interactive exploration, hyperparameter tuning, visualization, and model analysis capabilities.

---

# 🎯 Project Overview & Objectives

Traditional ecological surveys are slow and resource-intensive. This project leverages advanced gradient boosting to automate land-use mapping.

## Key Objectives

* Predict dominant forest cover type from geographical and ecological variables.
* Analyze the influence of terrain, hydrology, wilderness areas, and soil composition.
* Evaluate model performance using rigorous train/test separation.
* Provide an interactive interface for experimentation and educational purposes.

## Original Script vs Streamlit App

### Original Script (`main.py`)

* Fixed training pipeline
* Fixed hyperparameters
* Terminal-based execution
* Static evaluation outputs

### Streamlit Application (`app.py`)

* Interactive model training
* Hyperparameter adjustment
* Interactive dataset exploration
* Downloadable full dataset
* Interactive model interpretation tools
* Dynamic evaluation reports

---

# 🧭 Dataset Architecture

The model ingests **54 cartographic variables** (all numerical or encoded features).

## ⛰️ Topographic Profile

- 0: Elevation ➜ Height above sea level (meters)  
- 1: Aspect ➜ Slope direction (0 to 360 grades) 
- 2: Slope ➜ Terrain inclination (0 to 90 grades)


## 🛣️ Spatial Distances

- 3: Horizontal_Distance_To_Hydrology ➜ Distance to nearest surface water 
- 4: Vertical_Distance_To_Hydrology ➜ Vertical distance to water source  
- 5: Horizontal_Distance_To_Roadways ➜ Distance to nearest road 

## ☀️ Solar Exposure

- 6: Hillshade_9am ➜ Solar radiation at 9 AM  
- 7: Hillshade_Noon ➜ Solar radiation at noon  
- 8: Hillshade_3pm ➜ Solar radiation at 3 PM  

## 🔥 Fire History

- Horizontal Distance to Fire Points

## 🌲 Wilderness Areas

Binary ecological region indicators:

- 10: Wilderness_Area_1 ➜ Rawah Wilderness Area 
- 11: Wilderness_Area_2 ➜ Neota Wilderness Area  
- 12: Wilderness_Area_3 ➜ Comanche Peak Wilderness Area 
- 13: Wilderness_Area_4 ➜ Cache la Poudre Wilderness Area  

## 🌱 Soil Types

40 ecological soil classifications defined by the US Forest Service.

These variables capture:

* Soil composition
* Moisture retention
* Geological origin
* Drainage characteristics
* Ecological suitability

- 14: Soil_Type1 → Very rocky, shallow soil with high drainage (mountainous areas)
- 15: Soil_Type2 → Volcanic or igneous soil with moderate drainage
- 16: Soil_Type3 → Alluvial soil (river sediments)
- 17: Soil_Type4 → Shallow, stony soil with low water retention
- 18: Soil_Type5 → Deeper soil with higher moisture content (valley areas)
- 19: Soil_Type6 → Soil with surface organic matter
- 20: Soil_Type7 → Extremely rocky soil
- 21: Soil_Type8 → Mixed rock and sediment soil
- 22: Soil_Type9 → Moist soil with slow drainage
- 23: Soil_Type10 → Mountain-valley transition soil
- 24: Soil_Type11 → Eroded mountain soil
- 25: Soil_Type12 → Compacted soil with low infiltration
- 26: Soil_Type13 → Moderately fertile mineral soil
- 27: Soil_Type14 → Dry soil with sparse vegetation
- 28: Soil_Type15 → Deep soil with nutrients
- 29: Soil_Type16 → Mixed rock-organic soil
- 30: Soil_Type17 → Cold high-mountain soil
- 31: Soil_Type18 → Soil eroded by wind/water
- 32: Soil_Type19 → Moist soil with biological activity
- 33: Soil_Type20 → Ecological transition soil
- 34: Soil_Type21 → Highly compacted soil
- 35: Soil_Type22 → Glacial soil
- 36: Soil_Type23 → Fine sedimentary soil
- 37: Soil_Type24 → Very stony soil
- 38: Soil_Type25 → Deep and stable soil
- 39: Soil_Type26 → Soil with rapid drainage
- 40: Soil_Type27 → Moist soil in low-lying areas
- 41: Soil_Type28 → Nutrient-poor soil
- 42: Soil_Type29 → Fertile transition soil
- 43: Soil_Type30 → Variable mixed soil
- 44: Soil_Type31 → Soil with additional complex characteristics
- 45: Soil_Type32 → Soil with mixed geological variations
- 46: Soil_Type33 → Soil of heterogeneous composition
- 47: Soil_Type34 → Soil with variable structure
- 48: Soil_Type35 → Soil with moisture variation
- 49: Soil_Type36 → Specific mineral soil
- 50: Soil_Type37 → Eroded or sloping soil
- 51: Soil_Type38 → High-mountain soil (Leighcan-Catamount Complex, very rocky)
- 52: Soil_Type39 → Advanced ecological transition soil (Leighcan-Moran Complex, deep glacial)
- 53: Soil_Type40 → Last category of the system (Moran-Cryaquolls Complex, humid high mountain)

## 🎯 Target Variable

**Cover_Type (1–7)**

| Class | Forest Type                                                                  |
| ----- | ---------------------------------------------------------------------------- |
| 1     | Spruce/Fir → Fir and pine forests in cold climates and high mountain regions |
| 2     | Lodgepole Pine → Pine forests adapted to poor soils and cold climates        |
| 3     | Ponderosa Pine → Pine forests in drier, temperate zones                      |
| 4     | Cottonwood/Willow → Forests near rivers and wetlands                         |
| 5     | Aspen → Poplar forests, typical of regenerated mountainous areas             |
| 6     | Douglas-fir → Douglas fir forests, common in temperate zones                 |
| 7     | Krummholz → High-mountain vegetation, trees deformed by extreme weather      |

---

# ⚙️ Methodology

## 1. Data Loading

The project uses:

```python
fetch_covtype()
```

from Scikit-Learn.

Dataset size:

* 581,012 observations
* 54 predictive variables
* 7 target classes

---

## 2. Train/Test Split

Both implementations use:

* 80% Training Set
* 20% Test Set

with:

```python
stratify=y
```

to preserve class distribution.

---

## 3. Model

The classifier is:

```python
XGBClassifier
```

Default baseline configuration:

```python
n_estimators = 150
max_depth = 6
learning_rate = 0.1
subsample = 0.8
colsample_bytree = 0.8
eval_metric = "mlogloss"
```

### Original Script

Uses the fixed configuration above.

### Streamlit App

Allows interactive modification of:

* n_estimators
* max_depth
* learning_rate

before training.

---

# 🚀 Model Performance & Key Results

The core XGBoost classifier achieved a powerful **83.08% Global Accuracy** score.

## 🔧 Note on Results Interpretation

The performance metrics presented below correspond to the **optimized hyperparameter configuration defined in the original training script (`main.py`)**.

These results represent the output of a fixed and reproducible training configuration used during the initial experimentation phase of the project.

The Streamlit application allows users to modify hyperparameters interactively and obtain different results. Therefore, performance metrics generated inside the application may differ from the baseline values reported in this document depending on the selected configuration.

---

## Granular Metrics Report

| Class Name (Cover_Type) | Precision | Recall | F1-Score | Support |
| ----------------------- | --------- | ------ | -------- | ------- |
| 1: Spruce/Fir           | 0.82      | 0.80   | 0.81     | 42,368  |
| 2: Lodgepole Pine       | 0.83      | 0.87   | 0.85     | 56,661  |
| 3: Ponderosa Pine       | 0.84      | 0.88   | 0.86     | 7,151   |
| 4: Cottonwood/Willow    | 0.85      | 0.82   | 0.83     | 549     |
| 5: Aspen                | 0.86      | 0.43   | 0.57     | 1,899   |
| 6: Douglas-fir          | 0.80      | 0.68   | 0.74     | 3,473   |
| 7: Krummholz            | 0.92      | 0.86   | 0.89     | 4,102   |

---

# 📄 Classification Report Interpretation

### Precision

Of all samples predicted as a given class, how many were correct.

### Recall

Of all real samples belonging to a class, how many were correctly detected.

### F1-Score

Balance between Precision and Recall.

### Quick Interpretation

* Low Precision → many false positives.
* Low Recall → many missed cases.
* Low F1-score → poor balance between Precision and Recall.

---

# 🧩 Confusion Matrix Interpretation

The confusion matrix compares:

* Rows → Actual classes
* Columns → Predicted classes

### Reading the Matrix

* Values on the main diagonal are correct predictions.
* Values outside the diagonal are classification errors.

### Why It Matters

The confusion matrix reveals:

* Which classes are easiest to predict.
* Which classes are frequently confused.
* Potential bias toward majority classes.
* Weaknesses in rare ecological categories.

---

# 🧠 Strategic Insights from the Confusion Matrix

### 1. Dominant Class Stability

The model demonstrates strong robustness for Spruce/Fir and Lodgepole Pine.

### 2. High-Altitude Precision

Krummholz achieved exceptionally high precision.

### 3. Ecological Overlap Challenge

Aspen exhibits low recall due to ecological similarity with Lodgepole Pine environments.

---

# 📊 Feature Importance

The model successfully identified meaningful ecological patterns.

Top influential variables include:

🥇 Wilderness_Area_4 (Cache la Poudre)

🥈 Elevation

🥉 Soil_Type39 (Leighcan-Moran)

🏅 Soil_Type38 (Leighcan-Catamount)

🏅 Soil_Type40 (Moran-Cryaquolls)

The Streamlit application additionally provides:

* Interactive feature importance tables
* Top-10 ranking visualization
* Automatic graphical interpretation

---

# 🌐 Streamlit Web Application

The repository includes a complete interactive Streamlit dashboard.

## Features

### 📋 Dataset Preview

* First 50 records displayed directly in the interface
* Full dataset download button

### 📘 Dataset Documentation

Detailed explanation of:

* All 54 variables
* Wilderness Areas
* Soil Types
* Cover Type categories

### 📊 Dataset Metrics

Displays:

* Number of rows
* Number of features
* Number of classes

### 📈 Class Distribution

Interactive visualization of class balance.

### 🎛️ Hyperparameter Configuration

Adjust:

* n_estimators
* max_depth
* learning_rate

before model training.

### 🎯 Accuracy Evaluation

Real-time accuracy reporting.

### 📄 Classification Report

Interactive report visualization.

### 🧩 Confusion Matrix

Displayed as an exportable table rather than an image.

### 📌 Feature Importance

Interactive ranking and chart generation.

### 🧪 Sample Predictions

Shows:

* Predicted class
* Actual class
* Correct / Incorrect flag
* Accuracy over displayed samples

### 🔴 Live Application

Access the deployed Streamlit application here:

https://forestcovertypepredictionviaxgboost.streamlit.app/

---

# 🚀 How to Run

## Original Script

```bash
python main.py
```

---

## Streamlit Application

```bash
streamlit run app.py
```

To stop execution:

```bash
Ctrl + C
```

---

# 📁 Project Structure

```text
├── app.py
├── main.py
├── README.md
├── requirements.txt
└── images/
```

---

# 💡 Key Learnings

* Gradient Boosting for multiclass classification
* Ecological land-cover prediction
* Handling highly imbalanced classes
* Train/test validation strategies
* Feature importance analysis
* Model interpretability
* Interactive Machine Learning deployment with Streamlit

---

# 🌱 Future Improvements

* Hyperparameter optimization (GridSearchCV / Optuna)
* Cross-validation experiments
* SHAP explainability integration
* Interactive prediction form for custom inputs
* Model persistence and deployment
* Cloud-hosted Streamlit dashboard
* Advanced ecological interpretation modules
