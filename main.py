# =========================================
# 🌲 PROYECTO: FOREST COVER TYPE CON XGBOOST
# =========================================

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier

import os
import warnings

warnings.filterwarnings("ignore")
os.system('cls' if os.name == 'nt' else 'clear')

# =========================================
# 🟢 1. CARGAR DATASET
# =========================================
print("📥 Cargando dataset Forest Cover Type...")

data = fetch_covtype()

X = data.data
y = data.target - 1  # clases 0–6 para XGBoost

print("✅ Dataset cargado")
print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================================
# 🟢 2. GUARDAR DATASET EN CSV
# =========================================
print("\n💾 Guardando dataset en CSV...")

df = pd.DataFrame(X)
df["Cover_Type"] = y + 1

path = r"G:\Mi unidad\Python\prediccion_tipo_cobertura_forestal\forest_cover_dataset.csv"
df.to_csv(path, index=False)

print("✅ Dataset guardado en:")
print(path)
print(df)
# =========================================
# 🟢 3. EXPLICACIÓN GENERAL
# =========================================
print("\n📘 EXPLICACIÓN GENERAL DEL DATASET\n")

print("🌲 Dataset forestal con variables geográficas y ecológicas.")
print("Objetivo: predecir tipo de cobertura forestal.\n")

print("📌 VARIABLES 0–2 (Topografía básica):")
print("0: Elevation → Altura sobre el nivel del mar (en metros)")
print("1: Aspect → Dirección de la pendiente (orientación en grados de 0 a 360)")
print("2: Slope → Inclinación del terreno (en grados de 0 a 90)\n")

print("📌 VARIABLES 3–5 (Distancias e Infraestructura):")
print("3: Horizontal_Distance_To_Hydrology → Distancia horizontal al agua superficial más cercana")
print("4: Vertical_Distance_To_Hydrology → Diferencia vertical respecto al agua superficial más cercana")
print("5: Horizontal_Distance_To_Roadways → Distancia horizontal al camino o carretera más cercana\n")

print("📌 VARIABLES 6–8 (Iluminación Solar / Hillshade):")
print("6: Hillshade_9am → Índice de sombra/iluminación solar a las 9:00 a.m. (Solsticio de verano)")
print("7: Hillshade_Noon → Índice de sombra/iluminación solar al mediodía (Solsticio de verano)")
print("8: Hillshade_3pm → Índice de sombra/iluminación solar a las 3:00 p.m. (Solsticio de verano)\n")

print("📌 VARIABLE 9 (Dinámica de Incendios):")
print("9: Horizontal_Distance_To_Fire_Points → Distancia horizontal al origen de incendios históricos más cercano\n")

# =========================================
# 🌲 WILDERNESS AREAS
# =========================================
print("📌 VARIABLES 10–13 (Wilderness Areas / Regiones Ecológicas):")
print("10: Wilderness_Area_1 → Rawah Wilderness Area (Área ecológica 1)")
print("11: Wilderness_Area_2 → Neota Wilderness Area (Área ecológica 2)")
print("12: Wilderness_Area_3 → Comanche Peak Wilderness Area (Área ecológica 3)")
print("13: Wilderness_Area_4 → Cache la Poudre Wilderness Area (Área ecológica 4)\n")

print("⚠️ Son regiones ecológicas del estudio (anonimizadas y codificadas en binario).\n")

# =========================================
# 🌱 SOIL TYPES (1 POR 1 — SIN AGRUPAR)
# =========================================
print("🌱 INTERPRETACIÓN DE TIPOS DE SUELO (Soil_Type1 a Soil_Type40)\n")

print("⚠️ Nota importante:")
print("Clasificación ecológica del US Forest Service.")
print("No son 'arena' o 'arcilla' simples, sino combinaciones complejas de propiedades del suelo.\n")

print("📌 SOIL TYPES (Índices del 14 al 53):\n")

print("14: Soil_Type1 → Suelo muy rocoso, poco profundo, drenaje alto (zonas montañosas)")
print("15: Soil_Type2 → Suelo volcánico o ígneo, drenaje moderado")
print("16: Soil_Type3 → Suelo aluvial (sedimentos de ríos) [¡Variable de alto peso en XGBoost!]")
print("17: Soil_Type4 → Suelo poco profundo, pedregoso, baja retención de agua")
print("18: Soil_Type5 → Suelo más profundo, mayor humedad (zonas de valle)")
print("19: Soil_Type6 → Suelo con materia orgánica superficial")
print("20: Soil_Type7 → Suelo extremadamente rocoso")
print("21: Soil_Type8 → Suelo mixto de roca y sedimentos")
print("22: Soil_Type9 → Suelo húmedo con drenaje lento")
print("23: Soil_Type10 → Suelo de transición montaña-valle")

print("24: Soil_Type11 → Suelo de montaña erosionado")
print("25: Soil_Type12 → Suelo compacto con baja infiltración")
print("26: Soil_Type13 → Suelo mineral moderadamente fértil")
print("27: Soil_Type14 → Suelo seco con poca vegetación")
print("28: Soil_Type15 → Suelo profundo con nutrientes")
print("29: Soil_Type16 → Suelo mixto roca-orgánico")
print("30: Soil_Type17 → Suelo frío de alta montaña")
print("31: Soil_Type18 → Suelo erosionado por viento/agua")
print("32: Soil_Type19 → Suelo húmedo con actividad biológica")
print("33: Soil_Type20 → Suelo de transición ecológica")

print("34: Soil_Type21 → Suelo altamente compactado")
print("35: Soil_Type22 → Suelo de origen glaciar")
print("36: Soil_Type23 → Suelo sedimentario fino")
print("37: Soil_Type24 → Suelo muy pedregoso")
print("38: Soil_Type25 → Suelo profundo y estable")
print("39: Soil_Type26 → Suelo con drenaje rápido")
print("40: Soil_Type27 → Suelo húmedo de zonas bajas")
print("41: Soil_Type28 → Suelo pobre en nutrientes")
print("42: Soil_Type29 → Suelo fértil de transición")
print("43: Soil_Type30 → Suelo mixto variable")

print("44: Soil_Type31 → Suelo con características complejas adicionales")
print("45: Soil_Type32 → Suelo con variaciones geológicas mixtas")
print("46: Soil_Type33 → Suelo de composición heterogénea")
print("47: Soil_Type34 → Suelo con estructura variable")
print("48: Soil_Type35 → Suelo con variación de humedad")
print("49: Soil_Type36 → Suelo mineral específico")
print("50: Soil_Type37 → Suelo erosionado o en pendiente")
print("51: Soil_Type38 → Suelo de alta montaña (Complejo Leighcan - Catamount, muy rocoso)")
print("52: Soil_Type39 → Suelo de transición ecológica avanzada (Complejo Leighcan - Moran, profundo glaciar)")
print("53: Soil_Type40 → Última categoría del sistema (Complejo Moran - Cryaquolls, alta montaña húmedo)\n")

print("Cada Soil_Type representa una categoría ecológica del US Forest Service.")
print("Al estar vectorizadas, interactúan directamente con el algoritmo para identificar nichos biológicos.\n")

# =========================================
# Variable objetivo: Cover_Type
# =========================================

print("📌 VARIABLE OBJETIVO:")
print("Cover_Type → tipo de bosque (1 a 7)\n")

print("📌 INTERPRETACIÓN DE COVER_TYPE:\n")

print("Cover_Type es la variable que el modelo intenta predecir.")
print("Representa el tipo de bosque dominante en una ubicación geográfica específica.\n")

print("Cada valor significa:\n")

print("1: Spruce/Fir → Bosques de abeto y pino de climas fríos y alta montaña")
print("2: Lodgepole Pine → Bosques de pino resistentes a suelos pobres y climas fríos")
print("3: Ponderosa Pine → Bosques de pino en zonas más secas y templadas")
print("4: Cottonwood/Willow → Bosques cercanos a ríos y zonas húmedas")
print("5: Aspen → Bosques de álamos, típicos de zonas montañosas regeneradas")
print("6: Douglas-fir → Bosques de abeto Douglas, comunes en zonas templadas")
print("7: Krummholz → Vegetación en alta montaña, árboles deformados por el clima extremo\n")

print("🧠 RESUMEN:")
print("Cover_Type clasifica el tipo de ecosistema forestal basado en condiciones del terreno, suelo y ubicación.")

print("⚠️ Estas variables no tienen significado ecológico individual publicado.")
print("Se usan como parte del vector numérico del modelo.\n")

# =========================================
# 🎯 TARGET
# =========================================
print("📌 VARIABLE OBJETIVO:")
print("Cover_Type → tipo de bosque (1 a 7)\n")

# =========================================
# 🟢 4. DISTRIBUCIÓN
# =========================================
print("📊 DISTRIBUCIÓN DE CLASES")

unique, counts = np.unique(y, return_counts=True)

for u, c in zip(unique, counts):
    print(f"Clase {u+1}: {c}")

# =========================================
# 🟢 5. SPLIT
# =========================================
print("\n✂️ Dividiendo datos...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# =========================================
# 🟢 6. MODELO
# =========================================
print("\n🚀 Entrenando XGBoost...")

model = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

print("✅ Modelo entrenado")

# =========================================
# 🟢 7. PREDICCIÓN
# =========================================
print("\n🔮 Prediciendo...")

y_pred = model.predict(X_test)

# =========================================
# 🟢 8. EVALUACIÓN
# =========================================
print("\n📊 RESULTADOS")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\n📄 CLASSIFICATION REPORT (INTERPRETACIÓN)\n")

report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

print("🧠 ¿Qué significa esto?\n")

print("📌 Precision → De todas las veces que el modelo dijo una clase, cuántas fueron correctas.")
print("📌 Recall → De todas las veces que esa clase era real, cuántas detectó el modelo.")
print("📌 F1-score → Balance entre precision y recall.")

print("\n📌 Interpretación rápida:")
print("✔ Si precision es baja → el modelo se equivoca mucho cuando predice esa clase")
print("✔ Si recall es bajo → el modelo se le escapan casos reales de esa clase")
print("✔ F1 bajo → el modelo no está equilibrado para esa clase")

print("\n🧩 CONFUSION MATRIX (INTERPRETACIÓN)\n")

cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n🧠 ¿Cómo leerla?\n")

print("Las filas = valores reales")
print("Las columnas = predicciones del modelo\n")

print("📌 Ejemplo:")
print("Si ves [ [5000  100],")
print("           [  80  900] ]")

print("→ 5000 normales fueron bien clasificados")
print("→ 100 normales fueron confundidos como fraude")
print("→ 80 fraudes no fueron detectados (ERROR CRÍTICO)")
print("→ 900 fraudes fueron detectados correctamente\n")

print("⚠️ En problemas de fraude o bosque, los falsos negativos suelen ser lo más importante.")

# =========================================
# 🟢 9. IMPORTANCIA
# =========================================
print("\n📌 VARIABLES MÁS IMPORTANTES")

importances = model.feature_importances_
top_idx = np.argsort(importances)[-10:]

for i in top_idx:
    print(f"Feature {i} -> {importances[i]:.4f}")

# =========================================
# 🟢 10. PRUEBA FINAL
# =========================================
print("\n🧪 PRUEBA")

sample = X_test[0].reshape(1, -1)
pred = model.predict(sample)

cover_types = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

print("Predicción:", pred[0] + 1)
print("Nombre:", cover_types[pred[0] + 1])
print("Real:", y_test[0] + 1)