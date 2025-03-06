# 📌 Proyecto de Clasificación de Texto con TinyBERT y MLflow

Este proyecto implementa un sistema de clasificación de texto utilizando el modelo **TinyBERT**, con seguimiento completo de métricas y visualizaciones mediante **MLflow**.

---

## 📖 Descripción

El proyecto utiliza un modelo **TinyBERT** para clasificar textos del dataset **AG News** en cuatro categorías. Durante el entrenamiento, se hace un seguimiento detallado del rendimiento mediante diversas métricas (**accuracy, F1-score, precision, recall**) y se generan visualizaciones para facilitar el análisis de resultados.

---

## ✨ Características

✅ Utiliza el modelo pre-entrenado **TinyBERT** (`huawei-noah/TinyBERT_General_4L_312D`)
✅ Clasificación de textos en múltiples categorías
✅ Seguimiento de métricas con **MLflow**
✅ Generación automática de gráficas y visualizaciones 📊
✅ Matriz de confusión para análisis de errores 🧐
✅ Guardado del modelo entrenado 💾

---

## 📦 Requisitos

```bash
torch
transformers
datasets
scikit-learn
matplotlib
numpy
mlflow
seaborn
```

---

## 📂 Estructura del Proyecto

```
ProjectoMLFlow/
│
├── 📜 ModeloMLFlow.py         # Script principal con el código de entrenamiento
├── 📂 modelo_final/           # Directorio donde se guarda el modelo entrenado
├── 📂 results/                # Resultados y checkpoints del entrenamiento
├── 📂 graficas/               # Gráficas generadas durante el entrenamiento
│   ├── 📈 accuracy_evolution.png
│   ├── 📊 all_metrics_evolution.png
│   ├── 🎯 confusion_matrix.png
│   ├── 📊 f1_evolution.png
│   ├── 🏆 final_metrics.png
│   └── 📉 loss_function.png
├── 📂 logs/                   # Logs del entrenamiento
└── 📜 README.md               # Este archivo
```

---

## 🛠 Instalación

1️⃣ Clona el repositorio:
   ```bash
   git clone https://github.com/yourusername/Proyecto_MLFlow.git
   cd Proyecto_MLFlow
   ```

2️⃣ Instala las dependencias:
   ```bash
   pip install torch transformers datasets scikit-learn matplotlib numpy mlflow seaborn
   ```

---

## 🚀 Uso

Para ejecutar el entrenamiento completo:

```bash
python ModeloMLFlow.py
```

---

## 📚 Conjunto de Datos

El proyecto utiliza el dataset **AG News** de **Hugging Face**, que contiene textos clasificados en cuatro categorías:
- 🌍 **World**
- 🏆 **Sports**
- 💰 **Business**
- 🔬 **Sci/Tech**

Para el entrenamiento se utiliza una muestra reducida de **1000 ejemplos** para **training** y **500 para validación**.

---

## 📊 Seguimiento con MLflow

El proyecto utiliza **MLflow** para seguimiento de experimentos. Se registran:

- **📌 Parámetros**: nombre del modelo, longitud máxima del texto, tamaño de muestras, etc.
- **📈 Métricas**: loss, accuracy, F1-score, precision, recall
- **📂 Artefactos**: gráficas generadas durante el entrenamiento

Para ver los resultados en la UI de MLflow:

```bash
mlflow ui
```

---

## 🔍 Visualizaciones

El proyecto genera automáticamente las siguientes visualizaciones:

1️⃣ **📉 Evolución de la función de pérdida**: Muestra cómo cambia la pérdida durante el entrenamiento
2️⃣ **📈 Evolución de accuracy**: Seguimiento de la precisión durante las épocas de entrenamiento
3️⃣ **📊 Evolución de F1-Score**: Seguimiento del F1-Score durante el entrenamiento
4️⃣ **🏆 Métricas finales**: Gráfico de barras con las métricas finales del modelo
5️⃣ **🎯 Matriz de confusión**: Análisis detallado de los aciertos y errores del modelo por clase
6️⃣ **📊 Evolución de todas las métricas**: Gráfico combinado que muestra la evolución de accuracy, F1, precision y recall

---

## 🔧 Callback Personalizado

El proyecto implementa un **callback personalizado** (`MetricsCallback`) para registrar métricas detalladas durante el entrenamiento y almacenarlas en **MLflow**.

---

## ⚠️ Limitaciones

⚠️ Se utiliza una muestra reducida del dataset para agilizar el entrenamiento.
⚠️ Algunos archivos grandes (como checkpoints y modelos) pueden requerir **Git LFS** para su gestión en el repositorio.

---


