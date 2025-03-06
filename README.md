# Especializacion en IA UAO

## Grupo 3

## Integrantes: 

- Pablo Andrés Muñoz Martínez  -   **Código**: 2244676

- Leidy Yasmin Hoyos Parra - **Código**: 2245224 
                
- Johan David Mendoza Vargas  - **Código**: 2245019
                
- Yineth Tatiana Hernández Narvaez  -  **Código**: 2244789 
                
---

# 📌 Proyecto de Monitoreo en ML - IA con TinyBERT, BERT-Mini y DistilBERT con MLflow

Este proyecto implementa un modelo de clasificación de texto utilizando **TinyBERT**, **BERT-Mini** y **DistilBERT**, con seguimiento completo de métricas y visualizaciones mediante **MLflow**.

---

## 📖 Introducción

Este proyecto utiliza modelos de la familia BERT para clasificar textos del dataset **AG News** en cuatro categorías. Durante el entrenamiento, se monitorea el rendimiento mediante métricas clave como **accuracy, F1-score, precision y recall**, además de generar visualizaciones para facilitar el análisis de resultados.

El objetivo es comparar el rendimiento de estos modelos y determinar cuál ofrece la mejor precisión y generalización.

---

## 🎯 Justificación

La clasificación de texto es esencial en aplicaciones como la organización automática de noticias. Se ha elegido **AG News** por su relevancia y tamaño, mientras que los modelos de la familia **BERT** permiten balancear rendimiento y eficiencia computacional. **MLflow** facilita el monitoreo del entrenamiento y la visualización de métricas para detectar problemas como el sobreajuste.

---

## 🎯 Objetivos

1️⃣ **Entrenar** y comparar **TinyBERT**, **BERT-Mini** y **DistilBERT** en la tarea de clasificación de texto.  

2️⃣ **Monitorear** el entrenamiento mediante gráficas de la función de costo y métricas de desempeño.  

3️⃣ **Evaluar** el rendimiento de cada modelo con métricas específicas.  

4️⃣ **Utilizar MLflow** para registrar y visualizar los resultados del experimento.  

5️⃣ **Presentar un análisis comparativo** de los modelos.  

---

## 📂 Dataset: AG News

El dataset **AG News** contiene noticias categorizadas en cuatro clases:

- 🌍 **Mundo** (Clase 0)
- 🏆 **Deportes** (Clase 1)
- 💰 **Negocios** (Clase 2)
- 🔬 **Ciencia/Tecnología** (Clase 3)

Cada instancia consta de un título y una descripción de la noticia, junto con su etiqueta correspondiente. Se ha reducido el tamaño a **1000 muestras para entrenamiento** y **500 para test**.

---

## 🤖 Modelos

Se entrenaron y compararon los siguientes modelos:

- **TinyBERT** 🏋️‍♂️: Versión compacta y eficiente de BERT.

- **BERT-Mini** 📏: Un modelo con una estructura reducida de BERT.

- **DistilBERT** 🚀: Modelo liviano que retiene el 97% del rendimiento de BERT con solo el 60% de los parámetros.

---

## 📊 Análisis de Cada Modelo

### **TinyBERT**

#### **Evolución de Accuracy**
![Evolución de Accuracy TinyBERT](Proyecto_ml/graficas/tinybert_accuracy_evolution.png)

#### **Matriz de Confusión**
![Matriz de Confusión TinyBERT](Proyecto_ml/graficas/tinybert_confusion_matrix.png)

#### **Evolución de F1-Score**
![F1-Score TinyBERT](Proyecto_ml/graficas/tinybert_f1_evolution.png)

#### **Función de Costo**
![Pérdida TinyBERT](Proyecto_ml/graficas/tinybert_loss_function.png)

---

### **BERT-Mini**

#### **Evolución de Accuracy**
![Evolución de Accuracy BERT-Mini](Proyecto_ml/graficas/bert_mini_accuracy_evolution.png)

#### **Matriz de Confusión**
![Matriz de Confusión BERT-Mini](Proyecto_ml/graficas/bert_mini_confusion_matrix.png)

#### **Evolución de F1-Score**
![F1-Score BERT-Mini](Proyecto_ml/graficas/bert_mini_f1_evolution.png)

#### **Función de Costo**
![Pérdida BERT-Mini](Proyecto_ml/graficas/bert_mini_loss_function.png)

---

### **DistilBERT**

#### **Evolución de Accuracy**
![Evolución de Accuracy DistilBERT](Proyecto_ml/graficas/distilbert_accuracy_evolution.png)

#### **Matriz de Confusión**
![Matriz de Confusión DistilBERT](Proyecto_ml/graficas/distilbert_confusion_matrix.png)

#### **Evolución de F1-Score**
![F1-Score DistilBERT](Proyecto_ml/graficas/distilbert_f1_evolution.png)

#### **Función de Costo**
![Pérdida DistilBERT](Proyecto_ml/graficas/distilbert_loss_function.png)

---

## 📊 Comparación de Modelos

### **1️⃣ Comparación de Accuracy entre Modelos**
![Comparación de Accuracy](Proyecto_ml/graficas/comparison_accuracy.png)

### **2️⃣ Evolución de Accuracy**
![Evolución de Accuracy](Proyecto_ml/graficas/comparison_accuracy_evolution.png)

### **3️⃣ Comparación de Todas las Métricas**
![Comparación de Métricas](Proyecto_ml/graficas/comparison_all_metrics.png)

### **4️⃣ Comparación de F1-Score**
![Comparación de F1-Score](Proyecto_ml/graficas/comparison_f1.png)

### **5️⃣ Gráfico Radar – Comparación Completa**
![Gráfico Radar](Proyecto_ml/graficas/comparison_radar.png)

---

## 🏆 **Conclusión Final**

1️⃣ **DistilBERT es el mejor modelo**, con **mayor precisión, mejor estabilidad y menor confusión**. 

2️⃣ **BERT-Mini es una alternativa intermedia**, con buen rendimiento pero menor que DistilBERT.  

3️⃣ **TinyBERT es el menos eficiente**, con **más errores de clasificación y menor precisión**.  

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

## ⚠️ Limitaciones

⚠️ Uso de un dataset reducido para optimizar tiempos de entrenamiento.
⚠️ Algunas clases pueden requerir mayor ajuste en hiperparámetros.
⚠️ Para grandes modelos, puede requerirse **más recursos computacionales**.

---


