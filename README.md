
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
## MLflow desde local

1️⃣ **Metricas del Modelo 1**

![pantallazo1](Proyecto_ml/images/pantallazo1.png)


2️⃣ **Metricas del Modelo 2**

![pantallazo2](Proyecto_ml/images/pantallazo2.png)


3️⃣ **Metricas del Modelo 3**

![pantallazo3](Proyecto_ml/images/pantallazo3.png)


4️⃣ **Metricas del Modelo 4**

![pantallazo4](Proyecto_ml/images/pantallazo4.png)


5️⃣ **Overview**

![pantallazo5](Proyecto_ml/images/pantallazo5.png)


6️⃣ **Artifacts**

![pantallazo6](Proyecto_ml/images/pantallazo6.png)

---

## 📊 Análisis de Cada Modelo

### **TinyBERT**
#### **Evolución de Accuracy**
![Evolución de Accuracy TinyBERT](Proyecto_ml/graficas/tinybert_accuracy_evolution.png)
- **Crecimiento progresivo de accuracy**, comenzando en **0.715** y alcanzando **0.825**.
- Aunque mejora con las épocas, **su rendimiento es inferior a los otros modelos**, indicando menor capacidad de aprendizaje.
- La tendencia de crecimiento indica que aún podría mejorar con más entrenamiento, pero el ritmo de crecimiento sugiere que podría estancarse pronto.

#### **Matriz de Confusión**
![Matriz de Confusión TinyBERT](Proyecto_ml/graficas/tinybert_confusion_matrix.png)
- **Mayor confusión en las categorías Business y Sci/Tech**, lo que indica que el modelo **tiene dificultades en separar noticias de negocios y tecnología**.
- Predicciones más erróneas en comparación con los otros modelos.
- La clasificación de la categoría “Mundo” es más precisa que otras, lo que sugiere que el modelo diferencia mejor eventos globales que temas técnicos.

#### **Evolución de F1-Score**
![F1-Score TinyBERT](Proyecto_ml/graficas/tinybert_f1_evolution.png)
- **F1-score final de 0.8247**, reflejando que el modelo **no logra un balance óptimo entre precisión y recall**.
- **La diferencia entre accuracy y F1-score indica que el modelo tiende a predecir algunas categorías con mayor confianza que otras**, pero sigue teniendo errores significativos en ciertos segmentos del dataset.


#### **Función de Costo**
![Pérdida TinyBERT](Proyecto_ml/graficas/tinybert_loss_function.png)
- **Pérdida inicial más alta** y reducción más lenta, lo que sugiere que **requiere más entrenamiento para converger**.
- Puede haber indicios de sobreajuste si se entrena demasiado sin mejorar sustancialmente el desempeño.
- **La función de costo no se estabiliza completamente**, lo que indica que el modelo sigue ajustando parámetros sin alcanzar una convergencia clara.

---

### **BERT-Mini**
#### **Evolución de Accuracy**
![Evolución de Accuracy BERT-Mini](Proyecto_ml/graficas/bert_mini_accuracy_evolution.png)
-- **Accuracy final de 0.832**, lo que representa una mejora constante durante las épocas de entrenamiento.
- Su crecimiento es más estable que TinyBERT, pero no alcanza la precisión lograda por DistilBERT.
- La tendencia sugiere que el modelo podría beneficiarse de más épocas de entrenamiento para lograr un mayor ajuste sin caer en sobreajuste.

#### **Matriz de Confusión**
![Matriz de Confusión BERT-Mini](Proyecto_ml/graficas/bert_mini_confusion_matrix.png)
- **Menos confusión** que TinyBERT, lo que indica una mejor separación de clases.
- Sigue mostrando errores en la clasificación de noticias de **Negocios** y **Ciencia/Tecnología**, lo que sugiere que los términos utilizados en estos dominios pueden ser similares y confundir al modelo.
- Se observa una leve mejora en la correcta clasificación de la categoría **Mundo**, donde hay menos errores que en TinyBERT.

#### **Evolución de F1-Score**
![F1-Score BERT-Mini](Proyecto_ml/graficas/bert_mini_f1_evolution.png)
- **F1-score final de 0.8311**, mostrando un equilibrio entre precisión y recall.
- Su crecimiento es consistente y más estable que en TinyBERT, lo que sugiere que generaliza mejor.
- Aunque es mejor que TinyBERT, aún tiene margen de mejora, ya que no logra igualar el rendimiento de DistilBERT.

#### **Función de Costo**
![Pérdida BERT-Mini](Proyecto_ml/graficas/bert_mini_loss_function.png)
- **Reducción de pérdida más rápida** en comparación con TinyBERT, lo que indica que el modelo se ajusta más eficientemente.
- Se observa una tendencia de convergencia más clara, aunque podría beneficiarse de ajustes en la tasa de aprendizaje para optimizar aún más su desempeño.

---

### **DistilBERT**
#### **Evolución de Accuracy**
![Evolución de Accuracy DistilBERT](Proyecto_ml/graficas/distilbert_accuracy_evolution.png)
- **Accuracy final de 0.866**, la más alta de los tres modelos, lo que indica un mejor rendimiento general.
- La curva de aprendizaje muestra una estabilidad clara desde la primera época, lo que sugiere que el modelo converge más rápido y con menor necesidad de ajustes.
- Su alto desempeño sugiere que captura mejor las relaciones entre palabras dentro del dataset AG News.

#### **Matriz de Confusión**
![Matriz de Confusión DistilBERT](Proyecto_ml/graficas/distilbert_confusion_matrix.png)
- **Menor confusión general**, mostrando una clara mejora en la diferenciación de clases.
- Aunque sigue habiendo algunos errores en la clasificación de Negocios y Ciencia/Tecnología, estos son considerablemente menores que en los otros modelos.
- Su precisión en todas las categorías es más alta, con una reducción notable en los falsos positivos y falsos negativos.


#### **Evolución de F1-Score**
![F1-Score DistilBERT](Proyecto_ml/graficas/distilbert_f1_evolution.png)
- **F1-score de 0.8653**, lo que muestra que DistilBERT tiene el mejor balance entre precisión y recall.
- Su curva de mejora es más rápida que la de los otros modelos, lo que indica que aprende de manera más eficiente desde las primeras épocas.

#### **Función de Costo**
![Pérdida DistilBERT](Proyecto_ml/graficas/distilbert_loss_function.png)
- **Menor pérdida final** y mejor convergencia en comparación con los otros modelos.
- Se observa una reducción rápida y estable de la función de costo, lo que sugiere que el modelo encuentra un óptimo más eficientemente.
- No se observan indicios de sobreajuste, lo que indica que DistilBERT es capaz de generalizar mejor en los datos de prueba.

---

## 📊 Comparación de Modelos

### **Comparación de Accuracy entre Modelos**
![Comparación de Accuracy](Proyecto_ml/graficas/comparison_accuracy.png)
 **DistilBERT lidera con 0.866, mientras que TinyBERT queda en último lugar con 0.826**.
- **BERT-Mini se encuentra en un punto intermedio, con 0.832**, lo que muestra que tiene un desempeño aceptable pero no alcanza el nivel de DistilBERT.

### **Evolución de Accuracy**
![Evolución de Accuracy](Proyecto_ml/graficas/comparison_accuracy_evolution.png)
- **DistilBERT tiene una curva de aprendizaje más estable y rápida**.
- **TinyBERT muestra la curva de mejora más lenta**, lo que sugiere que necesita más entrenamiento para alcanzar un rendimiento competitivo.

### **Comparación de Todas las Métricas**
![Comparación de Métricas](Proyecto_ml/graficas/comparison_all_metrics.png)
- DistilBERT lidera en **accuracy, precision, recall y F1-score**.
- **TinyBERT es el modelo con menor recall**, indicando que tiene más dificultades en identificar correctamente todas las categorías.

### **Comparación de F1-Score**
![Comparación de F1-Score](Proyecto_ml/graficas/comparison_f1.png)
- **DistilBERT mantiene el mejor F1-score, asegurando un equilibrio óptimo en clasificación**.
- **La diferencia entre los modelos indica que TinyBERT es el más débil en términos de balance entre precisión y recall**.

### **Gráfico Radar – Comparación Completa**
![Gráfico Radar](Proyecto_ml/graficas/comparison_radar.png)
- **DistilBERT domina todas las métricas** y es el modelo más eficiente.
- **TinyBERT tiene las puntuaciones más bajas en todas las métricas evaluadas**.

---

## 🏆 **Conclusión Final**

1️⃣ **DistilBERT es el mejor modelo**, con **mayor precisión y menor confusión**.

2️⃣ **BERT-Mini es una alternativa intermedia**, con **buen rendimiento pero inferior a DistilBERT**.

3️⃣ **TinyBERT es el menos eficiente**, con **más errores de clasificación y menor precisión**.

4️⃣ **Los tres modelos muestran dificultades similares en diferenciar noticias de negocios y tecnología**, sugiriendo que se podrían mejorar los embeddings o el ajuste fino del dataset.

---

## ⚠️ Limitaciones

⚠️ Uso de un dataset reducido para optimizar tiempos de entrenamiento.
⚠️ Algunas clases pueden requerir mayor ajuste en hiperparámetros.
⚠️ Para grandes modelos, puede requerirse **más recursos computacionales**.

---


