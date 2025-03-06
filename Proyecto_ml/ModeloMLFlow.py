# -*- coding: utf-8 -*-
"""
Versión mejorada de MLflow para seguimiento y comparación de múltiples modelos de IA
"""
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import os
from torch.utils.data import DataLoader
import seaborn as sns
from transformers.trainer_callback import TrainerCallback
import pandas as pd

# Crear directorios para guardar gráficas y modelos si no existen
os.makedirs("./graficas", exist_ok=True)
os.makedirs("./modelos", exist_ok=True)

# Definir lista de modelos a evaluar
MODELS_TO_EVALUATE = [
    {
        "name": "TinyBERT",
        "model_id": "huawei-noah/TinyBERT_General_4L_312D",
        "dir_name": "tinybert"
    },
    {
        "name": "DistilBERT",
        "model_id": "distilbert-base-uncased",
        "dir_name": "distilbert"
    },
    {
        "name": "BERT-Mini",
        "model_id": "prajjwal1/bert-mini",
        "dir_name": "bert_mini"
    }
]

# Set MLflow experiment name
try:
    mlflow.set_experiment("multiple-text-classification-models")
except Exception as e:
    print(f"Advertencia al configurar experimento: {e}")
    print("Continuando con el experimento predeterminado.")

# Cargar el dataset 'ag_news' de Hugging Face
print("Cargando dataset...")
dataset = load_dataset('ag_news')

# Reducir el tamaño del dataset para hacerlo más rápido
dataset = dataset.shuffle(seed=42)
train_dataset = dataset['train'].select(range(1000))  # Usar solo 1000 muestras
val_dataset = dataset['test'].select(range(500))  # Usar solo 500 muestras

# Obtener los nombres de las clases para interpretación posterior
label_names = dataset['train'].features['label'].names
num_labels = len(label_names)

# Definir la longitud máxima para padding y truncamiento
MAX_LENGTH = 128

# Función para tokenizar las entradas de texto con cualquier tokenizador
def get_tokenize_function(tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=MAX_LENGTH)
    return tokenize_function

# Clase para Dataset de PyTorch
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'input_ids': torch.tensor(item['input_ids']),
            'attention_mask': torch.tensor(item['attention_mask']),
            'labels': torch.tensor(item['label']),
        }

# Callback personalizado para registrar métricas durante el entrenamiento
class MetricsCallback(TrainerCallback):
    def __init__(self, model_name):
        self.current_step = 0
        self.model_name = model_name
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1s = []
        self.val_precisions = []
        self.val_recalls = []
        self.steps = []
        self.epochs = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = logs.get("step", 0)
            
            # Registrar pérdida de entrenamiento
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
                self.steps.append(step)
                self.current_step = step
                try:
                    mlflow.log_metric(f"{self.model_name}_train_loss", logs["loss"], step=step)
                except Exception as e:
                    print(f"Error al registrar train_loss en MLflow: {e}")
            
            # Registrar métricas de evaluación
            if "eval_loss" in logs:
                self.val_losses.append(logs["eval_loss"])
                self.epochs.append(state.epoch)
                
                # También registrar las métricas adicionales de evaluación
                if "eval_accuracy" in logs:
                    self.val_accuracies.append(logs["eval_accuracy"])
                    try:
                        mlflow.log_metric(f"{self.model_name}_val_accuracy", logs["eval_accuracy"], step=self.current_step)
                    except Exception as e:
                        print(f"Error al registrar val_accuracy en MLflow: {e}")
                
                if "eval_f1" in logs:
                    self.val_f1s.append(logs["eval_f1"])
                    try:
                        mlflow.log_metric(f"{self.model_name}_val_f1", logs["eval_f1"], step=self.current_step)
                    except Exception as e:
                        print(f"Error al registrar val_f1 en MLflow: {e}")
                
                if "eval_precision" in logs:
                    self.val_precisions.append(logs["eval_precision"])
                    try:
                        mlflow.log_metric(f"{self.model_name}_val_precision", logs["eval_precision"], step=self.current_step)
                    except Exception as e:
                        print(f"Error al registrar val_precision en MLflow: {e}")
                
                if "eval_recall" in logs:
                    self.val_recalls.append(logs["eval_recall"])
                    try:
                        mlflow.log_metric(f"{self.model_name}_val_recall", logs["eval_recall"], step=self.current_step)
                    except Exception as e:
                        print(f"Error al registrar val_recall en MLflow: {e}")
                
                # Registrar loss de validación en MLflow
                try:
                    mlflow.log_metric(f"{self.model_name}_val_loss", logs["eval_loss"], step=self.current_step)
                except Exception as e:
                    print(f"Error al registrar val_loss en MLflow: {e}")

# Función para calcular métricas más completas
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calcular todas las métricas requeridas
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Función para guardar los gráficos y subirlos a MLflow
def guardar_y_subir_grafica(filename, titulo):
    path = f"./graficas/{filename}"
    plt.savefig(path)
    print(f"Gráfico guardado como '{path}'")
    try:
        mlflow.log_artifact(path)
    except Exception as e:
        print(f"Error al registrar gráfico {filename}: {e}")

# Para almacenar los resultados finales de cada modelo
model_results = {}

# Inicio del experimento MLflow
print("Configurando experimento MLflow...")
try:
    mlflow.start_run(run_name="comparacion-modelos")
except Exception as e:
    print(f"Error al iniciar run de MLflow: {e}")
    print("Continuando sin tracking de MLflow...")

# Iterar sobre los modelos a evaluar
for model_config in MODELS_TO_EVALUATE:
    model_name = model_config["name"]
    model_id = model_config["model_id"]
    dir_name = model_config["dir_name"]
    
    print(f"\n{'='*50}")
    print(f"Evaluando modelo: {model_name}")
    print(f"{'='*50}")
    
    # Cargar el tokenizador específico para este modelo
    print(f"Cargando tokenizador para {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Tokenizar datasets para este modelo específico
    print(f"Tokenizando datasets para {model_name}...")
    tokenize_function = get_tokenize_function(tokenizer)
    train_dataset_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_dataset_tokenized = val_dataset.map(tokenize_function, batched=True)
    
    # Convertir a formato PyTorch Dataset
    train_dataset_torch = TextDataset(train_dataset_tokenized)
    val_dataset_torch = TextDataset(val_dataset_tokenized)
    
    # Registrar parámetros del experimento para este modelo
    try:
        mlflow.log_params({
            f"{dir_name}_model_name": model_id,
            f"{dir_name}_max_length": MAX_LENGTH,
            f"{dir_name}_train_samples": len(train_dataset),
            f"{dir_name}_val_samples": len(val_dataset),
            f"{dir_name}_num_train_epochs": 2,
            f"{dir_name}_batch_size": 8
        })
    except Exception as e:
        print(f"Error al registrar parámetros para {model_name}: {e}")
    
    # Cargar el modelo para clasificación
    print(f"Cargando modelo {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=num_labels
    )
    
    # Definir los argumentos de entrenamiento
    model_output_dir = f"./results/{dir_name}"
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        logging_dir=f'./logs/{dir_name}',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",  # Desactivar integración predeterminada
    )
    
    # Inicializar callback personalizado
    metrics_callback = MetricsCallback(dir_name)
    
    # Configurar el Trainer con nuestro callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_torch,
        eval_dataset=val_dataset_torch,
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback],
    )
    
    # Entrenamiento
    print(f"Iniciando entrenamiento de {model_name}...")
    trainer.train()
    
    # Evaluación final del modelo
    print(f"Evaluando modelo final {model_name}...")
    eval_results = trainer.evaluate()
    
    # Registrar métricas finales
    print(f"\n=== Métricas Finales para {model_name} ===")
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")
        try:
            mlflow.log_metric(f"{dir_name}_final_{key}", value)
        except Exception as e:
            print(f"Error al registrar métrica final {key} para {model_name}: {e}")
    
    # Guardar el modelo entrenado
    model_save_path = f"./modelos/{dir_name}"
    trainer.save_model(model_save_path)
    print(f"Modelo {model_name} guardado en {model_save_path}")
    
    # Guardar los resultados para este modelo
    model_results[model_name] = {
        'metrics_callback': metrics_callback,
        'eval_results': eval_results
    }
    
    # Generar gráficas individuales para este modelo
    
    # 1. Gráfica de pérdida (loss) durante el entrenamiento
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_callback.steps, metrics_callback.train_losses, 'r-', label='Train')
    # Interpolar val_losses para alinear con los pasos
    if len(metrics_callback.val_losses) > 1 and len(metrics_callback.epochs) > 1:
        val_steps = [int(metrics_callback.epochs[i] * len(metrics_callback.steps) / max(metrics_callback.epochs)) 
                    for i in range(len(metrics_callback.epochs))]
        plt.plot(val_steps, metrics_callback.val_losses, 'b-', label='Test')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title(f'Función de Costo durante el Entrenamiento - {model_name}')
    plt.legend()
    plt.grid(True)
    guardar_y_subir_grafica(f"{dir_name}_loss_function.png", f"Función de Costo - {model_name}")
    
    # 2. Gráfica de Accuracy
    plt.figure(figsize=(10, 6))
    if len(metrics_callback.val_accuracies) > 0:
        plt.plot(metrics_callback.epochs, metrics_callback.val_accuracies, 'b-o', label='Test')
        plt.axhline(y=eval_results['eval_accuracy'], color='g', linestyle='--', label='Final Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Evolución de Accuracy durante el Entrenamiento - {model_name}')
        plt.legend()
        plt.grid(True)
        guardar_y_subir_grafica(f"{dir_name}_accuracy_evolution.png", f"Evolución de Accuracy - {model_name}")
    
    # 3. Gráfica de F1-Score
    plt.figure(figsize=(10, 6))
    if len(metrics_callback.val_f1s) > 0:
        plt.plot(metrics_callback.epochs, metrics_callback.val_f1s, 'g-o', label='Test')
        plt.axhline(y=eval_results['eval_f1'], color='b', linestyle='--', label='Final F1-Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1-Score')
        plt.title(f'Evolución de F1-Score durante el Entrenamiento - {model_name}')
        plt.legend()
        plt.grid(True)
        guardar_y_subir_grafica(f"{dir_name}_f1_evolution.png", f"Evolución de F1-Score - {model_name}")
    
    # 4. Matriz de confusión
    print(f"Calculando matriz de confusión para {model_name}...")
    try:
        # Obtener predicciones para el conjunto de validación
        val_dataloader = DataLoader(val_dataset_torch, batch_size=training_args.per_device_eval_batch_size)
        model.eval()
        device = trainer.model.device

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_dataloader:
                # Mover datos al mismo dispositivo que el modelo
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        # Calcular y visualizar la matriz de confusión
        conf_matrix = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(10, 8))
        
        # Usar un try-except específico para el llamado a heatmap que parece estar fallando
        try:
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_names, yticklabels=label_names)
        except Exception as e:
            print(f"Error al crear heatmap: {e}. Intentando con una implementación más simple.")
            plt.imshow(conf_matrix, cmap='Blues')
            plt.colorbar()
            # Añadir anotaciones manualmente
            for i in range(len(conf_matrix)):
                for j in range(len(conf_matrix[i])):
                    plt.text(j, i, conf_matrix[i, j], ha="center", va="center")
        
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.title(f'Matriz de Confusión - {model_name}')
        plt.tight_layout()
        guardar_y_subir_grafica(f"{dir_name}_confusion_matrix.png", f"Matriz de Confusión - {model_name}")
    except Exception as e:
        print(f"Error al calcular matriz de confusión para {model_name}: {e}")
        print("Continuando con las siguientes gráficas...")

# ---------------------------------------------------------
# Crear gráficas comparativas de todos los modelos evaluados
# ---------------------------------------------------------

print("\nGenerando gráficas comparativas de todos los modelos...")

# Métricas finales para comparar todos los modelos
model_names = []
accuracy_values = []
f1_values = []
precision_values = []
recall_values = []

for model_name, results in model_results.items():
    model_names.append(model_name)
    accuracy_values.append(results['eval_results']['eval_accuracy'])
    f1_values.append(results['eval_results']['eval_f1'])
    precision_values.append(results['eval_results']['eval_precision'])
    recall_values.append(results['eval_results']['eval_recall'])

# 1. Gráfico de barras de Accuracy para todos los modelos
plt.figure(figsize=(12, 6))
bars = plt.bar(model_names, accuracy_values, color=['steelblue', 'darkorange', 'forestgreen'])
plt.xlabel('Modelo')
plt.ylabel('Accuracy')
plt.title('Comparación de Accuracy entre Modelos')
plt.ylim(0, 1)

# Añadir valores en las barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.4f}', ha='center', va='bottom')

plt.tight_layout()
guardar_y_subir_grafica("comparison_accuracy.png", "Comparación de Accuracy")

# 2. Gráfico de barras de F1-Score para todos los modelos
plt.figure(figsize=(12, 6))
bars = plt.bar(model_names, f1_values, color=['steelblue', 'darkorange', 'forestgreen'])
plt.xlabel('Modelo')
plt.ylabel('F1-Score')
plt.title('Comparación de F1-Score entre Modelos')
plt.ylim(0, 1)

# Añadir valores en las barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.4f}', ha='center', va='bottom')

plt.tight_layout()
guardar_y_subir_grafica("comparison_f1.png", "Comparación de F1-Score")

# 3. Gráfico combinado de métricas para todos los modelos
plt.figure(figsize=(14, 8))

# Configurar el ancho de las barras
bar_width = 0.2
index = np.arange(len(model_names))

# Crear barras para cada métrica
plt.bar(index, accuracy_values, bar_width, label='Accuracy', color='steelblue')
plt.bar(index + bar_width, f1_values, bar_width, label='F1-Score', color='darkorange')
plt.bar(index + bar_width*2, precision_values, bar_width, label='Precision', color='forestgreen')
plt.bar(index + bar_width*3, recall_values, bar_width, label='Recall', color='firebrick')

# Etiquetas, título y leyenda
plt.xlabel('Modelo')
plt.ylabel('Valor')
plt.title('Comparación de Todas las Métricas entre Modelos')
plt.xticks(index + bar_width*1.5, model_names)
plt.legend()
plt.ylim(0, 1)

plt.tight_layout()
guardar_y_subir_grafica("comparison_all_metrics.png", "Comparación de Todas las Métricas")

# 4. Gráfico de líneas de evolución de Accuracy por época para todos los modelos
plt.figure(figsize=(12, 7))
for model_name, results in model_results.items():
    callback = results['metrics_callback']
    if len(callback.epochs) > 0 and len(callback.val_accuracies) > 0:
        plt.plot(callback.epochs, callback.val_accuracies, 'o-', label=model_name)

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Evolución de Accuracy por Modelo')
plt.legend()
plt.grid(True)
guardar_y_subir_grafica("comparison_accuracy_evolution.png", "Evolución de Accuracy por Modelo")

# 5. Gráfico de radar para comparar todas las métricas
# Preparar datos para el gráfico de radar
metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
num_metrics = len(metrics)

# Ángulos para el gráfico de radar
angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
angles += angles[:1]  # Cerrar el polígono

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

for i, model_name in enumerate(model_names):
    values = [
        accuracy_values[i],
        f1_values[i],
        precision_values[i],
        recall_values[i]
    ]
    values += values[:1]  # Cerrar el polígono
    
    ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
    ax.fill(angles, values, alpha=0.1)

# Configurar el gráfico
ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
ax.set_ylim(0, 1)
plt.legend(loc='upper right')
plt.title('Comparación de Modelos - Gráfico Radar')

plt.tight_layout()
guardar_y_subir_grafica("comparison_radar.png", "Comparación de Modelos - Radar")

# 6. Crear un DataFrame y guardar resultados en CSV para su uso posterior
results_df = pd.DataFrame({
    'Modelo': model_names,
    'Accuracy': accuracy_values,
    'F1-Score': f1_values,
    'Precision': precision_values,
    'Recall': recall_values
})

# Guardar resultados en CSV
results_csv_path = "./graficas/model_comparison_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"Resultados guardados en {results_csv_path}")

try:
    mlflow.log_artifact(results_csv_path)
except Exception as e:
    print(f"Error al registrar archivo CSV: {e}")

# Finalizar el run de MLflow
try:
    mlflow.end_run()
except Exception as e:
    print(f"Error al finalizar run de MLflow: {e}")

print("\nEntrenamiento, evaluación y generación de gráficas comparativas completados.")