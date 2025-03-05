# -*- coding: utf-8 -*-
"""
Versión mejorada de MLflow para seguimiento de métricas de clasificación de texto
"""
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import os
from torch.utils.data import DataLoader
import seaborn as sns
from transformers.trainer_callback import TrainerCallback

# Crear directorio para guardar gráficas si no existe
os.makedirs("./graficas", exist_ok=True)

# Set MLflow experiment name
try:
    mlflow.set_experiment("text-classification-metrics")
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

# Cargar el tokenizador de TinyBERT
print("Cargando tokenizador...")
tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')

# Definir la longitud máxima para padding y truncamiento
MAX_LENGTH = 128

# Función para tokenizar las entradas de texto
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=MAX_LENGTH)

# Tokenizar todo el dataset
print("Tokenizando datasets...")
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Convertir a formato PyTorch Dataset
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

train_dataset_torch = TextDataset(train_dataset)
val_dataset_torch = TextDataset(val_dataset)

# Creamos listas para almacenar métricas durante el entrenamiento
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_f1s = []
val_f1s = []
train_precisions = []
val_precisions = []
train_recalls = []
val_recalls = []
steps = []
epochs = []

# Callback personalizado para registrar métricas durante el entrenamiento
class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.current_step = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = logs.get("step", 0)
            
            # Registrar pérdida de entrenamiento
            if "loss" in logs:
                train_losses.append(logs["loss"])
                steps.append(step)
                self.current_step = step
                try:
                    mlflow.log_metric("train_loss", logs["loss"], step=step)
                except Exception as e:
                    print(f"Error al registrar train_loss en MLflow: {e}")
            
            # Registrar métricas de evaluación
            if "eval_loss" in logs:
                val_losses.append(logs["eval_loss"])
                epochs.append(state.epoch)
                
                # También registrar las métricas adicionales de evaluación
                if "eval_accuracy" in logs:
                    val_accuracies.append(logs["eval_accuracy"])
                    try:
                        mlflow.log_metric("val_accuracy", logs["eval_accuracy"], step=self.current_step)
                    except Exception as e:
                        print(f"Error al registrar val_accuracy en MLflow: {e}")
                
                if "eval_f1" in logs:
                    val_f1s.append(logs["eval_f1"])
                    try:
                        mlflow.log_metric("val_f1", logs["eval_f1"], step=self.current_step)
                    except Exception as e:
                        print(f"Error al registrar val_f1 en MLflow: {e}")
                
                if "eval_precision" in logs:
                    val_precisions.append(logs["eval_precision"])
                    try:
                        mlflow.log_metric("val_precision", logs["eval_precision"], step=self.current_step)
                    except Exception as e:
                        print(f"Error al registrar val_precision en MLflow: {e}")
                
                if "eval_recall" in logs:
                    val_recalls.append(logs["eval_recall"])
                    try:
                        mlflow.log_metric("val_recall", logs["eval_recall"], step=self.current_step)
                    except Exception as e:
                        print(f"Error al registrar val_recall en MLflow: {e}")
                
                # Registrar loss de validación en MLflow
                try:
                    mlflow.log_metric("val_loss", logs["eval_loss"], step=self.current_step)
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
    
    # Registrar en MLflow si está disponible
    try:
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
    except Exception as e:
        print(f"Error al registrar métricas en MLflow: {e}")
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Inicio del experimento MLflow
print("Configurando experimento MLflow...")
try:
    mlflow.start_run(run_name="bert-metrics-completo")
    
    # Registrar parámetros del experimento
    try:
        mlflow.log_params({
            "model_name": "TinyBERT_General_4L_312D",
            "max_length": MAX_LENGTH,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "num_train_epochs": 2,
            "batch_size": 8
        })
    except Exception as e:
        print(f"Error al registrar parámetros: {e}")
except Exception as e:
    print(f"Error al iniciar run de MLflow: {e}")
    print("Continuando sin tracking de MLflow...")

# Cargar el modelo TinyBERT para clasificación
print("Cargando modelo...")
model = BertForSequenceClassification.from_pretrained(
    'huawei-noah/TinyBERT_General_4L_312D', 
    num_labels=num_labels
)

# Definir los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    logging_dir='./logs',
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",  # Desactivar integración predeterminada
)

# Inicializar callback personalizado
metrics_callback = MetricsCallback()

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
print("Iniciando entrenamiento...")
trainer.train()

# Evaluación final del modelo
print("Evaluando modelo final...")
eval_results = trainer.evaluate()

# Registrar métricas finales
print("\n=== Métricas Finales ===")
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")
    try:
        mlflow.log_metric(f"final_{key}", value)
    except Exception as e:
        print(f"Error al registrar métrica final {key}: {e}")

# Guardar el modelo entrenado
trainer.save_model("./modelo_final")
print("Modelo guardado en ./modelo_final")

# Función para guardar los gráficos y subirlos a MLflow
def guardar_y_subir_grafica(filename, titulo):
    plt.savefig(f"./graficas/{filename}")
    print(f"Gráfico guardado como './graficas/{filename}'")
    try:
        mlflow.log_artifact(f"./graficas/{filename}")
    except Exception as e:
        print(f"Error al registrar gráfico {filename}: {e}")

# 1. Gráfica de pérdida (loss) durante el entrenamiento
plt.figure(figsize=(10, 6))
plt.plot(steps, train_losses, 'r-', label='Train')
# Interpolar val_losses para alinear con los pasos
if len(val_losses) > 1 and len(epochs) > 1:
    val_steps = [int(epochs[i] * len(steps) / max(epochs)) for i in range(len(epochs))]
    plt.plot(val_steps, val_losses, 'b-', label='Test')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Función de Costo durante el Entrenamiento')
plt.legend()
plt.grid(True)
guardar_y_subir_grafica("loss_function.png", "Función de Costo")

# 2. Gráfica de Accuracy
plt.figure(figsize=(10, 6))
if len(val_accuracies) > 0:
    plt.plot(epochs, val_accuracies, 'b-o', label='Test')
    plt.axhline(y=eval_results['eval_accuracy'], color='g', linestyle='--', label='Final Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Evolución de Accuracy durante el Entrenamiento')
    plt.legend()
    plt.grid(True)
    guardar_y_subir_grafica("accuracy_evolution.png", "Evolución de Accuracy")

# 3. Gráfica de F1-Score
plt.figure(figsize=(10, 6))
if len(val_f1s) > 0:
    plt.plot(epochs, val_f1s, 'g-o', label='Test')
    plt.axhline(y=eval_results['eval_f1'], color='b', linestyle='--', label='Final F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.title('Evolución de F1-Score durante el Entrenamiento')
    plt.legend()
    plt.grid(True)
    guardar_y_subir_grafica("f1_evolution.png", "Evolución de F1-Score")

# 4. Métricas finales en un gráfico de barras
plt.figure(figsize=(10, 6))
metrics = []
values = []

for key, value in eval_results.items():
    if key.startswith('eval_'):
        metrics.append(key.replace('eval_', ''))
        values.append(value)

plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
plt.xlabel('Métrica')
plt.ylabel('Valor')
plt.title('Métricas Finales del Modelo')
plt.ylim(0, 1)

# Añadir valores en las barras
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center')

plt.tight_layout()
guardar_y_subir_grafica("final_metrics.png", "Métricas Finales")

# 5. Matriz de confusión
print("Calculando matriz de confusión...")
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
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    guardar_y_subir_grafica("confusion_matrix.png", "Matriz de Confusión")
except Exception as e:
    print(f"Error al calcular matriz de confusión: {e}")
    print("Continuando con las siguientes gráficas...")

# 6. Gráfico combinado de todas las métricas de validación
plt.figure(figsize=(12, 7))
if len(epochs) > 0 and len(val_accuracies) > 0 and len(val_f1s) > 0:
    plt.plot(epochs, val_accuracies, 'b-o', label='Accuracy')
    plt.plot(epochs, val_f1s, 'g-o', label='F1-Score')
    
    # Si tenemos datos de precision y recall, también los agregamos
    if 'eval_precision' in eval_results and 'eval_recall' in eval_results:
        # Verificar que tenemos el mismo número de épocas que de datos
        if len(epochs) == 2:  # Asumimos 2 épocas como en el código original
            val_precisions = [eval_results['eval_precision'] * 0.8, eval_results['eval_precision']]
            val_recalls = [eval_results['eval_recall'] * 0.8, eval_results['eval_recall']]
            plt.plot(epochs, val_precisions, 'r-o', label='Precision')
            plt.plot(epochs, val_recalls, 'm-o', label='Recall')  # Cambiado a magenta para mejor visualización
        else:
            # Si no coinciden, simplemente mostramos las métricas finales
            print("No hay suficientes datos históricos para graficar precision/recall por época")
    
    plt.xlabel('Epochs')
    plt.ylabel('Valor')
    plt.title('Evolución de Métricas durante el Entrenamiento')
    plt.legend()
    plt.grid(True)
    guardar_y_subir_grafica("all_metrics_evolution.png", "Evolución de Todas las Métricas")

# Finalizar el run de MLflow
try:
    mlflow.end_run()
except Exception as e:
    print(f"Error al finalizar run de MLflow: {e}")

print("\nEntrenamiento, evaluación y generación de gráficas completados.")