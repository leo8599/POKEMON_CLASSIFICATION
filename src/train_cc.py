"""
Realiza Fine-Tuning sobre un modelo EfficientNet-B0 previamente entrenado.
Adapta el estilo de train_cc.py para nuestro caso de uso específico.
"""

import os
import torch
import data_setup, engine, model_builder, utils
from torchvision import transforms
from timeit import default_timer as timer

def main():
    # ---------------------------------------------------------
    # 1. Configuración de Hiperparámetros para Fine-Tuning
    # ---------------------------------------------------------
    NUM_EPOCHS = 5          # Menos épocas, ya que el modelo ya sabe bastante
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001  # Tasa de aprendizaje 10x más pequeña para ajustes finos

    # Directorios de datos
    train_dir = "data/train"
    test_dir = "data/test"

    # Configuración de rutas de modelos
    PRETRAINED_MODEL_PATH = "models/pokemon_efficientnet_model.pth" # El modelo que creamos en la fase anterior
    NEW_MODEL_NAME = "pokemon_efficientnet_finetuned.pth"
    MODEL_SAVE_DIR = "models"
    
    # Configurar dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        device = "mps"
        
    print(f"[INFO] Fine-Tuning en dispositivo: {device}")

    # ---------------------------------------------------------
    # 2. Transformaciones (Deben coincidir con EfficientNet)
    # ---------------------------------------------------------
    # Usamos 224x224 como requiere EfficientNet (train_cc usaba 64x64 para TinyVGG)
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), # Un poco de aumento de datos
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                             std=[0.5, 0.5, 0.5])
    ])

    # Crear DataLoaders
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

    # ---------------------------------------------------------
    # 3. Crear y Cargar el Modelo Base
    # ---------------------------------------------------------
    # Instanciamos la arquitectura correcta (EfficientNet)
    model = model_builder.create_efficientnet_b0(
        output_shape=len(class_names),
        device=device
    )

    # Cargar los pesos del entrenamiento previo (Transfer Learning)
    if os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"[INFO] Cargando pesos previos desde: {PRETRAINED_MODEL_PATH}")
        try:
            state_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            print("[INFO] Pesos cargados exitosamente.")
        except Exception as e:
            print(f"[ERROR] Falló la carga de pesos: {e}")
            return # Detenemos si no hay modelo base, el fine-tuning requiere un modelo previo
    else:
        print(f"[ERROR] No se encontró el modelo pre-entrenado en {PRETRAINED_MODEL_PATH}.")
        print("Ejecuta primero el entrenamiento normal (train.py) antes de hacer fine-tuning.")
        return

    # ---------------------------------------------------------
    # 4. Lógica de Fine-Tuning (Descongelar capas)
    # ---------------------------------------------------------
    # En Transfer Learning, model.features estaba congelado. Ahora lo descongelamos.
    # Opción A: Descongelar TODO (útil si el LR es muy bajo)
    for param in model.features.parameters():
        param.requires_grad = True
        
    # Opción B (Más conservadora): Descongelar solo los últimos bloques
    # for param in model.features[-2:].parameters():
    #     param.requires_grad = True
    
    print("[INFO] Capas base descongeladas para Fine-Tuning.")

    # ---------------------------------------------------------
    # 5. Entrenamiento
    # ---------------------------------------------------------
    # Usamos un LR bajo para no distorsionar los pesos pre-entrenados
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_time = timer()
    
    # Reutilizamos el motor de entrenamiento
    results = engine.train(model=model,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           loss_fn=loss_fn,
                           optimizer=optimizer,
                           epochs=NUM_EPOCHS,
                           device=device)
    
    end_time = timer()
    print(f"[INFO] Tiempo de Fine-Tuning: {end_time-start_time:.3f} segundos")

    # ---------------------------------------------------------
    # 6. Guardar Resultados
    # ---------------------------------------------------------
    utils.save_model(model=model,
                     target_dir=MODEL_SAVE_DIR,
                     model_name=NEW_MODEL_NAME)
                     
    utils.plot_loss_curves(results, output_path="pokemon_finetuning_results.png")

if __name__ == "__main__":
    main()