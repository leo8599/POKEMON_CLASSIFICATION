import torch
import model_builder
from torchviz import make_dot
from torchinfo import summary

# 1. Configurar dispositivo y crear el modelo
device = "cuda" if torch.cuda.is_available() else "cpu"
class_names_count = 150 # Pon aquí el número de clases de tus Pokemon (ej. 10, 150, etc.)

# Instanciamos el modelo igual que en el entrenamiento
model = model_builder.create_efficientnet_b0(output_shape=class_names_count, device=device)

# ---------------------------------------------------------
# OPCIÓN A: Resumen detallado en texto (Muy útil para reportes)
# ---------------------------------------------------------
print("\n[INFO] Generando resumen de la arquitectura (torchinfo)...")
model_stats = summary(model, 
                      input_size=(1, 3, 224, 224), # (Batch, Canales, Alto, Ancho)
                      col_names=["input_size", "output_size", "num_params", "trainable"],
                      col_width=20,
                      row_settings=["var_names"])
print(model_stats)

# Guardar esto en un archivo de texto para tu reporte
with open("model_architecture_summary.txt", "w") as f:
    f.write(str(model_stats))

# ---------------------------------------------------------
# OPCIÓN B: Gráfico visual del flujo de datos (torchviz)
# ---------------------------------------------------------
print("\n[INFO] Generando gráfico visual de la red (torchviz)...")
try:
    # Pasamos un dato falso (dummy) para trazar el camino
    x = torch.randn(1, 3, 224, 224).to(device)
    y = model(x)
    
    # Generamos el diagrama
    dot = make_dot(y, params=dict(model.named_parameters()))
    
    # Ajustamos formato y guardamos
    dot.format = 'png'
    dot.render("pokemon_model_architecture_viz") # Guarda como pokemon_model_architecture_viz.png
    print("[INFO] Gráfico guardado como 'pokemon_model_architecture_viz.png'")
    
except Exception as e:
    print(f"[AVISO] No se pudo generar el gráfico visual (probablemente falta Graphviz en el OS): {e}")
    print("No te preocupes, el resumen de texto de arriba suele ser suficiente para reportes técnicos.")