# prediccion.py
import joblib
import pandas as pd

print("=" * 50)
print("🔧 Script de predicción — carga modelo y predice una fila de ejemplo")
print("=" * 50)

from utils.helpers import load_model, preprocess_for_model, safe_predict

MODEL_PATH = 'modelo_predictor_total.pkl'

try:
    modelo = load_model(MODEL_PATH)
    print("✅ Modelo cargado exitosamente.")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    raise SystemExit(1)

print("\n📊 Predicción de ejemplo usando la primera fila de data/dataset.csv (si existe)")
try:
    df_demo = pd.read_csv('data/dataset.csv')
    row = df_demo.iloc[[0]]
except Exception:
    # Fallback a ejemplo manual
    row = pd.DataFrame([[100.0, 50.0, 5, 2024, 2, 0, 0, 10.0, 2.0]], columns=[
        'Subtotal', 'Precio', 'mes', 'año', 'dia_semana',
        'Genero_num', 'TipoFacturacion_num', 'diferencia_precio', 'ratio_subtotal_precio'
    ])

print("\n📥 Datos de entrada:")
print(row)

X = preprocess_for_model(row)
try:
    pred = safe_predict(modelo, X)
    print("\n" + "=" * 50)
    print(f"🔮 Predicción: {pred}")
    print("=" * 50)
except Exception as e:
    print(f"❌ Error durante la predicción: {e}")