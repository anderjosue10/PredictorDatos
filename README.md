# Mi App Streamlit - Predictor de Ventas

Archivos principales en esta carpeta:

- `modelo_predictor_total.pkl` — Modelo serialized (.pkl) ya presente.
- `prediccion.py` — Script de consola que carga el modelo y ejecuta una predicción de ejemplo.
- `app.py` — Aplicación Streamlit que carga el modelo y permite interactuar para predecir el monto de venta.

**Requisitos**

Instala dependencias en tu entorno (PowerShell):

```pwsh
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Ejecutar la predicción en consola**

```pwsh
# Ejecuta un ejemplo de predicción en consola
python prediccion.py
```

**Ejecutar la app Streamlit**

```pwsh
# Lanza la app (abrirá un navegador web local)
streamlit run app.py
```

**Notas importantes**

- Asegúrate de que `modelo_predictor_total.pkl` esté en la misma carpeta que `app.py` y `prediccion.py`.
- Nunca cargues archivos `.pkl` de fuentes no confiables.
- Si el modelo fue entrenado con versiones específicas de librerías (por ejemplo XGBoost), puede necesitarse la misma versión para evitar incompatibilidades.

Si quieres, puedo:
- Ejecutar `prediccion.py` aquí para verificar la salida (si me autorizas a correrlo).
- Añadir validaciones adicionales (chequear columnas esperadas antes de predecir).
- Crear un archivo `app_ventas.py` alternativo con interfaz distinta.
