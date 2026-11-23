# Guía para sacar las evidencias de ejecución

## 1. Preparar el entorno

Abre una terminal dentro de la carpeta `proyecto_consumo` y ejecuta:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

En Windows el segundo comando es `venv\Scripts\activate`.

## 2. Capturas a sacar

Para cada captura, que se vea la ventana completa. En las pantallas del navegador, que se vea la barra de dirección. Guarda cada una con el nombre indicado.

| N° | Qué hacer | Qué debe verse | Nombre del archivo |
|----|-----------|----------------|--------------------|
| 1 | En la terminal: `python generar_datos.py` | "Se generaron 150 hogares ficticios", las primeras filas y la correlación entre temperatura y horas de aire | `captura_01_generar_csv.png` |
| 2 | Abrir `datos_hogares.csv` (Excel, Numbers o el editor) | 150 filas y las 5 columnas | `captura_02_csv.png` |
| 3 | En la terminal: `python app.py` | La línea `Running on http://127.0.0.1:5001` | `captura_03_servidor.png` |
| 4 | Abrir `http://127.0.0.1:5001` en el navegador | Pantalla de inicio con el modelo y la ecuación | `captura_04_inicio.png` |
| 5 | Paso 01, clic en **Cargar datos** | Display con 150 hogares y 5 columnas | `captura_05_cargar.png` |
| 6 | Paso 02, parte de arriba | Primeras 10 filas y tabla de promedio, mínimo y máximo | `captura_06_datos.png` |
| 7 | Paso 02, bajar con el scroll | Matriz de correlación | `captura_07_correlacion.png` |
| 8 | Paso 03, clic en **Entrenar modelo** | División 120 / 30 y las matrices X y Y | `captura_08_matrices_X_Y.png` |
| 9 | Paso 03, bajar con el scroll | Xᵀ, XᵀX, XᵀY, las líneas de código y β | `captura_09_matrices_beta.png` |
| 10 | Paso 04 | Los 5 coeficientes con su interpretación y la ecuación | `captura_10_coeficientes.png` |
| 11 | Paso 05, escribir los datos de un hogar (por ejemplo 4 personas, 12 dispositivos, 5 horas, 28 °C) | Formulario lleno, antes de enviar | `captura_11_hogar_nuevo.png` |
| 12 | Clic en **Predecir consumo** (paso 06) | Display con los kWh estimados y la tabla "Cómo se calculó" | `captura_12_prediccion.png` |
| 13 | Paso 07 | MAE, RMSE y R² con su explicación | `captura_13_metricas.png` |
| 14 | Paso 08 | Gráfica de consumo real vs. predicho | `captura_14_grafica.png` |
| 15 | Opcional: otro hogar distinto (por ejemplo 1 persona, 5 dispositivos, 0 horas, 18 °C) | Que la predicción cambia según los datos | `captura_15_segunda_prediccion.png` |

Notas:
- Los pasos se hacen en orden. El led amarillo del panel izquierdo indica que el paso ya se puede usar.
- La gráfica también queda guardada como `static/grafica.png`, y esa imagen la puedes poner directo en el informe.
- Si reinicias el servidor, hay que volver a cargar los datos y entrenar, porque todo se guarda en memoria.

## 3. Guion para la demostración en vivo (3 a 4 minutos)

1. Mostrar `generar_datos.py` y explicar la fórmula con la que se inventan los datos y por qué se les agrega ruido.
2. Correr `python app.py`, cargar el CSV y mostrar las correlaciones. Comentar que temperatura y horas de aire están relacionadas, pero no al 100 %.
3. Entrenar el modelo y explicar X, Xᵀ, Y y β en pantalla.
4. Abrir `modelo.py` y señalar las cuatro líneas de la ecuación normal, en especial `np.linalg.inv`.
5. Mostrar los coeficientes y leer en voz alta la interpretación de uno o dos.
6. Ingresar un hogar nuevo y explicar la tabla "Cómo se calculó".
7. Mostrar MAE, RMSE, R² y la gráfica.

## 4. Si algo falla

| Problema | Solución |
|----------|----------|
| `ModuleNotFoundError` | No está activado el entorno virtual o falta `pip install -r requirements.txt` |
| "No se encontró datos_hogares.csv" | Ejecutar primero `python generar_datos.py` |
| Puerto en uso | Cambiar `port=5001` por otro número al final de `app.py` |
| Las fuentes se ven distintas | Las fuentes se descargan de internet. Sin conexión se usan fuentes del sistema y todo funciona igual |
