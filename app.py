import math
import os
import runpy
import threading
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for

import modelo

app = Flask(__name__)

# rutas de los archivos
CARPETA = os.path.dirname(os.path.abspath(__file__))
NOMBRE_CSV = "datos_hogares.csv"
RUTA_CSV = os.path.join(CARPETA, NOMBRE_CSV)
RUTA_GRAFICA = os.path.join(CARPETA, "static", "grafica.png")

# la grafica usa pyplot, que no es seguro con varios hilos a la vez
CANDADO_GRAFICA = threading.Lock()

# aqui guardamos lo que se va haciendo en cada paso
estado = {"tabla": None, "modelo": None, "prediccion": None}

# pasos del menu: (numero, nombre, funcion de la ruta)
PASOS = [
    (1, "Cargar CSV", "cargar"),
    (2, "Mostrar y analizar datos", "datos"),
    (3, "Entrenar el modelo", "entrenar"),
    (4, "Coeficientes", "coeficientes"),
    (5, "Ingresar hogar nuevo", "nuevo"),
    (6, "Predicción", "prediccion"),
    (7, "Evaluar el modelo", "evaluar"),
    (8, "Gráfica real vs. predicho", "grafica"),
]


def esta_disponible(numero):
    # el paso 1 siempre se puede hacer
    if numero == 1:
        return True
    # los pasos 2 y 3 necesitan el CSV cargado
    if numero in (2, 3):
        return estado["tabla"] is not None
    # del 4 al 8 necesitan el modelo entrenado
    return estado["modelo"] is not None


@app.context_processor
def datos_para_plantillas():
    # estas variables las pueden usar todas las plantillas (el menu lateral)
    return {
        "pasos": PASOS,
        "disponible": esta_disponible,
        "pagina_actual": request.endpoint,
    }


@app.template_filter("fantasma")
def fantasma(valor):
    # cambia cada digito por un 8 (para los numeros apagados del display)
    texto = str(valor)
    resultado = ""
    for letra in texto:
        if letra.isdigit():
            resultado = resultado + "8"
        else:
            resultado = resultado + letra
    # agrega 8 a la izquierda para que se vean digitos apagados detras del numero
    while len(resultado) < 6:
        resultado = "8" + resultado
    return resultado


@app.template_filter("formato")
def formato(valor, decimales=1):
    # si el numero es entero lo muestra sin decimales (4 en vez de 4.0)
    if float(valor) == int(valor):
        return str(int(valor))
    return f"{valor:.{decimales}f}"


SIN_CSV = ("Primero carga el CSV (paso 01). Si el servidor estuvo inactivo, se borró lo que tenía en "
           "memoria: vuelve a cargar los datos y a entrenar el modelo.")


def falta(mensaje, destino, texto_boton):
    # pagina de aviso cuando se entra a un paso sin haber hecho el anterior
    return render_template("aviso.html", mensaje=mensaje, destino=destino, texto_boton=texto_boton)


def falta_modelo():
    # si no hay datos cargados, el servidor se reinicio (en Render se duerme y pierde la memoria)
    if estado["tabla"] is None:
        return falta("No hay un modelo entrenado. Si el servidor estuvo inactivo, se borró lo que tenía "
                     "en memoria: vuelve a cargar los datos (paso 01) y a entrenar el modelo (paso 03).",
                     "cargar", "Ir al paso 01")
    return falta("Primero entrena el modelo (paso 03).", "entrenar", "Ir al paso 03")


def preparar_matriz(matriz, max_filas=6, max_cols=6, decimales=1):
    # recorta la matriz para mostrar solo una parte y la pasa a texto
    filas, columnas = matriz.shape
    recorte = matriz[:max_filas, :max_cols]
    texto = []
    for fila in recorte:
        texto.append([f"{v:.{decimales}f}" for v in fila])
    return {
        "filas": texto,
        "forma": f"{filas} × {columnas}",
        "mas_filas": filas > max_filas,
        "mas_cols": columnas > max_cols,
    }


def predecir_prueba():
    # predice el consumo de los hogares de prueba (el 20 %)
    m = estado["modelo"]
    X_prueba = modelo.armar_X(m["test"])
    Y_prueba = modelo.armar_Y(m["test"])
    Y_pred = X_prueba @ m["beta"]
    return Y_prueba, Y_pred


def hacer_grafica(Y_real, Y_pred, r2):
    with CANDADO_GRAFICA:
        _dibujar_grafica(Y_real, Y_pred, r2)


def _dibujar_grafica(Y_real, Y_pred, r2):
    # grafica de puntos: consumo real (x) contra consumo predicho (y)
    os.makedirs(os.path.dirname(RUTA_GRAFICA), exist_ok=True)
    plt.rcParams["font.family"] = "monospace"
    figura, ejes = plt.subplots(figsize=(7, 5.5))
    figura.patch.set_facecolor("#0a0b0d")
    ejes.set_facecolor("#101216")

    ejes.scatter(Y_real, Y_pred, color="#ffd60a", s=45, label="Hogares de prueba")

    # linea de prediccion perfecta (real = predicho)
    minimo = min(Y_real.min(), Y_pred.min())
    maximo = max(Y_real.max(), Y_pred.max())
    ejes.plot([minimo, maximo], [minimo, maximo], color="#e7e5de", linestyle="--",
              linewidth=1, label="Predicción perfecta")

    ejes.set_xlabel("Consumo real (kWh)", color="#e7e5de")
    ejes.set_ylabel("Consumo predicho (kWh)", color="#e7e5de")
    ejes.set_title(f"Consumo real vs. predicho  (R² = {r2:.3f})", color="#ffd60a")
    ejes.tick_params(colors="#868b95")
    for borde in ejes.spines.values():
        borde.set_color("#262a31")
    ejes.grid(color="#262a31", linewidth=0.6)
    ejes.legend(facecolor="#101216", edgecolor="#262a31", labelcolor="#e7e5de")

    figura.tight_layout()
    figura.savefig(RUTA_GRAFICA, dpi=130, facecolor=figura.get_facecolor())
    plt.close(figura)


@app.route("/")
def inicio():
    return render_template("index.html")


# ---------- paso 1: cargar CSV ----------
@app.route("/cargar", methods=["GET", "POST"])
def cargar():
    mensaje = None
    if request.method == "POST":
        if not os.path.exists(RUTA_CSV):
            # si falta el CSV lo generamos con generar_datos.py (misma semilla, mismos datos)
            try:
                runpy.run_path(os.path.join(CARPETA, "generar_datos.py"))
            except Exception:
                pass
        if not os.path.exists(RUTA_CSV):
            mensaje = "No se encontró datos_hogares.csv. Ejecuta primero: python generar_datos.py"
        else:
            tabla = pd.read_csv(RUTA_CSV)
            columnas_necesarias = modelo.COLUMNAS_X + [modelo.COLUMNA_Y]
            if not all(c in tabla.columns for c in columnas_necesarias):
                mensaje = "El CSV no tiene las columnas esperadas."
            else:
                estado["tabla"] = tabla
                # si se cargan datos nuevos, se borra el modelo anterior
                estado["modelo"] = None
                estado["prediccion"] = None
    return render_template("cargar.html", mensaje=mensaje, tabla=estado["tabla"], archivo=NOMBRE_CSV)


# ---------- paso 2: mostrar y analizar datos ----------
@app.route("/datos")
def datos():
    tabla = estado["tabla"]
    if tabla is None:
        return falta(SIN_CSV, "cargar", "Ir al paso 01")

    # primeras 10 filas
    primeras = tabla.head(10)

    # promedio, minimo y maximo de cada columna
    resumen = []
    for columna in tabla.columns:
        resumen.append({
            "nombre": columna,
            "media": tabla[columna].mean(),
            "minimo": tabla[columna].min(),
            "maximo": tabla[columna].max(),
        })

    # matriz de correlacion
    correlacion = tabla.corr()
    nombres = list(tabla.columns)
    filas_correlacion = []
    for fila in nombres:
        valores = [correlacion.loc[fila, columna] for columna in nombres]
        filas_correlacion.append((fila, valores))

    return render_template("datos.html", primeras=primeras, resumen=resumen,
                           nombres=nombres, filas_correlacion=filas_correlacion,
                           total=len(tabla))


# ---------- paso 3: entrenar el modelo ----------
@app.route("/entrenar", methods=["GET", "POST"])
def entrenar():
    if estado["tabla"] is None:
        return falta(SIN_CSV, "cargar", "Ir al paso 01")

    if request.method == "POST":
        estado["modelo"] = modelo.entrenar(estado["tabla"])
        estado["prediccion"] = None
        return redirect(url_for("entrenar"))

    m = estado["modelo"]
    matrices = None
    if m is not None:
        matrices = {
            "X": preparar_matriz(m["X"]),
            "Xt": preparar_matriz(m["Xt"]),
            "Y": preparar_matriz(m["Y"]),
            "XtX": preparar_matriz(m["XtX"]),
            "XtY": preparar_matriz(m["XtY"]),
            "beta": preparar_matriz(m["beta"], decimales=4),
        }
    return render_template("entrenar.html", m=m, matrices=matrices)


# ---------- paso 4: coeficientes ----------
@app.route("/coeficientes")
def coeficientes():
    if estado["modelo"] is None:
        return falta_modelo()

    beta = estado["modelo"]["beta"].ravel()

    # frase de interpretacion de cada coeficiente
    def frase(valor, sujeto):
        verbo = "suma" if valor >= 0 else "resta"
        return f"{sujeto} {verbo} {abs(valor):.2f} kWh al mes."

    coefs = [
        {"simbolo": "β₀", "nombre": "término independiente", "valor": beta[0],
         "texto": f"Consumo base de {beta[0]:.2f} kWh cuando todas las variables valen 0. "
                  "Es un punto de partida matemático, no un hogar real."},
        {"simbolo": "β₁", "nombre": "personas", "valor": beta[1],
         "texto": frase(beta[1], "Cada persona adicional")},
        {"simbolo": "β₂", "nombre": "dispositivos electrónicos", "valor": beta[2],
         "texto": frase(beta[2], "Cada dispositivo electrónico adicional")},
        {"simbolo": "β₃", "nombre": "horas de aire acondicionado", "valor": beta[3],
         "texto": frase(beta[3], "Cada hora diaria adicional de aire acondicionado")},
        {"simbolo": "β₄", "nombre": "temperatura", "valor": beta[4],
         "texto": frase(beta[4], "Cada grado adicional de temperatura")},
    ]

    # ecuacion final con los numeros
    nombres = ["personas", "dispositivos", "horas A/C", "temperatura"]
    ecuacion = f"consumo = {beta[0]:.2f}"
    for i in range(4):
        signo = "+" if beta[i + 1] >= 0 else "-"
        ecuacion = ecuacion + f" {signo} {abs(beta[i + 1]):.2f}·{nombres[i]}"

    return render_template("coeficientes.html", coefs=coefs, ecuacion=ecuacion)


# ---------- paso 5: ingresar hogar nuevo ----------
@app.route("/nuevo")
def nuevo():
    if estado["modelo"] is None:
        return falta_modelo()
    valores = {"personas": 4, "dispositivos": 12, "horas": 5, "temperatura": 28}
    return render_template("nuevo.html", valores=valores, error=None)


# ---------- paso 6: predecir ----------
@app.route("/predecir", methods=["GET", "POST"])
def predecir():
    if estado["modelo"] is None:
        return falta_modelo()

    # si entran por la barra de direcciones (GET) los mandamos al formulario
    if request.method == "GET":
        return redirect(url_for("nuevo"))

    # leemos los 4 datos del formulario
    try:
        entrada = [
            float(request.form["personas"]),
            float(request.form["dispositivos"]),
            float(request.form["horas"]),
            float(request.form["temperatura"]),
        ]
    except (ValueError, KeyError):
        return render_template("nuevo.html", valores=request.form,
                               error="Escribe solo números en todos los campos.")

    # nan o infinito tambien se convierten a float pero no sirven
    if not all(math.isfinite(v) for v in entrada):
        return render_template("nuevo.html", valores=request.form,
                               error="Escribe solo números en todos los campos.")

    # personas, dispositivos y horas no pueden ser negativos
    if entrada[0] < 0 or entrada[1] < 0 or entrada[2] < 0:
        return render_template("nuevo.html", valores=request.form,
                               error="Personas, dispositivos y horas no pueden ser negativos.")

    beta = estado["modelo"]["beta"].ravel()
    resultado = modelo.predecir(estado["modelo"]["beta"], entrada)
    if not math.isfinite(resultado):
        return render_template("nuevo.html", valores=request.form,
                               error="Esos números son demasiado grandes. Escribe valores más realistas.")

    # cuanto aporta cada termino a la prediccion
    nombres = ["β₀ (término independiente)", "β₁ · personas", "β₂ · dispositivos",
               "β₃ · horas de A/C", "β₄ · temperatura"]
    valores_x = [1] + entrada
    terminos = []
    for i in range(5):
        terminos.append({
            "nombre": nombres[i],
            "beta": beta[i],
            "valor": valores_x[i],
            "aporte": beta[i] * valores_x[i],
        })

    # avisamos si algun dato esta fuera del rango con el que se entreno
    avisos = []
    etiquetas = ["personas", "dispositivos", "horas de A/C", "temperatura"]
    for i, columna in enumerate(modelo.COLUMNAS_X):
        minimo = estado["tabla"][columna].min()
        maximo = estado["tabla"][columna].max()
        if entrada[i] < minimo or entrada[i] > maximo:
            avisos.append(f"{etiquetas[i]}: {entrada[i]:g} está fuera del rango de los datos "
                          f"({minimo:g} a {maximo:g}).")

    estado["prediccion"] = {
        "entrada": entrada,
        "resultado": resultado,
        "terminos": terminos,
        "avisos": avisos,
    }
    return redirect(url_for("prediccion"))


@app.route("/prediccion")
def prediccion():
    if estado["modelo"] is None:
        return falta_modelo()
    return render_template("prediccion.html", p=estado["prediccion"])


# ---------- paso 7: evaluar el modelo ----------
@app.route("/evaluar")
def evaluar():
    if estado["modelo"] is None:
        return falta_modelo()

    Y_real, Y_pred = predecir_prueba()
    mae, rmse, r2 = modelo.calcular_metricas(Y_real, Y_pred)

    # primeras 10 filas de prueba: real, predicho y error
    filas = []
    for i in range(min(10, len(Y_real))):
        real = Y_real[i][0]
        predicho = Y_pred[i][0]
        filas.append((real, predicho, real - predicho))

    return render_template("evaluar.html", mae=mae, rmse=rmse, r2=r2,
                           filas=filas, total=len(Y_real))


# ---------- paso 8: grafica ----------
@app.route("/grafica")
def grafica():
    if estado["modelo"] is None:
        return falta_modelo()

    Y_real, Y_pred = predecir_prueba()
    mae, rmse, r2 = modelo.calcular_metricas(Y_real, Y_pred)
    hacer_grafica(Y_real.ravel(), Y_pred.ravel(), r2)

    # el numero al final del nombre evita que el navegador use una imagen vieja
    return render_template("grafica.html", version=int(time.time()), r2=r2, total=len(Y_real))


if __name__ == "__main__":
    # en Render la variable PORT existe; en tu computadora no (ahi se usa 5001 porque
    # en Mac el 5000 suele estar ocupado por AirPlay)
    produccion = "PORT" in os.environ
    puerto = int(os.environ.get("PORT", 5001))
    host = "0.0.0.0" if produccion else "127.0.0.1"
    # el modo debug solo se activa en local, o con FLASK_DEBUG=1
    debug = os.environ.get("FLASK_DEBUG", "0" if produccion else "1") == "1"
    app.run(host=host, port=puerto, debug=debug)
