import numpy as np

# columnas que usa el modelo
COLUMNAS_X = ["personas", "dispositivos", "horas_aire_acondicionado", "temperatura"]
COLUMNA_Y = "consumo_kwh"


def dividir_datos(tabla, proporcion_entrenamiento=0.8, semilla=42):
    # mezclamos los numeros de fila al azar (con semilla para que sea repetible)
    generador = np.random.default_rng(semilla)
    indices = generador.permutation(len(tabla))

    # 80 % de las filas para entrenar y 20 % para probar
    corte = int(len(tabla) * proporcion_entrenamiento)
    entrenamiento = tabla.iloc[indices[:corte]]
    prueba = tabla.iloc[indices[corte:]]
    return entrenamiento, prueba


def armar_X(tabla):
    # pasamos las 4 variables a una matriz de numeros
    datos = tabla[COLUMNAS_X].to_numpy(dtype=float)

    # columna de unos para poder calcular el termino independiente (beta0)
    unos = np.ones((datos.shape[0], 1))

    # X queda de tamano n x 5
    X = np.hstack([unos, datos])
    return X


def armar_Y(tabla):
    # vector columna con el consumo real, tamano n x 1
    return tabla[[COLUMNA_Y]].to_numpy(dtype=float)


def entrenar(tabla):
    entrenamiento, prueba = dividir_datos(tabla)

    # matrices del modelo
    X = armar_X(entrenamiento)
    Y = armar_Y(entrenamiento)

    # ecuacion normal: beta = (Xt X)^-1 Xt Y
    Xt = X.T
    XtX = Xt @ X
    XtY = Xt @ Y
    beta = np.linalg.inv(XtX) @ XtY

    # guardamos todo para mostrarlo en pantalla
    return {
        "train": entrenamiento,
        "test": prueba,
        "X": X,
        "Xt": Xt,
        "Y": Y,
        "XtX": XtX,
        "XtY": XtY,
        "beta": beta,
    }


def predecir(beta, entrada):
    # entrada = [personas, dispositivos, horas, temperatura]
    # le agregamos el 1 al inicio para multiplicar por beta0
    x = np.array([[1] + list(entrada)], dtype=float)

    # (1 x 5) @ (5 x 1) = (1 x 1)
    resultado = x @ beta
    return float(resultado[0][0])


def calcular_metricas(Y_real, Y_pred):
    errores = Y_real - Y_pred

    # MAE: promedio de los errores sin signo
    mae = np.mean(np.abs(errores))

    # RMSE: raiz del promedio de los errores al cuadrado
    rmse = np.sqrt(np.mean(errores ** 2))

    # R2: 1 - (suma de errores al cuadrado / variacion total)
    suma_errores = np.sum(errores ** 2)
    variacion_total = np.sum((Y_real - np.mean(Y_real)) ** 2)
    r2 = 1 - suma_errores / variacion_total

    return float(mae), float(rmse), float(r2)
