import os
import numpy as np
import pandas as pd

# cantidad de hogares ficticios (cada fila es un hogar en un mes)
N = 150

# la semilla hace que siempre salgan los mismos datos al ejecutar el script
generador = np.random.default_rng(2026)

# X1: personas que viven en el hogar (de 1 a 7, lo mas comun es 3 o 4)
personas = generador.choice(
    [1, 2, 3, 4, 5, 6, 7],
    size=N,
    p=[0.08, 0.17, 0.22, 0.25, 0.15, 0.08, 0.05],
)

# X2: dispositivos electronicos (celulares, laptops, PCs, monitores, TVs)
# crecen con la cantidad de personas, mas un poco de ruido
dispositivos = np.round(4 + 1.8 * personas + generador.normal(0, 3, N))
dispositivos = np.clip(dispositivos, 3, 25)

# X4: temperatura promedio del mes en grados centigrados
temperatura = generador.normal(25, 4.5, N)
temperatura = np.round(np.clip(temperatura, 14, 35), 1)

# X3: horas de aire acondicionado por dia
# si hace mas calor se usa mas, pero le sumamos ruido para que
# no dependa solo de la temperatura (no queden casi 100 % correlacionadas)
horas = 0.8 * (temperatura - 20) + generador.normal(0, 3, N)
horas = np.round(np.clip(horas, 0, 14), 1)

# Y: consumo mensual en kWh
# esta es la formula "real" con la que se inventan los datos:
# consumo = 25 + 14*personas + 5*dispositivos + 20*horas + 1.5*temperatura + ruido
ruido = generador.normal(0, 12, N)
consumo = 25 + 14 * personas + 5 * dispositivos + 20 * horas + 1.5 * temperatura + ruido
consumo = np.round(np.clip(consumo, 30, None), 1)

# armamos la tabla con las 5 columnas
tabla = pd.DataFrame({
    "personas": personas,
    "dispositivos": dispositivos.astype(int),
    "horas_aire_acondicionado": horas,
    "temperatura": temperatura,
    "consumo_kwh": consumo,
})

# guardamos el CSV en la misma carpeta del script
carpeta = os.path.dirname(os.path.abspath(__file__))
ruta = os.path.join(carpeta, "datos_hogares.csv")
tabla.to_csv(ruta, index=False)

print("Se generaron", len(tabla), "hogares ficticios")
print("Archivo guardado en:", ruta)
print()
print(tabla.head())
print()
correlacion = tabla["temperatura"].corr(tabla["horas_aire_acondicionado"])
print("Correlacion entre temperatura y horas de aire:", round(correlacion, 2))
