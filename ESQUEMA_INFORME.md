# Esquema del informe técnico

Predicción del consumo mensual de electricidad de un hogar con regresión lineal múltiple

## Portada
Título, nombre, materia (Álgebra Lineal), docente y fecha. Portada simple: texto negro, sin cuadros ni marcos.

## 1. Introducción y objetivo
- Qué problema se resuelve: estimar los kWh que consume un hogar en un mes.
- Objetivo general: desarrollar en Python un sistema que prediga el consumo con regresión lineal múltiple, aplicando Álgebra Lineal.
- Objetivos específicos: generar datos ficticios, calcular los coeficientes con la ecuación normal, evaluar el modelo con MAE, RMSE y R², y construir una interfaz web para usarlo.

## 2. Descripción de los datos
- Tabla de variables con su unidad:

| Variable | Descripción | Unidad |
|----------|-------------|--------|
| X1 personas | Personas que viven en el hogar | personas |
| X2 dispositivos | Celulares, laptops, PCs, monitores y TVs | cantidad |
| X3 horas_aire_acondicionado | Uso diario del aire acondicionado | horas por día |
| X4 temperatura | Temperatura promedio del mes | °C |
| Y consumo_kwh | Consumo eléctrico del mes | kWh |

- Cómo se generaron los datos (`generar_datos.py`): 150 hogares ficticios (cada fila es un hogar en un mes), una fórmula con ruido aleatorio y una semilla fija para poder repetirlos. Aclarar que los datos son ficticios, como pide el docente.
- Cómo se relacionan X2 con X1 y X3 con X4, y por qué se agregó ruido.
- Captura de las primeras filas, tabla de promedio, mínimo y máximo, y la matriz de correlación (captura 06 y 07).

## 3. Marco matemático
- Modelo: `consumo = β₀ + β₁·personas + β₂·dispositivos + β₃·horas + β₄·temperatura + error`.
- Forma matricial: `Y = Xβ + error`.
- Idea de la ecuación normal: se buscan los β que minimizan la suma de los errores al cuadrado, lo que lleva a `XᵀX β = XᵀY` y de ahí a `β = (XᵀX)⁻¹ XᵀY`.
- Condición para que exista la inversa: las columnas de X deben ser linealmente independientes (determinante de XᵀX distinto de cero).

## 4. Identificación de las matrices
Explicar cada una con su tamaño (n = 120 hogares de entrenamiento) y un ejemplo con las primeras filas que muestra el programa (capturas 08 y 09):

| Matriz | Qué es | Tamaño |
|--------|--------|--------|
| X | Datos de los hogares, con una primera columna de unos para β₀ | 120 × 5 |
| Xᵀ | Transpuesta de X | 5 × 120 |
| Y | Consumo real de cada hogar | 120 × 1 |
| β | Coeficientes del modelo | 5 × 1 |

También explicar los productos XᵀX (5 × 5) y XᵀY (5 × 1), y por qué la primera columna de X es de unos.

## 5. Implementación
- Estructura de archivos: `generar_datos.py`, `modelo.py`, `app.py`, plantillas HTML y `estilo.css`.
- Qué hace cada función de `modelo.py`: `dividir_datos`, `armar_X`, `armar_Y`, `entrenar`, `predecir`, `calcular_metricas`.
- Las cuatro líneas de la ecuación normal en NumPy, sin scikit-learn.
- División 80 % entrenamiento (120 hogares) y 20 % prueba (30 hogares), con semilla fija.
- Funcionamiento de la interfaz: los 8 pasos del menú.

## 6. Resultados
- Coeficientes obtenidos (captura 10) y la ecuación final con números.
- Interpretación de cada coeficiente con ejemplos, por ejemplo "cada persona adicional suma X kWh al mes".
- Comparación con la fórmula usada para generar los datos (25, 14, 5, 20 y 1.5). Los valores calculados se parecen a los reales y las diferencias vienen del ruido. Los que más se alejan son β₀ y el de la temperatura. Este último se puede explicar porque temperatura y horas de aire están correlacionadas (0.72), así que el modelo reparte entre las dos variables parte del efecto del calor.
- Por qué β₀ no es un hogar real.

## 7. Evaluación del modelo
- Definición de MAE, RMSE y R² con su fórmula.
- Valores obtenidos con los 30 hogares de prueba (captura 13). Con los datos entregados salen MAE ≈ 11.4 kWh, RMSE ≈ 14.5 kWh y R² ≈ 0.977.
- Interpretación: el error promedio frente al consumo típico (alrededor de 260 kWh) y qué significa que R² sea cercano a 1.
- Por qué se evalúa con datos que el modelo no vio.

## 8. Gráficas
- Consumo real vs. predicho (`static/grafica.png`, captura 14).
- Interpretación: los puntos cerca de la línea punteada son buenas predicciones, y qué hogares se alejan más.
- Opcional: una gráfica de errores (real − predicho).

## 9. Demostración de predicción
- Un hogar nuevo con sus datos y el desglose de la predicción (capturas 11, 12 y 15).
- Qué pasa si los datos están fuera del rango con el que se entrenó el modelo.

## 10. Conclusiones y limitaciones
- El modelo cumple el objetivo y las matrices permiten calcular los coeficientes sin librerías de aprendizaje automático.
- Limitaciones: los datos son ficticios, el modelo asume una relación lineal, hay correlación entre algunas variables y solo se usan 150 hogares.
- Mejoras posibles: usar datos reales anonimizados y autorizados, agregar más variables o probar relaciones no lineales.

## Anexos
- Código fuente completo.
- `datos_hogares.csv`.
- Todas las capturas de ejecución.

---

# Preguntas que puede hacer el docente

**¿Por qué la matriz X lleva una columna de unos?**
Para que el modelo tenga un término independiente β₀. Sin esa columna la recta tendría que pasar por el origen.

**¿Qué pasa si XᵀX no tiene inversa?**
Ocurre cuando alguna variable es combinación lineal de otras (por ejemplo, si una columna se repite). Entonces `np.linalg.inv` falla y hay que quitar la variable redundante.

**¿Por qué se divide en 80 % y 20 %?**
Para medir el modelo con hogares que no usó al entrenar. Si se midiera con los mismos datos del entrenamiento, el resultado sería demasiado optimista.

**¿Qué significa un coeficiente?**
Cuánto cambia el consumo cuando esa variable sube una unidad y las demás se quedan iguales.

**¿Qué diferencia hay entre MAE y RMSE?**
El MAE es el error promedio sin signo. El RMSE eleva los errores al cuadrado antes de promediar, así que castiga más los errores grandes.

**¿Por qué los datos son ficticios?**
Lo exige el docente, y además permite saber la fórmula real y comprobar que el modelo recupera coeficientes parecidos.

**¿Dónde está el Álgebra Lineal en el código?**
En `modelo.py`: la transpuesta `X.T`, los productos de matrices con `@` y la inversa con `np.linalg.inv`.
