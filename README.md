# 🧠 Librería de Machine Learning (Desde Cero)

Este proyecto es una implementación educativa y orientada a objetos de algoritmos clásicos de Machine Learning construida completamente desde cero en Python. El objetivo de esta librería es comprender las matemáticas y la lógica interna ("bajo el capó") de los modelos de aprendizaje automático, sin depender de librerías externas de ML como Scikit-Learn.

El proyecto soporta tanto tareas de **Clasificación** como de **Regresión**, e incluye herramientas para el preprocesamiento de datos, entrenamiento de modelos, evaluación de métricas y selección de atributos.

---

## 🏗️ Arquitectura del Proyecto

El código está estructurado en módulos altamente cohesivos utilizando los principios de la Programación Orientada a Objetos (Herencia, Polimorfismo y Clases Abstractas).

### 1. `registro.py` (Unidad básica de datos)
Gestiona la información a nivel de fila individual.
* **`Registro` (Abstracta):** Almacena una lista de números reales (`atributos`) y provee métodos matemáticos como `distancia_euclidea` y `distancia_manhattan`.
* **`RegistroClasificacion`:** Hereda de `Registro`. Su `objetivo` (etiqueta a predecir) es una cadena de texto (`str`), ej: `"Iris-setosa"`.
* **`RegistroRegresion`:** Hereda de `Registro`. Su `objetivo` es un valor numérico continuo (`float`), ej: `250000.0` (precio).

### 2. `dataset.py` (Gestor de colecciones)
Gestiona conjuntos de registros y operaciones sobre la matriz de datos.
* **`DataSet` (Abstracta):** Almacena una lista de objetos `Registro`.
  * *Funciones clave:* `crear_subconjunto()` para dividir datos, `calcular_min_max()` (hace la transpuesta de la matriz para hallar extremos) y `eliminar_atributos()` para descartar columnas basura.
* **`DataSetClasificacion` / `DataSetRegresion`:** Implementaciones específicas según la naturaleza del problema.

### 3. `preprocesado.py` (Normalización de datos)
Evita que atributos con escalas grandes (ej. sueldo) dominen sobre atributos pequeños (ej. edad).
* **`Preprocesamiento` (Abstracta):** Define el contrato con `ajustar(dataset)` y `transformar_dataSet(dataset)`.
* **`NormalizadorMaxMin`:** Escala todos los valores al rango [0, 1].
* **`NormalizadorZ_Score`:** Estandariza los datos a media 0 y desviación típica 1 (ideal para outliers).
> **Nota de diseño:** Los normalizadores previenen el *Data Leakage* aplicando las transformaciones al conjunto de Test basándose exclusivamente en los parámetros aprendidos del conjunto de Train.

### 4. `modelos.py` (Algoritmos Predictivos)
El núcleo inteligente del proyecto. Todos heredan de la clase base `Modelo` e implementan los métodos `entrenar(dataset)` y `predecir(registro)`.
* **`Clasificador_kNN` y `Regresor_kNN`:** Algoritmo de los k-Vecinos Más Cercanos. (Incluye un mecanismo de seguridad para que un registro se omita a sí mismo si se evalúa dentro de su propio conjunto de entrenamiento).
* **`Clasificador_Centroide`:** Calcula el punto medio (centroide) de cada clase y clasifica según la distancia mínima a estos centros.
* **`RegresionLinealMultiple`:** Implementa una red neuronal de una sola capa mediante el cálculo de pesos (`w`) y sesgo (`b`), optimizados iterativamente para predecir valores continuos.

### 5. `validacion.py` (Evaluación y Optimización)
Evalúa cómo de buenos son los modelos prediciendo datos que nunca han visto.
* **Evaluadores (`ValidacionClasificacion` / `ValidacionRegresion`):** * *Funciones clave:* `validacion_simple` (Hold-out test) y `validacion_cruzada` (k-Fold).
  * *Métricas dinámicas soportadas:* Accuracy y Error Rate (Clasificación) | MAE, MSE y RMSE (Regresión).
* **`SeleccionAtributos`:** Ordena los atributos por su correlación absoluta de Pearson con el objetivo. Si se pasa al validador junto al escalador, ambos se ajustan solo con entrenamiento, también dentro de cada bolsa.

Para buscar hiperparámetros, reserva primero una prueba final. Compara los candidatos mediante validación cruzada en el conjunto de desarrollo, con las mismas bolsas contiguas para todos. Esa CV es una **puntuación interna**; ajusta de nuevo solo la configuración elegida con todo el conjunto de desarrollo y evalúala una vez en la prueba reservada.

---

## Experimento reproducible: Wine

Desde la raíz del repositorio, ejecuta el ejemplo principal. No requiere instalar paquetes externos:

```bash
python3 -m examples.wine
```

El ejemplo reserva aproximadamente el 20 % de cada clase con semilla `42`. Elige `k` por validación cruzada en el resto de los datos y ajusta el escalador Z-score solo con entrenamiento en cada bolsa. En la prueba reservada compara kNN con una línea base que siempre predice la clase mayoritaria del conjunto de desarrollo. Muestra el soporte por clase, el *accuracy*, el *accuracy* balanceado (promedio del *recall* por clase) y el *recall* de cada clase. Es una sola partición de un dataset pequeño; no mide la variación entre distintas particiones.

Fuente: [Wine de UCI, CC BY 4.0](datos/wine/README.md).

Las [demos de clase](examples/) se conservan por separado. Para ejecutar las pruebas automatizadas:

```bash
python3 -m unittest discover -s tests -v
```

## 📂 Estructura del Proyecto

```text
📁 Proyecto_Machine_Learning/
├── 📄 README.md                                    # Documentación principal del proyecto
│
├── 📁 datos/                                       # Directorio de datasets (archivos de datos)
│   ├── 📁 BreastCancerWisconsinDiagnostic/         # Daros de cancer de mama en Wisconsin (concurso)
│   ├── 📁 ConcreteCompressiveStrength/             # Datos de la fuerza compresiva del cemento (concurso)             
│   ├── 📁 wine/                                    # Datos en bruto de vinos (Clasificación)
│   ├── 📄 BostonHousing.csv                        # Precios de casas en Boston (Regresión)
│   ├── 📄 diabetes.csv                             # Diagnóstico de pacientes (Clasificación)
│   └── 📄 iris.xlsx                                # Especies de flores (Clasificación)
│
├── 📁 src/                                         # Código fuente (El "motor" de la librería)
│   ├── 📄 factoria.py                              # Lector de datos tanto CSV como Excel             
│   ├── 📄 dataset.py                               # Manejo de tablas, particiones y filtros
│   ├── 📄 modelos.py                               # Algoritmos predictivos (kNN, Centroide, RLM)
│   ├── 📄 preprocesado.py                          # Normalizadores de escala (MaxMin, Z-Score)
│   ├── 📄 registro.py                              # Estructura de filas (Clasificación y Regresión)
│   └── 📄 validacion.py                            # Evaluación de métricas, k-Fold y Pearson
│
├── examples/                                      # Experimento Wine y demos de clase
└── tests/                                         # Pruebas automatizadas con unittest
```
