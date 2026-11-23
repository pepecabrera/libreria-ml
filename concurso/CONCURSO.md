# 🏆 Resolución Oposiciones Machine Learning (Top 2)

**Candidato:** Pepe Cabrera
**Metodología:** Grid Search (Búsqueda en Cuadrícula) y Detección de Trampas.

## 1. Planteamiento del Problema
El objetivo de este reto consiste en comparar configuraciones de k-NN para dos datasets mediante **validación cruzada de 5 bolsas (m=5) sin barajado previo**. Se varían $k$, la distancia, el preprocesamiento y la fracción de atributos seleccionados. El voto y la media son uniformes, por lo que no se busca un esquema de pesos.

## 2. Análisis de Datos y Evasión de "Trampas" (Data Quirks)

Antes de automatizar la búsqueda, se realizó un análisis exploratorio que reveló dos puntos críticos que destruirían cualquier modelo no supervisado:

1. **La trampa del ID en Breast Cancer (WDBC):**
   La columna 0 del dataset contiene identificadores únicos de paciente (ej. `842302`). En un algoritmo basado en cálculo espacial como el k-NN (Distancia Euclídea), dejar una variable con magnitudes de cientos de miles distorsiona el hiperplano y anula el efecto de las características morfológicas reales del tumor.
   * **Solución implementada:** Se ha forzado la carga con `indice_objetivo=1` y posteriormente se ha utilizado el método `eliminar_atributos([0])` para purgar el ID del espacio dimensional.

2. **Magnitudes dispersas en Concrete Strength:**
   Los componentes del hormigón oscilan entre valores de rango 0-10 (Superplastificantes) hasta 300-500 (Cemento). Sin normalización, el kNN actuaría como un modelo de 1 sola variable (Cemento).
   * **Solución implementada:** Se han inyectado de forma programática métodos de estandarización Z-Score y MaxMin.

## 3. Estrategia de Evaluación: El problema del "No Shuffle"
Al aplicar Validación Cruzada $m=5$ sin `shuffle`, los datos se parten en bloques secuenciales. Si el dataset está ordenado por clases (por ejemplo, todos los tumores malignos al principio y los benignos al final), el modelo entrenará con una clase e intentará predecir una clase que jamás ha visto, hundiendo el Accuracy.

Para mitigar esto, hemos implementado un **Grid Search de Fuerza Bruta**. El script anida bucles para probar matemáticamente todas las combinaciones posibles:
* **$K$:** `[1, 3, 5, 7, 9, 11, 15, 19, 23]`
* **Distancia:** `[Euclídea, Manhattan]`
* **Preprocesado:** `[Sin Normalizar, MaxMin, Z-Score]`
* **Fracción de atributos:** `[1.0, 0.75, 0.5]`

Al forzar el recorrido completo, delegamos en la potencia de cómputo la búsqueda del "espacio latente" menos afectado por la partición secuencial del archivo, asegurando el menor MSE (Error Cuadrático Medio) en regresión y el mayor Accuracy en clasificación.

## 4. Conclusión
El enfoque adoptado garantiza matemáticamente haber encontrado el óptimo global dentro del espacio de búsqueda acotado, aplicando principios de Programación Orientada a Objetos para mantener el pipeline limpio, escalable y resistente a fallos.

## 5. Enfoque Montecarlo
Otro enfoque alternativo basado en la aleatoriedad para comparar resultados con nuestro enfoque original.
