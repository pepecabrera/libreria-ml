# Librería educativa de machine learning

Proyecto individual de clase de Pepe Cabrera. Implementé desde cero en Python el núcleo de la librería: kNN para clasificación y regresión, clasificación por centroides, regresión lineal por descenso de gradiente y herramientas para cargar, preprocesar y evaluar datos. Los datasets y los enunciados de la asignatura no son de mi autoría.

## Ejecutar el experimento de Wine

Desde la raíz del repositorio, con **Python 3.13**, ejecuta el mismo comando que usa [GitHub Actions](.github/workflows/ci.yml). No requiere instalar paquetes:

```bash
python -m examples.wine
```

El experimento divide las 178 observaciones de Wine en 142 de desarrollo y 36 de prueba estratificada, con semilla `42`. Elige `k` por validación cruzada de cinco bolsas en desarrollo y ajusta el escalado Z-score solo con el entrenamiento de cada bolsa. Después reentrena con todo el conjunto de desarrollo y compara kNN con una línea base que siempre predice su clase mayoritaria (clase `2`). Ambos modelos se evalúan en las mismas 36 muestras reservadas.

| Modelo | Accuracy en prueba | Accuracy balanceada en prueba |
| --- | ---: | ---: |
| Clase mayoritaria de desarrollo | 38,89 % | 33,33 % |
| kNN (`k=7`) | 94,44 % | 95,24 % |

La salida incluye también el soporte y el *recall* de las tres clases. La *accuracy* balanceada es el promedio del *recall* por clase. Los resultados corresponden a una sola partición de un dataset pequeño: no muestran cuánto cambiarían con otra partición ni garantizan rendimiento en otros datasets.

**Puntuación interna y prueba final:** el 97,24 % de *accuracy* media en validación cruzada se usó para elegir `k`; no es la métrica final. La tabla muestra solo la prueba reservada de Wine. No corresponde al antiguo concurso de clase, que ya no forma parte de la versión actual y cuyos resultados no se presentan como una evaluación independiente.

## Estructura del repositorio

```text
libreria-ml/
├── src/                          # Núcleo de la librería
│   ├── dataset.py                # Colecciones de registros y subconjuntos
│   ├── factoria.py               # Lectores de CSV y Excel
│   ├── modelos.py                # kNN, centroides y regresión lineal
│   ├── preprocesado.py           # Escalado MinMax y Z-score
│   ├── registro.py               # Observaciones y cálculo de distancias
│   └── validacion.py             # Particiones, métricas y selección por Pearson
├── examples/                     # Experimentos y demos ejecutables
│   └── wine.py                   # Experimento principal con línea base
├── tests/                        # Pruebas automatizadas con unittest
├── datos/                        # Datos usados por el experimento y las demos
│   ├── wine/                     # Dataset Wine y su atribución
│   └── otros datasets
├── .github/workflows/
│   └── ci.yml                    # Ejecuta las pruebas y el ejemplo de Wine
└── requirements-excel.txt        # Dependencias opcionales para leer Excel
```

## Código, pruebas y demos

`src/` contiene los modelos, los lectores CSV y Excel, el escalado MinMax y Z-score, y la selección de atributos por Pearson. El validador ajusta las transformaciones solo con los datos de entrenamiento de cada partición.

```bash
python -m unittest discover -s tests -v
```

Las [demos de clase](examples/) están separadas del ejemplo principal. Algunas requieren archivos locales que no se incluyen en un clon nuevo; consulta el [inventario de datos](datos/README.md). Para usar los lectores de Excel (`openpyxl` para `.xlsx` y `xlrd` para `.xls`), instala las dependencias opcionales:

```bash
python -m pip install -r requirements-excel.txt
```

## Datos, materiales y licencias

El [inventario de datos](datos/README.md) documenta las fuentes, las condiciones de redistribución y qué materiales quedaron fuera. El código no tiene una licencia explícita; las licencias de los datasets no se aplican al código.
