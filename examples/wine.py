"""Compare kNN with a majority-class baseline on the UCI Wine dataset."""

from collections import Counter, defaultdict
from pathlib import Path
from random import Random

from src.factoria import FactoriaCSV
from src.modelos import Clasificador_kNN
from src.preprocesado import NormalizadorZ_Score
from src.validacion import ValidacionClasificacion


RUTA_WINE = Path(__file__).resolve().parents[1] / "datos" / "wine" / "wine.data"
SEMILLA = 42
FRACCION_PRUEBA = 0.2
VALORES_K = (1, 3, 5, 7, 9)


def separar_estratificado(datos, fraccion_prueba, semilla):
    """Reserva aproximadamente la misma fraccion de cada clase."""
    if not 0 < fraccion_prueba < 1:
        raise ValueError("La fraccion de prueba debe estar entre 0 y 1.")

    grupos = defaultdict(list)
    for registro in datos.registros:
        grupos[registro.objetivo].append(registro)

    rng = Random(semilla)
    desarrollo, prueba = [], []
    for clase in sorted(grupos):
        registros = grupos[clase]
        if len(registros) < 2:
            raise ValueError(f"La clase {clase} necesita al menos dos registros.")
        rng.shuffle(registros)
        n_prueba = min(len(registros) - 1, max(1, round(len(registros) * fraccion_prueba)))
        prueba.extend(registros[:n_prueba])
        desarrollo.extend(registros[n_prueba:])

    rng.shuffle(desarrollo)
    rng.shuffle(prueba)
    return datos.crear_subconjunto(desarrollo), datos.crear_subconjunto(prueba)


def main():
    datos = FactoriaCSV.crear_dataset_clasificacion(
        RUTA_WINE, indice_objetivo=0, tiene_cabecera=False,
    )
    desarrollo, prueba = separar_estratificado(datos, FRACCION_PRUEBA, SEMILLA)
    validador = ValidacionClasificacion()
    soporte_desarrollo = Counter(reg.objetivo for reg in desarrollo.registros)
    clases = sorted(soporte_desarrollo)
    mayoritaria = max(clases, key=soporte_desarrollo.__getitem__)

    resultados_cv = {
        k: validador.validacion_cruzada(
            Clasificador_kNN(k), desarrollo, m=5, normalizador=NormalizadorZ_Score(),
        )
        for k in VALORES_K
    }
    mejor_k = max(VALORES_K, key=resultados_cv.__getitem__)
    pred_knn = validador.predecir_particion(
        Clasificador_kNN(mejor_k), desarrollo, prueba, NormalizadorZ_Score(),
    )
    reales = [reg.objetivo for reg in prueba.registros]
    soporte_prueba = Counter(reales)

    print(f"Wine | semilla={SEMILLA} | prueba estratificada={FRACCION_PRUEBA:.0%}")
    print(f"Desarrollo: {len(desarrollo.registros)} | Prueba reservada: {len(reales)}")
    print("\nSoporte por clase")
    print(f"{'Clase':<8} {'Desarrollo':>10} {'Prueba':>8}")
    for clase in clases:
        print(f"{clase:<8} {soporte_desarrollo[clase]:>10} {soporte_prueba[clase]:>8}")

    print("\nAccuracy interna (CV de 5 bolsas sobre desarrollo):")
    print(" | ".join(f"k={k}: {score:.4f}" for k, score in resultados_cv.items()))
    print(f"k elegido: {mejor_k} | Clase mayoritaria de desarrollo: {mayoritaria}")

    print("\nMetricas en la prueba reservada")
    cabecera_recall = " ".join(f"{f'Recall {clase}':>10}" for clase in clases)
    print(f"{'Modelo':<13} {'Accuracy':>9} {'Acc. bal.':>9} {cabecera_recall}")
    for nombre, predichos in (("Mayoritaria", [mayoritaria] * len(reales)), ("kNN", pred_knn)):
        aciertos = Counter(real for real, pred in zip(reales, predichos) if real == pred)
        recalls = [aciertos[clase] / soporte_prueba[clase] for clase in clases]
        recall_texto = " ".join(f"{recall:>10.4f}" for recall in recalls)
        accuracy = validador.calcular_metricas(predichos, reales)
        print(f"{nombre:<13} {accuracy:>9.4f} {sum(recalls) / len(recalls):>9.4f} {recall_texto}")


if __name__ == "__main__":
    main()
