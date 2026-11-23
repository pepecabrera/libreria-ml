import unittest
from collections import Counter

from examples.wine import FRACCION_PRUEBA, RUTA_WINE, SEMILLA, separar_estratificado
from src.factoria import FactoriaCSV


class TestExperimentoWine(unittest.TestCase):
    def test_particion_estratificada_reproducible_y_sin_solapamiento(self):
        datos = FactoriaCSV.crear_dataset_clasificacion(
            RUTA_WINE, indice_objetivo=0, tiene_cabecera=False,
        )

        def filas(dataset):
            return [(reg.objetivo, tuple(reg.atributos)) for reg in dataset.registros]

        desarrollo, prueba = separar_estratificado(datos, FRACCION_PRUEBA, SEMILLA)
        otro_desarrollo, otra_prueba = separar_estratificado(datos, FRACCION_PRUEBA, SEMILLA)

        self.assertEqual(filas(desarrollo), filas(otro_desarrollo))
        self.assertEqual(filas(prueba), filas(otra_prueba))
        self.assertTrue(set(filas(desarrollo)).isdisjoint(filas(prueba)))
        self.assertCountEqual(filas(desarrollo) + filas(prueba), filas(datos))
        self.assertEqual(Counter(clase for clase, _ in filas(prueba)), {"1": 12, "2": 14, "3": 10})
