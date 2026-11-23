import csv
import unittest
from pathlib import Path

from src.factoria import FactoriaCSV


DATOS = Path(__file__).resolve().parents[1] / "datos"


class TestFactoriaCSV(unittest.TestCase):
    def test_wine_conserva_la_primera_observacion(self):
        dataset = FactoriaCSV.crear_dataset_clasificacion(
            DATOS / "wine" / "wine.data", indice_objetivo=0, tiene_cabecera=False,
        )

        self.assertEqual(len(dataset.registros), 178)
        self.assertEqual(len(dataset.nombres_atributos), 13)
        self.assertEqual(dataset.registros[0].objetivo, "1")
        self.assertEqual(dataset.registros[0].atributos[0], 14.23)

    def test_wdbc_conserva_la_primera_observacion(self):
        dataset = FactoriaCSV.crear_dataset_clasificacion(
            DATOS / "BreastCancerWisconsinDiagnostic" / "wdbc.data",
            indice_objetivo=1, tiene_cabecera=False,
        )

        self.assertEqual(len(dataset.registros), 569)
        self.assertEqual(len(dataset.nombres_atributos), 31)
        self.assertEqual(dataset.registros[0].objetivo, "M")
        self.assertEqual(dataset.registros[0].atributos[:2], [842302, 17.99])

    def test_boston_usa_medv_sin_incluir_su_categoria(self):
        ruta = DATOS / "BostonHousing.csv"
        dataset = FactoriaCSV.crear_dataset_regresion(
            ruta, indice_objetivo="MEDV", tiene_cabecera=True,
            excluir_columnas=("CAT. MEDV",),
        )
        predictores = [
            "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
            "AGE", "DIS", "RAD", "TAX", "PTRATIO", "LSTAT",
        ]

        self.assertEqual(dataset.nombres_atributos, predictores)
        with ruta.open(encoding="utf-8-sig", newline="") as archivo:
            filas = list(csv.DictReader(archivo))
        self.assertEqual(len(dataset.registros), len(filas))
        for registro, fila in zip(dataset.registros, filas):
            self.assertEqual(registro.objetivo, float(fila["MEDV"]))
            self.assertEqual(registro.atributos, [float(fila[nombre]) for nombre in predictores])
