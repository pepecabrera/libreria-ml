import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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

    def test_csv_sin_cabecera_conserva_la_primera_fila_con_objetivo_intermedio(self):
        with TemporaryDirectory() as directorio:
            ruta = Path(directorio) / "sintetico.csv"
            ruta.write_text("1,A,2.5\n2,B,3.5\n", encoding="utf-8")
            dataset = FactoriaCSV.crear_dataset_clasificacion(
                ruta, indice_objetivo=1, tiene_cabecera=False,
            )

        self.assertEqual(len(dataset.nombres_atributos), 2)
        self.assertEqual([reg.objetivo for reg in dataset.registros], ["A", "B"])
        self.assertEqual([reg.atributos for reg in dataset.registros], [[1.0, 2.5], [2.0, 3.5]])

    def test_esquema_boston_usa_medv_sin_incluir_su_categoria(self):
        predictores = [
            "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
            "AGE", "DIS", "RAD", "TAX", "PTRATIO", "LSTAT",
        ]
        atributos = list(range(12))
        with TemporaryDirectory() as directorio:
            ruta = Path(directorio) / "sintetico.csv"
            with ruta.open("w", encoding="utf-8", newline="") as archivo:
                escritor = csv.writer(archivo)
                escritor.writerow(predictores + ["MEDV", "CAT. MEDV"])
                escritor.writerow(atributos + [25.5, 1])
            dataset = FactoriaCSV.crear_dataset_regresion(
                ruta, indice_objetivo="MEDV", tiene_cabecera=True,
                excluir_columnas=("CAT. MEDV",),
            )

        self.assertEqual(dataset.nombres_atributos, predictores)
        self.assertEqual(len(dataset.registros), 1)
        self.assertEqual(dataset.registros[0].objetivo, 25.5)
        self.assertEqual(dataset.registros[0].atributos, atributos)
