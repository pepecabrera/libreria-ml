import unittest
from unittest.mock import patch

from src.dataset import DataSetClasificacion, DataSetRegresion
from src.modelos import Clasificador_kNN, Regresor_kNN, Regresor_lineal_multiple
from src.registro import Registro, RegistroClasificacion, RegistroRegresion


class TestKNN(unittest.TestCase):
    def test_clasificador_vota_por_mayoria_y_desempata_por_cercania(self):
        datos = DataSetClasificacion()
        for x, etiqueta in [(0, "A"), (3, "B"), (4, "B")]:
            datos.agregar_registro(RegistroClasificacion([x], etiqueta))

        consulta = Registro([1])
        for k, esperada in [(2, "A"), (3, "B")]:
            with self.subTest(k=k):
                modelo = Clasificador_kNN(k)
                modelo.entrenar(datos)
                self.assertEqual(modelo.predecir(consulta), esperada)

    def test_regresor_promedia_solo_los_vecinos_mas_cercanos(self):
        datos = DataSetRegresion()
        for x, objetivo in [(0, 10), (3, 20), (4, 100)]:
            datos.agregar_registro(RegistroRegresion([x], objetivo))

        modelo = Regresor_kNN(k=2)
        modelo.entrenar(datos)
        self.assertEqual(modelo.predecir(Registro([1])), 15.0)

    def test_rechaza_k_que_no_es_un_entero_positivo(self):
        for k in (0, -1, 1.5, "2", True):
            with self.subTest(k=k), self.assertRaisesRegex(ValueError, "entero positivo"):
                Registro([0]).k_vecinos([], k)
            for tipo_modelo in (Clasificador_kNN, Regresor_kNN):
                with self.subTest(k=k, modelo=tipo_modelo), self.assertRaisesRegex(ValueError, "entero positivo"):
                    tipo_modelo(k)

    def test_rechaza_si_no_hay_suficientes_vecinos(self):
        casos = (
            (Clasificador_kNN, DataSetClasificacion(), RegistroClasificacion([0], "A")),
            (Regresor_kNN, DataSetRegresion(), RegistroRegresion([0], 10)),
        )
        for tipo_modelo, datos, registro in casos:
            with self.subTest(modelo=tipo_modelo):
                modelo = tipo_modelo(k=2)
                datos.agregar_registro(registro)
                modelo.entrenar(datos)
                with self.assertRaisesRegex(ValueError, "k=2.*1 disponibles"):
                    modelo.predecir(Registro([1]))
                # El propio registro no cuenta como vecino aunque figure en entrenamiento.
                with self.assertRaisesRegex(ValueError, "k=2.*0 disponibles"):
                    modelo.predecir(registro)


class TestRegresionLineal(unittest.TestCase):
    def test_recupera_la_solucion_de_un_plano_conocido(self):
        datos = DataSetRegresion()
        for x1, x2 in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            datos.agregar_registro(RegistroRegresion([x1, x2], 3 + 2 * x1 - x2))

        modelo = Regresor_lineal_multiple(learning_rate=0.1, epocas=200)
        with patch("src.modelos.random.uniform", return_value=0.0):
            modelo.entrenar(datos)

        self.assertAlmostEqual(modelo.sesgo, 3.0, places=6)
        self.assertAlmostEqual(modelo.pesos[0], 2.0, places=6)
        self.assertAlmostEqual(modelo.pesos[1], -1.0, places=6)
        self.assertAlmostEqual(modelo.predecir(Registro([2, -3])), 10.0, places=6)
