import unittest

from src.dataset import DataSetRegresion
from src.modelos import Regresor_kNN
from src.preprocesado import NormalizadorMaxMin
from src.registro import RegistroRegresion
from src.validacion import SeleccionAtributos, ValidacionRegresion


def crear_datos(filas):
    datos = DataSetRegresion()
    datos.set_cabeceras(["x", "z", "y"])
    for x, z, y in filas:
        datos.agregar_registro(RegistroRegresion([x, z], y))
    return datos


class TestValidacion(unittest.TestCase):
    def test_cambiar_etiquetas_de_prueba_no_cambia_columnas_ni_predicciones(self):
        train = crear_datos([(0, 0, 0), (1, 1, 1), (2, 1, 2), (3, 0, 3)])
        test = crear_datos([(0.1, 1000, 0), (100, -1000, 3)])
        test_reetiquetado = crear_datos([(0.1, 1000, 1000), (100, -1000, -1000)])
        validador = ValidacionRegresion()

        for evaluacion in (test, test_reetiquetado):
            with self.subTest(etiquetas=[r.objetivo for r in evaluacion.registros]):
                selector = SeleccionAtributos(p=0.5)
                escalador = NormalizadorMaxMin()
                predicciones = validador.predecir_particion(
                    Regresor_kNN(k=1), train, evaluacion, escalador, selector=selector,
                )
                self.assertEqual(selector.indices_seleccionados, [0])
                self.assertEqual(predicciones, [0, 3])
                self.assertEqual(escalador.minimos, [0])
                self.assertEqual(escalador.maximos, [3])

    def test_cv_comparte_bolsas_contiguas_y_ajusta_solo_con_train(self):
        datos = crear_datos((i, i % 2, i) for i in range(7))
        particiones = []
        ajustes = []

        class ValidadorRegistro(ValidacionRegresion):
            def evaluar_particion(self, modelo, train, test, normalizador, *, selector):
                particiones.append(([r.objetivo for r in train.registros],
                                    [r.objetivo for r in test.registros]))
                puntuacion = super().evaluar_particion(modelo, train, test, normalizador, selector=selector)
                ajustes.append((selector.indices_seleccionados, normalizador.minimos, normalizador.maximos))
                return puntuacion

        selector = SeleccionAtributos(p=0.5)
        escalador = NormalizadorMaxMin()
        for k in (1, 2):
            ValidadorRegistro().validacion_cruzada(Regresor_kNN(k), datos, 3, escalador, selector=selector)

        self.assertEqual(particiones, [
            ([2, 3, 4, 5, 6], [0, 1]),
            ([0, 1, 4, 5, 6], [2, 3]),
            ([0, 1, 2, 3], [4, 5, 6]),
        ] * 2)
        self.assertEqual(ajustes, [([0], [2], [6]), ([0], [0], [6]), ([0], [0], [3])] * 2)
        self.assertIsNone(selector.indices_seleccionados)
        self.assertEqual(escalador.minimos, [])
