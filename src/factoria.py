import csv
from itertools import chain
from typing import Union

from .dataset import DataSetClasificacion, DataSetRegresion
from .registro import RegistroClasificacion, RegistroRegresion

# Importamos openpyxl para Excel. 
# Nota: Debes instalarlo previamente con 'pip install openpyxl'
try:
    import openpyxl
except ImportError:
    openpyxl = None


class FactoriaCSV:
    """Clase factoría para crear DataSets a partir de archivos CSV."""

    @staticmethod
    def crear_dataset_clasificacion(
        ruta_fichero: str, indice_objetivo: int = -1, *, tiene_cabecera: bool = True,
    ) -> DataSetClasificacion:
        """Carga etiquetas de texto; usa tiene_cabecera=False para CSV sin cabecera."""
        return FactoriaCSV._crear_dataset(
            ruta_fichero, DataSetClasificacion(), RegistroClasificacion, str,
            indice_objetivo, tiene_cabecera,
        )

    @staticmethod
    def crear_dataset_regresion(
        ruta_fichero: str, indice_objetivo: Union[int, str] = -1, *,
        tiene_cabecera: bool = True, excluir_columnas=(),
    ) -> DataSetRegresion:
        """Carga un objetivo numérico por índice o nombre y permite excluir columnas por nombre."""
        return FactoriaCSV._crear_dataset(
            ruta_fichero, DataSetRegresion(), RegistroRegresion, float,
            indice_objetivo, tiene_cabecera, excluir_columnas,
        )

    @staticmethod
    def _crear_dataset(
        ruta, dataset, tipo_registro, convertir_objetivo, indice_objetivo,
        tiene_cabecera, excluir_columnas=(),
    ):
        with open(ruta, encoding='utf-8-sig', newline='') as archivo:
            filas = (fila for fila in csv.reader(archivo) if any(fila))
            primera = next(filas, None)
            if primera is None:
                return dataset

            if tiene_cabecera:
                cabeceras = [nombre.strip() for nombre in primera]
            else:
                cabeceras = [f"columna_{i}" for i in range(len(primera))]
                filas = chain((primera,), filas)

            indices = range(len(cabeceras))
            if isinstance(indice_objetivo, str):
                indice_objetivo = cabeceras.index(indice_objetivo)
            indice_objetivo = indices[indice_objetivo]
            excluidos = {cabeceras.index(nombre) for nombre in excluir_columnas}
            predictores = [i for i in indices if i != indice_objetivo and i not in excluidos]
            dataset.set_cabeceras([cabeceras[i] for i in predictores] + [cabeceras[indice_objetivo]])

            for fila in filas:
                if len(fila) != len(cabeceras):
                    raise ValueError("Una fila del CSV no coincide con el número de columnas esperado.")
                objetivo = convertir_objetivo(fila[indice_objetivo].strip())
                atributos = [float(fila[i]) for i in predictores]
                dataset.agregar_registro(tipo_registro(atributos, objetivo))

        return dataset


class FactoriaXLS:
    """Clase factoría para crear DataSets a partir de archivos Excel."""

    @staticmethod
    def _comprobar_libreria():
        if openpyxl is None:
            raise ImportError("La librería 'openpyxl' es obligatoria para leer Excel. Ejecuta 'pip install openpyxl'.")

    @staticmethod
    def crear_dataset_clasificacion(ruta_fichero: str, indice_objetivo: int = -1) -> DataSetClasificacion:
        FactoriaXLS._comprobar_libreria()
        dataset = DataSetClasificacion()
        
        # Cargamos el libro de Excel (data_only=True para ignorar fórmulas y coger el valor final)
        wb = openpyxl.load_workbook(ruta_fichero, data_only=True)
        sheet = wb.active
        
        # Extraemos todas las filas
        filas = list(sheet.iter_rows(values_only=True))
        if not filas:
            return dataset
            
        # 1. Leer cabeceras
        cabeceras = list(filas[0])
        obj_header = cabeceras.pop(indice_objetivo)
        cabeceras.append(obj_header)
        # Nos aseguramos de que las cabeceras sean strings
        dataset.set_cabeceras([str(c) for c in cabeceras])
        
        # 2. Leer los datos
        for fila in filas[1:]:
            if not any(fila): 
                continue
                
            fila_lista = list(fila)
            objetivo = str(fila_lista.pop(indice_objetivo)).strip()
            # Los valores en Excel pueden venir como enteros o nulos, los forzamos a float
            atributos = [float(x) for x in fila_lista if x is not None]
            
            registro = RegistroClasificacion(atributos, objetivo)
            dataset.agregar_registro(registro)
            
        return dataset

    @staticmethod
    def crear_dataset_regresion(ruta_fichero: str, indice_objetivo: int = -1) -> DataSetRegresion:
        FactoriaXLS._comprobar_libreria()
        dataset = DataSetRegresion()
        
        wb = openpyxl.load_workbook(ruta_fichero, data_only=True)
        sheet = wb.active
        
        filas = list(sheet.iter_rows(values_only=True))
        if not filas:
            return dataset
            
        cabeceras = list(filas[0])
        obj_header = cabeceras.pop(indice_objetivo)
        cabeceras.append(obj_header)
        dataset.set_cabeceras([str(c) for c in cabeceras])
        
        for fila in filas[1:]:
            if not any(fila):
                continue
                
            fila_lista = list(fila)
            # En regresión, el objetivo es float
            objetivo = float(fila_lista.pop(indice_objetivo))
            atributos = [float(x) for x in fila_lista if x is not None]
            
            registro = RegistroRegresion(atributos, objetivo)
            dataset.agregar_registro(registro)
            
        return dataset
    
# Importamos xlrd para soportar archivos Excel antiguos (.xls)
try:
    import xlrd
except ImportError:
    xlrd = None


class FactoriaOldXLS:
    """Clase factoría para crear DataSets a partir de archivos Excel antiguos (.xls)."""

    @staticmethod
    def _comprobar_libreria():
        if xlrd is None:
            raise ImportError("La librería 'xlrd' es obligatoria para leer .xls. Ejecuta 'pip install xlrd'.")

    @staticmethod
    def crear_dataset_clasificacion(ruta_fichero: str, indice_objetivo: int = -1) -> DataSetClasificacion:
        FactoriaOldXLS._comprobar_libreria()
        dataset = DataSetClasificacion()

        wb = xlrd.open_workbook(ruta_fichero)
        sheet = wb.sheet_by_index(0)

        if sheet.nrows == 0:
            return dataset

        # 1. Cabeceras
        cabeceras = sheet.row_values(0)
        obj_header = cabeceras.pop(indice_objetivo)
        cabeceras.append(obj_header)
        dataset.set_cabeceras([str(c) for c in cabeceras])

        # 2. Datos
        for i in range(1, sheet.nrows):
            fila = sheet.row_values(i)

            if not any(fila):
                continue

            fila_lista = list(fila)
            objetivo = str(fila_lista.pop(indice_objetivo)).strip()
            atributos = [float(x) for x in fila_lista]

            registro = RegistroClasificacion(atributos, objetivo)
            dataset.agregar_registro(registro)

        return dataset

    @staticmethod
    def crear_dataset_regresion(ruta_fichero: str, indice_objetivo: int = -1) -> DataSetRegresion:
        FactoriaOldXLS._comprobar_libreria()
        dataset = DataSetRegresion()

        wb = xlrd.open_workbook(ruta_fichero)
        sheet = wb.sheet_by_index(0)

        if sheet.nrows == 0:
            return dataset

        # 1. Cabeceras
        cabeceras = sheet.row_values(0)
        obj_header = cabeceras.pop(indice_objetivo)
        cabeceras.append(obj_header)
        dataset.set_cabeceras([str(c) for c in cabeceras])

        # 2. Datos
        for i in range(1, sheet.nrows):
            fila = sheet.row_values(i)

            if not any(fila):
                continue

            fila_lista = list(fila)
            objetivo = float(fila_lista.pop(indice_objetivo))
            atributos = [float(x) for x in fila_lista]

            registro = RegistroRegresion(atributos, objetivo)
            dataset.agregar_registro(registro)

        return dataset
