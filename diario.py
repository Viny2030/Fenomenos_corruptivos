import os
import pandas as pd
from datetime import datetime

# 1. Asegura que la carpeta 'data' exista en el servidor de GitHub
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def ejecutar_robot():
    print(
        f"Iniciando scraping del BORA para el día: {datetime.now().strftime('%Y-%m-%d')}"
    )

    # Aquí va tu lógica de scraping. Ejemplo de datos:
    data = [
        {
            "fecha": datetime.now().strftime("%Y-%m-%d"),
            "detalle": "Análisis automático de fenómenos corruptivos - BORA",
            "estado": "Procesado",
        }
    ]
    df = pd.DataFrame(data)

    # 2. Define el nombre exacto que buscas en PyCharm (.xlsx)
    file_name = f"reporte_fenomenos_{datetime.now().strftime('%Y%m%d')}.xlsx"
    path_completo = os.path.join(DATA_DIR, file_name)

    # 3. ARMA EL EXCEL (Usa to_excel, no to_csv)
    try:
        df.to_excel(path_completo, index=False)
        print(f"ARCHIVO GENERADO: {path_completo}")
    except Exception as e:
        print(f"Error al armar el Excel: {e}")


if __name__ == "__main__":
    ejecutar_robot()