import pandas as pd
import os
import unicodedata
from datetime import datetime

# ==========================================
# 1. CONFIGURACIÓN E INFRAESTRUCTURA
# ==========================================
if os.path.exists("/app"):
    DATA_DIR = "/app/data"
else:
    DATA_DIR = os.path.join(os.getcwd(), "data")

# ==========================================
# 2. BASE DE CONOCIMIENTO (REGLAS EXPORTABLES)
# ==========================================
REGLAS_CLASIFICACION = {
    "Privatización / Concesión": [
        "concesión",
        "privatización",
        "venta de pliegos",
        "adjudicación",
        "licitación pública nacional e internacional",
    ],
    "Contratos Públicos": [
        "obra pública",
        "redeterminación de precios",
        "contratación directa",
        "ajuste de contrato",
        "continuidad de obra",
        "compra directa",
    ],
    "Tarifas Servicios Públicos": [
        "cuadro tarifario",
        "aumento de tarifa",
        "revisión tarifaria",
        "ente regulador",
        "precio mayorista",
        "peaje",
        "transporte de energía",
    ],
    "Autorizaciones de Precios": [
        "compensación cambiaria",
        "diferencia de cambio",
        "precio máximo",
        "secretaría de comercio",
        "ajuste de precio",
        "valores de referencia",
    ],
    "Precios Sector Privado (Salud/Educación)": [
        "medicina prepaga",
        "cuota colegio",
        "arancel educativo",
        "superintendencia de servicios de salud",
        "autorízase aumento",
        "instituciones de enseñanza pública de gestión privada",
    ],
    "Jubilaciones / Pensiones": [
        "movilidad jubilatoria",
        "haber mínimo",
        "anses",
        "índice de actualización",
        "bono previsional",
        "sistema integrado previsional argentino",
    ],
    "Traslado de Impuestos": [
        "traslado a precios",
        "incidencia impositiva",
        "impuesto al consumo",
        "tasas y contribuciones",
        "percepción impositiva",
        "regimen de retención",
    ],
}

PALABRAS_ALERTA_AUDITORIA = [
    "millones",
    "asignación",
    "transferencia",
    "fondo fiduciario",
    "partida presupuestaria",
    "erogación",
    "compra",
    "contratación",
    "pago",
]

MATRIZ_TEORICA = {
    "Privatización / Concesión": {
        "origen": "Patrimonio Estatal",
        "destino": "Empresas Privadas (Rent Seeking)",
        "mecanismo": "Subvaluación de activos o canon bajo",
        "certeza_nivel": "Alta",
        "puntos_certeza": 30,
    },
    "Contratos Públicos": {
        "origen": "Contribuyentes (Impuestos Futuros)",
        "destino": "Empresas Contratistas",
        "mecanismo": "Sobreprecios o continuación ineficiente",
        "certeza_nivel": "Media-Alta",
        "puntos_certeza": 25,
    },
    "Tarifas Servicios Públicos": {
        "origen": "Usuarios / Población",
        "destino": "Empresas Concesionarias",
        "mecanismo": "Aumento de tarifa o subsidio cruzado",
        "certeza_nivel": "Muy Alta",
        "puntos_certeza": 40,
    },
    "Autorizaciones de Precios": {
        "origen": "Tesoro Nacional / Consumidor",
        "destino": "Sectores Regulados",
        "mecanismo": "Validación estatal de aumentos o subsidios",
        "certeza_nivel": "Alta",
        "puntos_certeza": 30,
    },
    "Precios Sector Privado (Salud/Educación)": {
        "origen": "Salario de los Trabajadores",
        "destino": "Empresas de Salud/Educación",
        "mecanismo": "Autorización de aumento por encima de inflación",
        "certeza_nivel": "Alta",
        "puntos_certeza": 30,
    },
    "Jubilaciones / Pensiones": {
        "origen": "Jubilados (Ingreso Diferido)",
        "destino": "Estado (Tesoro)",
        "mecanismo": "Fórmula de movilidad a la baja / Inflación",
        "certeza_nivel": "Muy Alta",
        "puntos_certeza": 40,
    },
    "Traslado de Impuestos": {
        "origen": "Consumidor Final",
        "destino": "Estado / Empresas",
        "mecanismo": "Traslado de carga fiscal (Doble imposición)",
        "certeza_nivel": "Muy Alta",
        "puntos_certeza": 40,
    },
}

# ==========================================
# 3. FUNCIONES DE HIGIENE Y CLASIFICACIÓN
# ==========================================


def limpiar_texto_curado(texto):
    """Normaliza el texto para evitar errores por codificación."""
    if not isinstance(texto, str):
        return ""
    texto = unicodedata.normalize("NFKC", texto)
    return texto.strip()


def clasificar_texto_interno(texto):
    """
    Función interna para clasificar texto crudo si llega 'No identificado'.
    Usa las reglas definidas arriba.
    """
    texto = str(texto).lower()
    for tipo, palabras in REGLAS_CLASIFICACION.items():
        if any(p in texto for p in palabras):
            return tipo
    return "No identificado"


def recuperar_evidencia_xai(row):
    """
    [XAI] Busca qué palabra clave activó la regla.
    """
    tipo = row.get("tipo_decision")
    texto = str(row.get("detalle", "")).lower()

    if not tipo or tipo == "No identificado":
        return "-"

    if tipo in REGLAS_CLASIFICACION:
        palabras = REGLAS_CLASIFICACION[tipo]
        for p in palabras:
            if p.lower() in texto:
                return p
    return "Inferencia implícita"


def flag_revision_humana(row):
    """Detecta Falsos Negativos para auditoría."""
    if row["tipo_decision"] == "No identificado":
        texto = str(row.get("detalle", "")).lower()
        matches = [p for p in PALABRAS_ALERTA_AUDITORIA if p in texto]
        if matches:
            return f"⚠️ REVISAR: Posible {matches[0]}"
    return "OK"


# ==========================================
# 4. LÓGICA DE NEGOCIO
# ==========================================


def aplicar_matriz_teorica(tipo_decision):
    return MATRIZ_TEORICA.get(
        tipo_decision,
        {
            "origen": "-",
            "destino": "-",
            "mecanismo": "-",
            "certeza_nivel": "Nula",
            "puntos_certeza": 0,
        },
    )


def desglosar_indice(row):
    if row["tipo_decision"] == "No identificado":
        return pd.Series(
            {
                "idx_legalidad": 0,
                "idx_discrecionalidad": 0,
                "idx_certeza": 0,
                "indice_total": 0,
                "elaboracion_indice": "No aplica",
            }
        )

    p_legal = 30
    p_discrecional = 30
    datos_teoricos = MATRIZ_TEORICA.get(row["tipo_decision"])
    p_certeza = datos_teoricos["puntos_certeza"] if datos_teoricos else 0
    total = p_legal + p_discrecional + p_certeza

    return pd.Series(
        {
            "idx_legalidad": p_legal,
            "idx_discrecionalidad": p_discrecional,
            "idx_certeza": p_certeza,
            "indice_total": total,
            "elaboracion_indice": f"Leg({p_legal})+Dis({p_discrecional})+Cert({p_certeza})",
        }
    )


# ==========================================
# 5. ORQUESTADOR PRINCIPAL
# ==========================================


def analizar_boletin(df):
    """
    Recibe el DataFrame crudo, lo CLASIFICA, limpia, audita y enriquece.
    """
    if df.empty:
        return df, None, pd.DataFrame()

    # PASO 1: Curado de Datos
    df["detalle"] = df["detalle"].apply(limpiar_texto_curado)

    # PASO 2: Clasificación Robusta
    if "tipo_decision" not in df.columns:
        df["tipo_decision"] = "No identificado"

    df["tipo_decision"] = df.apply(
        lambda row: clasificar_texto_interno(row["detalle"])
        if row["tipo_decision"] == "No identificado"
        else row["tipo_decision"],
        axis=1,
    )

    # PASO 3: Enriquecimiento Teórico
    detalles = df["tipo_decision"].apply(aplicar_matriz_teorica)
    df = pd.concat(
        [df.reset_index(drop=True), pd.json_normalize(detalles).reset_index(drop=True)],
        axis=1,
    )

    # PASO 4: Índices
    desglose = df.apply(desglosar_indice, axis=1)
    df = pd.concat([df, desglose], axis=1)

    # PASO 5: XAI y Auditoría
    df["evidencia_xai"] = df.apply(recuperar_evidencia_xai, axis=1)
    df["auditoria_estado"] = df.apply(flag_revision_humana, axis=1)

    # PASO 6: Generación de Tablas (Glosario y Marco Teórico Actualizado)

    desc_tipos = (
        "7 Tipos: 1.Privatizaciones/Concesiones, 2.Contratos Públicos, "
        "3.Tarifas, 4.Autorizaciones de Precios, 5.Precios Sector Privado, "
        "6.Jubilaciones/Pensiones, 7.Traslado de Impuestos."
    )

    desc_evidencia = (
        "Elemento textual (keyword) detectado en la norma que fundamenta "
        "su clasificación dentro de uno de los 7 fenómenos teóricos."
    )

    # Definición de la fórmula del índice y su consecuencia
    desc_formula = (
        "Fórmula: Leg(30) + Dis(30) + Cert(25-40). "
        "Suma Legalidad, Discrecionalidad y Certeza Teórica. "
        "Consecuencia: Un valor alto confirma que el acto, aunque legal, transfiere riqueza."
    )

    glosario_data = [
        {"Columna": "fecha", "Descripción": "Fecha publicación B.O."},
        {"Columna": "tipo_decision", "Descripción": desc_tipos},
        {"Columna": "evidencia_xai", "Descripción": desc_evidencia},
        {
            "Columna": "auditoria_estado",
            "Descripción": "[Control] Alerta de revisión humana.",
        },
        {"Columna": "indice_total", "Descripción": "Intensidad del riesgo (0-100%)."},
        {"Columna": "detalle", "Descripción": "Texto completo de la norma."},
        # NUEVA COLUMNA AGREGADA
        {"Columna": "elaboracion_indice", "Descripción": desc_formula},
    ]
    df_glosario = pd.DataFrame(glosario_data)

    causas_data = []
    for causa, data in MATRIZ_TEORICA.items():
        causas_data.append(
            {
                "Fenómeno / Causa": causa,
                "Origen (Víctima)": data["origen"],
                "Destino (Beneficiario)": data["destino"],
                "Mecanismo": data["mecanismo"],
                "Certeza Teórica": f"{data['certeza_nivel']} ({data['puntos_certeza']} pts)",
            }
        )
    df_causas = pd.DataFrame(causas_data)

    # PASO 7: Exportación
    columnas_ordenadas = [
        "fecha",
        "seccion",
        "tipo_decision",
        "evidencia_xai",
        "auditoria_estado",
        "indice_total",
        "elaboracion_indice",
        "origen",
        "destino",
        "mecanismo",
        "detalle",
        "link",
    ]
    cols_final = [c for c in columnas_ordenadas if c in df.columns]
    df_final = df[cols_final]

    fecha_str = datetime.now().strftime("%Y%m%d")
    output_path = os.path.join(DATA_DIR, f"reporte_fenomenos_{fecha_str}.xlsx")

    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_final.to_excel(writer, index=False, sheet_name="Analisis")

            # Formateo visual del Excel
            ws = writer.sheets["Analisis"]
            ws.column_dimensions["D"].width = 25
            ws.column_dimensions["E"].width = 25
            ws.column_dimensions["K"].width = 60

            df_causas.to_excel(writer, index=False, sheet_name="Marco Teorico")
            ws_teoria = writer.sheets["Marco Teorico"]
            ws_teoria.column_dimensions["A"].width = 35
            ws_teoria.column_dimensions["B"].width = 30

            df_glosario.to_excel(writer, index=False, sheet_name="Glosario")

    except Exception as e:
        print(f"Error guardando Excel: {e}")

    return df, output_path, df_glosario