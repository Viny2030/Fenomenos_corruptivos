📘 README.md — Fenómenos Corruptivos Legales
Implementación computacional de The Great Corruption
1. Descripción general

Este proyecto implementa un sistema computacional de análisis institucional basado en la teoría desarrollada en el artículo The Great Corruption.

El objetivo NO es detectar delitos ni corrupción penal, sino identificar y analizar decisiones estatales legales que, según la teoría económica, pueden generar transferencias regresivas de ingresos mediante mecanismos discrecionales.

El sistema analiza publicaciones del Boletín Oficial de la República Argentina, clasifica las decisiones detectadas y construye un Índice de Fenómeno Corruptivo de carácter explicativo y no acusatorio.

2. Marco teórico (resumen)

Según The Great Corruption, existen fenómenos corruptivos que:

Son legales

No implican necesariamente violación de normas

Surgen de decisiones discrecionales del Estado

Generan transferencias de ingresos regresivas

Benefician a grupos concentrados (empresas, sectores específicos, o el propio Estado)

Tienen impacto social negativo

Este proyecto traduce esa teoría en variables observables y medibles, sin criminalización.

3. Qué hace el sistema

El pipeline completo es:

Boletín Oficial
      ↓
Detección de decisiones estatales relevantes
      ↓
Clasificación del tipo de decisión
      ↓
Identificación de la transferencia de ingresos
      ↓
Cálculo del Índice de Fenómeno Corruptivo
      ↓
Reporte analítico + Dashboard explicativo

4. Componentes del proyecto
📄 main.py

Scrapea el Boletín Oficial

Detecta normas relevantes

Clasifica el tipo de decisión estatal

Genera un CSV base

📄 analisis.py

Aplica la teoría económica

Determina:

Dirección de la transferencia de ingresos

Índice de Fenómeno Corruptivo (0–10)

Nivel de riesgo teórico (Bajo / Medio / Alto)

Genera un reporte Excel

📊 dashboard.py (Streamlit)

Visualiza resultados

Explica la teoría

Permite exploración interactiva

Hace visible la transferencia regresiva

5. Variables principales
Variable	Descripción
tipo_decision	Tipo de decisión estatal (tarifas, concesiones, impuestos, etc.)
transferencia	Dirección de la transferencia de ingresos
indice_fenomeno_corruptivo	Intensidad teórica del fenómeno (0–10)
nivel_riesgo_teorico	Clasificación cualitativa
detalle	Texto oficial de la norma
link	Fuente oficial
6. Interpretación correcta de los resultados

⚠️ Advertencia metodológica importante

El sistema NO acusa

El sistema NO judicializa

El sistema NO afirma ilegalidad

El índice mide intensidad del fenómeno corruptivo legal, entendida como:

Grado potencial de transferencia regresiva generado por una decisión estatal discrecional.

## Base Teórica e Investigación
Este proyecto es parte de una investigación personal, basada en el artículo:

**"Great corruption – theory of corrupt phenomena"**
*Publicado en Journal of Financial Crime (2020).*

### Definición de Variables de Salida:
- **fecha**: Emisión en el Boletín Oficial.
- **tipo_decision**: Clasificación según los 7 escenarios de la Gran Corrupción.
- **transferencia**: Sector económico afectado (Ej: Jubilados, Estado).
- **indice_fenomeno_corruptivo**: Intensidad de 0 a 10.
- **nivel_riesgo_teorico**: Evaluación de opacidad (Bajo/Medio/Alto).
- **link**: Auditoría manual de la norma.

[Ver artículo en Emerald Insight](https://www.emerald.com/jfc/article-abstract/28/2/580/224032/Great-corruption-theory-of-corrupt-phenomena)

# 1. Clonar el repositorio
git clone https://github.com/Viny2030/Fenomenos_corruptivos.git
cd Fenomenos_corruptivos

# 2. Construir la imagen
docker build -t fenomenos_corruptivos:v3_robusta .

# 3. Ejecutar el contenedor
docker run -d -p 8502:8501 -v "${PWD}/data:/app/data" fenomenos_corruptivos:v3_robusta