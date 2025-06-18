# 🛍️ tec2025sql: Asistente de Análisis de Datos para Tiendas

Este proyecto es un asistente inteligente para análisis de datos de negocio, orientado a tiendas y distribuidores, desarrollado para el Tec de Monterrey. Utiliza FastAPI, OpenAI y una base de datos SQLite con datos simulados de productos, ventas y distribuidores. Permite hacer preguntas en lenguaje natural y obtener respuestas interpretadas, junto con consultas SQL y resultados en tablas.

## Características principales
- **Interfaz web amigable**: Consulta y visualiza resultados en tablas y texto interpretado.
- **Procesamiento de lenguaje natural**: Usa OpenAI para transformar preguntas en SQL y respuestas comprensibles.
- **Base de datos simulada**: Incluye productos, distribuidores y ventas realistas.
- **Backend con FastAPI**: API moderna y eficiente.
- **Adaptación automática de queries a SQLite**.

## Estructura del proyecto
- `main.py`: API principal (FastAPI) y lógica de procesamiento de preguntas.
- `create_db.py`: Script para crear y poblar la base de datos `store.db` con datos de ejemplo.
- `prompt.txt`: Prompt base para el modelo de lenguaje, con reglas y estructura de la base de datos.
- `requirements.txt`: Dependencias del proyecto.
- `templates/index.html`: Interfaz web para interactuar con el asistente.
- `.env`: Llave de API de OpenAI (no compartir públicamente).

## Instalación y uso rápido
1. **Clona el repositorio y entra al directorio:**
   ```bash
   git clone <repo-url>
   cd tec2025sql
   ```
2. **Instala dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Crea la base de datos de ejemplo:**
   ```bash
   python create_db.py
   ```
4. **Configura tu clave de OpenAI en `.env`** (ya incluida en este ejemplo).
5. **Inicia el servidor:**
   ```bash
   uvicorn main:app --reload
   ```
6. **Abre la app en tu navegador:**
   [http://localhost:8000](http://localhost:8000)

## ¿Cómo funciona?
- El usuario escribe una pregunta de negocio (ej: "¿Cuáles son los productos más vendidos?").
- El backend usa OpenAI para generar una consulta SQL y una interpretación.
- Se ejecuta el SQL en la base de datos local y se muestra el resultado en tabla y texto.

## Estructura de la base de datos
- **Productos**: Artículos con nombre, categoría, precios, etc.
- **Distribuidores**: Información de distribuidores y zonas.
- **Ventas**: Registros de ventas con fechas, descuentos y cantidades.

## Personalización
- Puedes modificar `create_db.py` para cambiar los datos simulados.
- El prompt en `prompt.txt` define el comportamiento y reglas del asistente.

## Dependencias principales
- fastapi
- uvicorn
- openai
- pydantic
- jinja2
- langgraph
- langchain

## Créditos
Desarrollado por el Tec de Monterrey, 2025. Proyecto educativo y demostrativo.
