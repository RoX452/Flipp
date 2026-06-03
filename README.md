# DocInsight Engine

Una herramienta construida en Python para extraer información y analizar documentos PDF usando la arquitectura RAG (Retrieval-Augmented Generation). Permite subir un archivo, indexarlo en una base de datos vectorial y hacer consultas sobre su contenido en lenguaje natural.

## Arquitectura y Stack Tecnológico

- **LangChain:** Framework utilizado para conectar el flujo de datos, manejar los prompts y gestionar la ejecución.
- **ChromaDB:** Base de datos vectorial que corre de forma local para almacenar los fragmentos del texto y realizar la búsqueda de similitud.
- **Google Gemini API:** Se utiliza el modelo de Google para leer el contexto recuperado del PDF y generar las respuestas coherentes.
- **Streamlit:** Librería para montar una interfaz web sencilla e interactiva donde el usuario puede cargar los archivos.

## Cómo Funciona

1. **Procesamiento inicial:** El documento PDF se procesa y se divide en fragmentos de texto más pequeños.
2. **Embedding:** Estos fragmentos se convierten a vectores (representaciones numéricas) y se guardan en ChromaDB.
3. **Recuperación:** Cuando el usuario ingresa una pregunta, el sistema busca en la base de datos los fragmentos que mejor responden a la consulta.
4. **Respuesta:** Se envían los fragmentos encontrados junto a la pregunta al LLM, el cual elabora la respuesta basándose únicamente en la información del documento.

## Instrucciones de Uso Local

1. Clonar el repositorio.
2. Instalar dependencias con `pip install -r requirements.txt`.
3. Crear un archivo `.env` en la raíz del proyecto y añadir tu clave de API: `GOOGLE_API_KEY=tu_api_key_aqui`.
4. Ejecutar el servidor local con `streamlit run app.py`.
