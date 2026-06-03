import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Cargar variables de entorno (como GOOGLE_API_KEY)
load_dotenv()

st.set_page_config(page_title="Flipp", page_icon="📄", layout="wide")

st.title("📄 Flipp")
st.markdown("Sube tus pdfs")

# Inicializar estado para guardar el índice vectorial
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Sidebar para búsqueda semántica e información
with st.sidebar:
    st.header("Búsqueda Semántica Rápida")
    semantic_query = st.text_input("Busca conceptos en todos los PDFs:")
    st.markdown("---")
    if st.button("Limpiar Base de Datos (Reiniciar)"):
        st.session_state.vector_store = None
        st.rerun()

uploaded_files = st.file_uploader("Sube archivos PDF para analizar", type="pdf", accept_multiple_files=True)

if uploaded_files and st.session_state.vector_store is None:
    with st.spinner("Leyendo y procesando todos los documentos..."):
        all_splits = []
        
        for uploaded_file in uploaded_files:
            # Guardar archivo temporalmente
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # 1. Cargar el PDF
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            # Añadir metadatos de nombre de archivo a cada documento
            for doc in docs:
                doc.metadata["source_filename"] = uploaded_file.name

            # 2. Dividir el texto en fragmentos (chunks)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)

            # Limpieza
            os.unlink(tmp_path)

        # 3. Crear Embeddings Locales (Gratis, rápidos y sin errores de API)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        try:
            # Corporate Fallback: Intentar usar ChromaDB (Estándar de la industria)
            from langchain_chroma import Chroma
            vector_store = Chroma.from_documents(documents=all_splits, embedding=embeddings)
        except Exception:
            # Fallback a InMemoryVectorStore si Windows Defender / AppLocker bloquean la DLL C++ de gRPC
            from langchain_core.vectorstores import InMemoryVectorStore
            vector_store = InMemoryVectorStore.from_documents(documents=all_splits, embedding=embeddings)
            
        st.session_state.vector_store = vector_store
    
    st.success("¡Base de conocimiento creada! Ya puedes hacer preguntas o buscar.")

# Lógica de búsqueda semántica en la barra lateral
if st.session_state.vector_store is not None and semantic_query:
    with st.sidebar:
        st.write("**Coincidencias encontradas:**")
        # Búsqueda de similitud simple sin el LLM
        results = st.session_state.vector_store.similarity_search(semantic_query, k=3)
        for i, res in enumerate(results):
            filename = res.metadata.get("source_filename", "Desconocido")
            page = res.metadata.get("page", "N/A")
            st.info(f"**PDF:** {filename} (Pág. {page})\n\n...{res.page_content[:150]}...")

if st.session_state.vector_store is not None:
    st.markdown("---")
    st.subheader("Chat con tu base de conocimiento")
    user_query = st.chat_input("Escribe aquí tu pregunta...")

    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        
        with st.spinner("Analizando múltiples documentos..."):
            # 4. Configurar el LLM de Gemini
            llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)

            # 5. Crear el prompt inyectando el contexto (RAG)
            system_prompt = (
                "Eres un asistente analista de documentos. "
                "Usa únicamente los siguientes fragmentos de contexto recuperado para responder a la pregunta del usuario. "
                "Si la respuesta no se encuentra en el contexto, indica claramente que la información no está en los documentos, no intentes inventarla.\n\n"
                "Contexto recuperado de la base de datos vectorial:\n{context}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            # 6. Construir la cadena RAG (Retrieval Chain)
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            # 7. Ejecutar consulta
            response = rag_chain.invoke({"input": user_query})

            with st.chat_message("assistant"):
                st.write(response["answer"])

            # 8. Mostrar las fuentes
            with st.expander("Fuentes de información consultadas"):
                for i, doc in enumerate(response["context"]):
                    filename = doc.metadata.get("source_filename", "Desconocido")
                    st.caption(f"Fragmento {i+1} - {filename} (Pág. {doc.metadata.get('page', 'N/A')}):")
                    st.write(doc.page_content)
