import streamlit as st
import os
import tempfile
import re
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
st.set_page_config(page_title="Flipp", page_icon="📄", layout="wide")

# Helper function to highlight keywords
def highlight_text(text, query):
    if not query: return text
    # Extraer palabras de más de 3 letras para resaltar
    words = [re.escape(w) for w in query.split() if len(w) > 3]
    if not words: return text
    pattern = re.compile(f"({'|'.join(words)})", re.IGNORECASE)
    return pattern.sub(r'<mark style="background-color: #ffd700; color: black; border-radius: 3px; padding: 0 2px;">\1</mark>', text)

st.title("📄 Flipp")
st.markdown("Sube tus pdfs")

# Inicialización de estados
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "expanded_chunk" not in st.session_state:
    st.session_state.expanded_chunk = None

with st.sidebar:
    st.header("Búsqueda Rápida")
    semantic_query = st.text_input("Busca conceptos en todos los PDFs:")
    st.markdown("---")
    if st.button("Limpiar Base de Datos (Reiniciar)"):
        st.session_state.vector_store = None
        st.session_state.messages = []
        st.session_state.expanded_chunk = None
        st.rerun()

uploaded_files = st.file_uploader("Sube archivos PDF para analizar", type="pdf", accept_multiple_files=True)

if uploaded_files and st.session_state.vector_store is None:
    with st.spinner("Leyendo y procesando todos los documentos..."):
        all_splits = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_filename"] = uploaded_file.name

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)
            os.unlink(tmp_path)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        from langchain_core.vectorstores import InMemoryVectorStore
        st.session_state.vector_store = InMemoryVectorStore.from_documents(documents=all_splits, embedding=embeddings)
    
    st.success("¡Base de conocimiento creada! Ya puedes hacer preguntas o buscar.")

# Panel Lateral de Búsqueda
if st.session_state.vector_store is not None and semantic_query:
    with st.sidebar:
        st.write("**Coincidencias encontradas:**")
        results = st.session_state.vector_store.similarity_search(semantic_query, k=3)
        for i, res in enumerate(results):
            filename = res.metadata.get("source_filename", "Desconocido")
            page = res.metadata.get("page", "N/A")
            
            # Mostrar un fragmento corto
            snippet = res.page_content[:150] + "..."
            snippet_hl = highlight_text(snippet, semantic_query)
            
            st.info(f"**PDF:** {filename} (Pág. {page})")
            st.markdown(f"<div style='font-size:0.9em; margin-bottom: 10px;'>{snippet_hl}</div>", unsafe_allow_html=True)
            
            # Botón para expandir al centro
            if st.button("Ver texto completo", key=f"btn_expand_{i}"):
                st.session_state.expanded_chunk = {
                    "text": res.page_content,
                    "filename": filename,
                    "page": page,
                    "query": semantic_query
                }

if st.session_state.vector_store is not None:
    st.markdown("---")
    
    # Renderizar el fragmento expandido si existe
    if st.session_state.expanded_chunk:
        chunk = st.session_state.expanded_chunk
        st.subheader("🔍 Vista detallada del fragmento")
        with st.container():
            st.caption(f"**Fuente:** {chunk['filename']} | **Página:** {chunk['page']}")
            highlighted_full = highlight_text(chunk['text'], chunk['query'])
            st.markdown(f"<div style='background-color:#1e1e1e; padding:15px; border-radius:5px; border-left: 4px solid #ffd700;'>{highlighted_full}</div>", unsafe_allow_html=True)
            if st.button("✖ Cerrar detalle"):
                st.session_state.expanded_chunk = None
                st.rerun()
        st.markdown("---")

    st.subheader("Chat con tu documento")

    # Mostrar el historial del chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("Fuentes de información consultadas"):
                    for i, doc in enumerate(msg["sources"]):
                        filename = doc.metadata.get("source_filename", "Desconocido")
                        st.caption(f"Fragmento {i+1} - {filename} (Pág. {doc.metadata.get('page', 'N/A')}):")
                        hl_text = highlight_text(doc.page_content, msg.get("query", ""))
                        st.markdown(hl_text, unsafe_allow_html=True)

    user_query = st.chat_input("Escribe aquí tu pregunta...")

    if user_query:
        # Añadir al historial
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("user"):
            st.write(user_query)
        
        with st.spinner("Analizando múltiples documentos..."):
            llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)
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

            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            response = rag_chain.invoke({"input": user_query})

            # Mostrar respuesta
            with st.chat_message("assistant"):
                st.write(response["answer"])
                
                # Mostrar fuentes interactivas
                with st.expander("Fuentes de información consultadas"):
                    for i, doc in enumerate(response["context"]):
                        filename = doc.metadata.get("source_filename", "Desconocido")
                        st.caption(f"Fragmento {i+1} - {filename} (Pág. {doc.metadata.get('page', 'N/A')}):")
                        hl_text = highlight_text(doc.page_content, user_query)
                        st.markdown(hl_text, unsafe_allow_html=True)

            # Guardar en historial
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response["answer"],
                "sources": response["context"],
                "query": user_query
            })
