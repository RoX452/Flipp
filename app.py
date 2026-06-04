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

# Helper function to highlight keywords with a translucent green capsule
def highlight_text(text, query):
    if not query: return text
    
    stop_words = {"para", "como", "este", "esta", "estos", "estas", "pero", "porque", "cuando", "donde", "quien", "resumeme", "explicame", "dime", "cual", "sobre", "aquel", "aquella", "tiene"}
    # Extraer palabras clave (ignorar palabras comunes y cortas)
    words = [re.escape(w) for w in query.split() if len(w) > 3 and w.lower() not in stop_words]
    
    if not words: return text
    pattern = re.compile(f"({'|'.join(words)})", re.IGNORECASE)
    
    # Cápsula verde translúcida y elegante
    capsule_style = "background-color: rgba(46, 204, 113, 0.2); border: 1px solid rgba(46, 204, 113, 0.6); border-radius: 12px; padding: 2px 8px; color: inherit; font-weight: 500;"
    return pattern.sub(rf'<mark style="{capsule_style}">\1</mark>', text)

st.title("📄 Flipp")
st.markdown("Sube tus pdfs")

# Inicialización de estados
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "expanded_chunk" not in st.session_state:
    st.session_state.expanded_chunk = None
if "last_semantic_query" not in st.session_state:
    st.session_state.last_semantic_query = ""

# Variable global para guardar las coincidencias de la barra lateral
valid_results = []

with st.sidebar:
    st.header("Búsqueda Rápida")
    semantic_query = st.text_input("Escribe una palabra o concepto:")
    
    # Si el usuario busca una nueva palabra, limpiamos el bloque expandido viejo
    if semantic_query != st.session_state.last_semantic_query:
        st.session_state.expanded_chunk = None
        st.session_state.last_semantic_query = semantic_query
        
    st.markdown("---")
    if st.button("Limpiar Base de Datos", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.all_chunks = []
        st.session_state.messages = []
        st.session_state.expanded_chunk = None
        st.session_state.last_semantic_query = ""
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

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)
            os.unlink(tmp_path)

        st.session_state.all_chunks = all_splits

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        from langchain_core.vectorstores import InMemoryVectorStore
        st.session_state.vector_store = InMemoryVectorStore.from_documents(documents=all_splits, embedding=embeddings)
    
    st.success("Ya puedes hacer preguntas o buscar.")

# Panel Lateral de Búsqueda
if st.session_state.vector_store is not None and semantic_query:
    with st.sidebar:
        st.write(f"**Coincidencias encontradas:** *(Buscando en {len(st.session_state.all_chunks)} fragmentos)*")
        
        query_words = semantic_query.lower().split()
        
        for chunk in st.session_state.all_chunks:
            chunk_text = chunk.page_content.lower()
            if all(w in chunk_text for w in query_words):
                valid_results.append(chunk)
                if len(valid_results) >= 10:  
                    break
        
        if not valid_results:
            st.warning(f"No se encontró la palabra '{semantic_query}' exactamente escrita así en ningún PDF.")
            if len(st.session_state.all_chunks) == 0:
                st.error("Error: La memoria está vacía. Por favor elimina los PDFs y súbelos de nuevo.")
        
        for i, res in enumerate(valid_results):
            filename = res.metadata.get("source_filename", "Desconocido")
            page = res.metadata.get("page", "N/A")
            
            snippet = res.page_content[:120] + "..."
            snippet_hl = highlight_text(snippet, semantic_query)
            
            with st.container(border=True):
                st.caption(f"📄 {filename} (Pág. {page})")
                st.markdown(f"<div style='font-size:0.85em; margin-bottom: 10px; color: #a0a0a0;'>{snippet_hl}</div>", unsafe_allow_html=True)
                
                if st.button("Abrir fragmento 📖", key=f"btn_expand_{i}", use_container_width=True):
                    st.session_state.expanded_chunk = {
                        "text": res.page_content,
                        "filename": filename,
                        "page": page,
                        "query": semantic_query
                    }

if st.session_state.vector_store is not None:
    st.markdown("---")
    
    if st.session_state.expanded_chunk:
        chunk = st.session_state.expanded_chunk
        
        st.markdown(f"**📖 Viendo detalle de:** `{chunk['filename']}` *(Pág. {chunk['page']})*")
                
        highlighted_full = highlight_text(chunk['text'], chunk['query'])
        st.markdown(
            f"""
            <div style='background-color: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 8px; border-left: 5px solid #2ecc71; font-style: italic; margin-bottom: 10px;'>
                {highlighted_full}
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        if st.button("← Ocultar detalle"):
            st.session_state.expanded_chunk = None
            st.rerun()

    col_chat1, col_chat2 = st.columns([0.8, 0.2])
    with col_chat1:
        st.subheader("Chat con tu documento")
    with col_chat2:
        if st.button("🗑️ Limpiar Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Mostrar el historial del chat con AVATARES PERSONALIZADOS
    for msg in st.session_state.messages:
        avatar_icon = "🧑‍💻" if msg["role"] == "user" else "✨"
        with st.chat_message(msg["role"], avatar=avatar_icon):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("Ver fuentes de esta respuesta"):
                    for i, doc in enumerate(msg["sources"]):
                        filename = doc.metadata.get("source_filename", "Desconocido")
                        st.caption(f"Fragmento {i+1} - {filename} (Pág. {doc.metadata.get('page', 'N/A')}):")
                        st.markdown(doc.page_content)

    user_query = st.chat_input("Escribe aquí tu pregunta...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(user_query)
        
        with st.spinner("Analizando documentos..."):
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

            # LÓGICA CRÍTICA: Forzar a Gemini a usar SOLO las coincidencias de la barra lateral si existen
            if semantic_query and valid_results:
                context_docs = valid_results
            else:
                # Si no hay búsqueda en la barra lateral, hacemos búsqueda semántica normal pero con k=3 para no traer basura de otros PDFs
                context_docs = st.session_state.vector_store.similarity_search(user_query, k=3)

            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            
            response_answer = question_answer_chain.invoke({
                "context": context_docs,
                "input": user_query
            })

            with st.chat_message("assistant", avatar="✨"):
                st.write(response_answer)
                
                with st.expander("Ver fuentes de esta respuesta"):
                    for i, doc in enumerate(context_docs):
                        filename = doc.metadata.get("source_filename", "Desconocido")
                        st.caption(f"Fragmento {i+1} - {filename} (Pág. {doc.metadata.get('page', 'N/A')}):")
                        st.markdown(doc.page_content)

            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_answer,
                "sources": context_docs,
                "query": user_query
            })
