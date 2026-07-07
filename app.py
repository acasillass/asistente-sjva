import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx  # El nuevo traductor para Word
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import sqlite3

# --- CONFIGURACIÓN DE BASE DE DATOS LOCAL (SQLite) ---
conn = sqlite3.connect('usuarios.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS apoderados (correo TEXT, contrasena TEXT)''')

# Inyectamos tu usuario de prueba si la base de datos está vacía
c.execute("SELECT * FROM apoderados")
if not c.fetchall():
    c.execute("INSERT INTO apoderados (correo, contrasena) VALUES ('casillas.alvaro@gmail.com', '12345678')")
    conn.commit()

# --- CONFIGURACIÓN DE IA ---
load_dotenv()
cliente = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

st.set_page_config(page_title="Asistente Escolar Pro", layout="wide")

# --- CONTROL DE ACCESO ---
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    st.title("🔐 Acceso para Apoderados")
    st.write("Por favor, inicia sesión para ingresar al asistente.")
    
    correo = st.text_input("Correo electrónico")
    contrasena = st.text_input("Contraseña", type="password")
    
    if st.button("Entrar"):
        c.execute("SELECT * FROM apoderados WHERE correo=? AND contrasena=?", (correo, contrasena))
        usuario_encontrado = c.fetchone()
        
        if usuario_encontrado:
            st.session_state.autenticado = True
            st.rerun()
        else:
            st.error("Credenciales incorrectas. El usuario no existe en la base de datos local.")

else:
    # --- LA FÁBRICA DE VECTORES DIRECTA (MULTI-FORMATO) ---
    def procesar_documento(archivo):
        texto_completo = ""
        nombre_archivo = archivo.name.lower()
        
        # 1. Traductor para PDF
        if nombre_archivo.endswith('.pdf'):
            lector = PdfReader(archivo)
            for pagina in lector.pages:
                texto_extraido = pagina.extract_text()
                if texto_extraido:
                    texto_completo += texto_extraido
                    
        # 2. Traductor para Word (.docx)
        elif nombre_archivo.endswith('.docx'):
            doc = docx.Document(archivo)
            for parrafo in doc.paragraphs:
                texto_completo += parrafo.text + "\n"
                
        # 3. Traductor para Texto (.txt)
        elif nombre_archivo.endswith('.txt'):
            # Los txt vienen en bytes, hay que decodificarlos a texto normal
            texto_completo = archivo.getvalue().decode("utf-8")
        
        # Si el documento estaba en blanco
        if not texto_completo.strip():
            return None
        
        # Cortamos el texto y lo convertimos a matemáticas (vectores)
        separador = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        pedazos = separador.split_text(texto_completo)
        
        modelo = SentenceTransformer("all-MiniLM-L6-v2")
        vectores = modelo.encode(pedazos)
        
        dimension = vectores.shape[1]
        indice = faiss.IndexFlatL2(dimension)
        indice.add(np.array(vectores))
        
        return {"indice": indice, "pedazos": pedazos, "modelo": modelo}

    # --- INTERFAZ DE USUARIO ---
    with st.sidebar:
        st.title("⚙️ Administración")
        st.write("Bienvenido al panel.")
        if st.button("Cerrar Sesión"):
            st.session_state.autenticado = False
            st.rerun()
            
        # Ampliamos el uploader para que acepte los tres formatos
        archivo_subido = st.file_uploader("Sube el comunicado", type=["pdf", "docx", "txt"])
        
        if archivo_subido:
            with st.spinner("Procesando documento..."):
                memoria = procesar_documento(archivo_subido)
                if memoria is not None:
                    st.session_state.memoria_vectores = memoria
                    st.success(f"✅ Memoria creada exitosamente desde: {archivo_subido.name}")
                else:
                    st.error("⚠️ El archivo está vacío o es una imagen escaneada sin texto.")

    st.title("🤖 Asistente Inteligente del Colegio")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Pregunta sobre el colegio..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            system_prompt = "Eres el asistente escolar de un colegio. Sé amable y claro. "
            
            if "memoria_vectores" in st.session_state:
                memoria = st.session_state.memoria_vectores
                vector_pregunta = memoria["modelo"].encode([prompt])
                distancias, indices = memoria["indice"].search(np.array(vector_pregunta), k=3)
                pedazos_encontrados = [memoria["pedazos"][i] for i in indices[0] if i < len(memoria["pedazos"])]
                
                contexto_encontrado = "\n\n".join(pedazos_encontrados)
                system_prompt += f"Responde ÚNICAMENTE basado en esta información oficial:\n\n{contexto_encontrado}"
            else:
                system_prompt += "Indica que aún no tienes documentos cargados para responder."

            try:
                response = cliente.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                full_response = response.choices[0].message.content
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error de conexión con la IA: {e}")