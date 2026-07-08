import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
import pandas as pd  # <-- LA NUEVA HERRAMIENTA PARA LEER GOOGLE SHEETS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- CONFIGURACIÓN DE BASE DE DATOS (GOOGLE SHEETS) ---
# Aquí está tu enlace exacto de Google Sheets
URL_GOOGLE_SHEET = "https://docs.google.com/spreadsheets/d/1CEpQSW6bcnNve92DUyuSa38-c9-fXoQLSiQ2zGADKIU/edit?usp=sharing"

@st.cache_data(ttl=60) # Actualiza la lista de contraseñas cada 60 segundos
def cargar_usuarios():
    # El código transforma mágicamente tu enlace de lectura a un formato de datos puros (CSV)
    url_base = URL_GOOGLE_SHEET.split('/edit')[0]
    url_csv = f"{url_base}/export?format=csv"
    
    # Descarga la tabla y la convierte en texto plano
    df = pd.read_csv(url_csv)
    # Limpiamos espacios en blanco por si se pasaron al tipear en el Excel
    df['correo'] = df['correo'].astype(str).str.strip()
    df['contrasena'] = df['contrasena'].astype(str).str.strip()
    return df

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
    
    correo = st.text_input("Correo electrónico").strip()
    contrasena = st.text_input("Contraseña", type="password").strip()
    
    if st.button("Entrar"):
        try:
            usuarios_df = cargar_usuarios()
            # Buscamos si el correo y la contraseña hacen match exacto en tu Excel
            match = usuarios_df[(usuarios_df['correo'] == correo) & (usuarios_df['contrasena'] == contrasena)]
            
            if not match.empty:
                st.session_state.autenticado = True
                st.rerun()
            else:
                st.error("Credenciales incorrectas. El usuario no existe o la contraseña está mal escrita.")
        except Exception as e:
            st.error(f"Error al conectar con Google Sheets. Verifica que el enlace sea público. Detalle: {e}")

else:
    # --- LA FÁBRICA DE VECTORES DIRECTA (MULTI-FORMATO) ---
    def procesar_documento(archivo):
        texto_completo = ""
        nombre_archivo = archivo.name.lower()
        
        if nombre_archivo.endswith('.pdf'):
            lector = PdfReader(archivo)
            for pagina in lector.pages:
                texto_extraido = pagina.extract_text()
                if texto_extraido:
                    texto_completo += texto_extraido
                    
        elif nombre_archivo.endswith('.docx'):
            doc = docx.Document(archivo)
            for parrafo in doc.paragraphs:
                texto_completo += parrafo.text + "\n"
                
        elif nombre_archivo.endswith('.txt'):
            texto_completo = archivo.getvalue().decode("utf-8")
        
        if not texto_completo.strip():
            return None
        
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
