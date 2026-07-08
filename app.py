import streamlit as st
from openai import OpenAI
import os
import glob # <-- Nueva herramienta para leer carpetas enteras
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- CONFIGURACIÓN DE BASE DE DATOS (GOOGLE SHEETS) ---
URL_GOOGLE_SHEET = "https://docs.google.com/spreadsheets/d/1CEpQSW6bcnNve92DUyuSa38-c9-fXoQLSiQ2zGADKIU/edit?usp=sharing"

@st.cache_data(ttl=60)
def cargar_usuarios():
    url_base = URL_GOOGLE_SHEET.split('/edit')[0]
    url_csv = f"{url_base}/export?format=csv"
    df = pd.read_csv(url_csv)
    df['correo'] = df['correo'].astype(str).str.strip()
    df['contrasena'] = df['contrasena'].astype(str).str.strip()
    return df

# --- EL CEREBRO GLOBAL (LEE LA CARPETA AUTOMÁTICAMENTE) ---
# st.cache_resource hace que el bot lea los archivos solo 1 vez y comparta esta memoria con todos los celulares
@st.cache_resource(show_spinner="Entrenando al asistente con los documentos oficiales del colegio...")
def construir_cerebro_global():
    # Busca todos los archivos dentro de la carpeta "documentos"
    rutas_archivos = glob.glob("documentos/*")
    
    if not rutas_archivos:
        return None
        
    texto_completo = ""
    
    for ruta in rutas_archivos:
        nombre_archivo = ruta.lower()
        try:
            if nombre_archivo.endswith('.pdf'):
                lector = PdfReader(ruta)
                for pagina in lector.pages:
                    texto_extraido = pagina.extract_text()
                    if texto_extraido:
                        texto_completo += texto_extraido + "\n"
                        
            elif nombre_archivo.endswith('.docx'):
                doc = docx.Document(ruta)
                for parrafo in doc.paragraphs:
                    texto_completo += parrafo.text + "\n"
                    
            elif nombre_archivo.endswith('.txt'):
                with open(ruta, "r", encoding="utf-8") as f:
                    texto_completo += f.read() + "\n"
        except Exception as e:
            print(f"Error leyendo el archivo {ruta}: {e}")
            
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

# --- CONFIGURACIÓN DE IA ---
load_dotenv()
cliente = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

st.set_page_config(page_title="Asistente Escolar Pro", layout="wide")

# Construimos la memoria global antes de que el usuario haga nada
memoria_global = construir_cerebro_global()

# --- CONTROL DE ACCESO ---
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    st.title("🔐 Acceso para Apoderados")
    st.write("Por favor, inicia sesión para ingresar al asistente del colegio.")
    
    correo = st.text_input("Correo electrónico").strip()
    contrasena = st.text_input("Contraseña", type="password").strip()
    
    if st.button("Entrar"):
        try:
            usuarios_df = cargar_usuarios()
            match = usuarios_df[(usuarios_df['correo'] == correo) & (usuarios_df['contrasena'] == contrasena)]
            
            if not match.empty:
                st.session_state.autenticado = True
                st.rerun()
            else:
                st.error("Credenciales incorrectas. Verifica tu acceso.")
        except Exception as e:
            st.error("Error al conectar con la base de datos de apoderados.")

else:
    # Interfaz limpia para el apoderado (sin subida de archivos)
    with st.sidebar:
        st.write("Conectado de forma segura.")
        if st.button("Cerrar Sesión"):
            st.session_state.autenticado = False
            st.rerun()

    st.title("🤖 Asistente Inteligente del Colegio")
    
    if memoria_global is None:
        st.info("ℹ️ El administrador aún no ha cargado documentos oficiales en el repositorio.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Pregunta sobre normativas, circulares o eventos..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            system_prompt = "Eres el asistente escolar oficial. Responde de manera formal y amable. "
            
            if memoria_global is not None:
                vector_pregunta = memoria_global["modelo"].encode([prompt])
                distancias, indices = memoria_global["indice"].search(np.array(vector_pregunta), k=3)
                pedazos_encontrados = [memoria_global["pedazos"][i] for i in indices[0] if i < len(memoria_global["pedazos"])]
                
                contexto_encontrado = "\n\n".join(pedazos_encontrados)
                system_prompt += f"Responde ÚNICAMENTE basado en estos fragmentos de los documentos oficiales del colegio:\n\n{contexto_encontrado}\n\nSi la respuesta no está en el texto, indica amablemente que no tienes esa información."
            else:
                system_prompt += "Indica que no hay información disponible en este momento."

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
                st.error(f"Error de conexión: {e}")
