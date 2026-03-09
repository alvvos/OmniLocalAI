import streamlit as st
from openai import OpenAI
from duckduckgo_search import DDGS
import base64

# 1. Configuracion de la pagina limpia
st.set_page_config(page_title="Asistente Corporativo Multimodal", layout="wide")

estilo_oculto = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
"""
st.markdown(estilo_oculto, unsafe_allow_html=True)

cliente = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# 2. Inicializar memoria de conversaciones
if "chats" not in st.session_state:
    st.session_state.chats = {"Conversacion 1": []}
if "chat_actual" not in st.session_state:
    st.session_state.chat_actual = "Conversacion 1"

# 3. Panel Lateral: Selector de modelo, chats y subida de archivos
with st.sidebar:
    st.title("Configuración")
    
    # Selector de modelo IA
    modelo_seleccionado = st.selectbox(
        "Modelo de IA",
        ["llama3.1", "llama3.2-vision"],
        help="Usa Llama 3.1 para texto general y Llama 3.2 Vision si vas a subir imágenes."
    )
    st.markdown("---")
    
    st.title("Conversaciones")
    if st.button("Nueva Conversacion"):
        nuevo_nombre = f"Conversacion {len(st.session_state.chats) + 1}"
        st.session_state.chats[nuevo_nombre] = []
        st.session_state.chat_actual = nuevo_nombre
        st.rerun()

    chat_seleccionado = st.radio(
        "Historial", 
        list(st.session_state.chats.keys()), 
        index=list(st.session_state.chats.keys()).index(st.session_state.chat_actual)
    )
    
    if chat_seleccionado != st.session_state.chat_actual:
        st.session_state.chat_actual = chat_seleccionado
        st.rerun()

    st.markdown("---")
    st.markdown("Herramientas")
    
    usar_internet = st.checkbox("Buscar en Internet (Tiempo real)")
    archivo_texto = st.file_uploader("Adjuntar texto", type=["txt"])
    
    # Nueva opcion: Subida de imagenes
    imagen_adjunta = st.file_uploader("Adjuntar imagen", type=["png", "jpg", "jpeg"])

# 4. Interfaz principal
st.title(st.session_state.chat_actual)
st.markdown("---")

mensajes_actuales = st.session_state.chats[st.session_state.chat_actual]

# Mostrar historial de texto (ignoramos el contenido crudo de las imagenes para no saturar la pantalla)
for mensaje in mensajes_actuales:
    if mensaje["role"] != "system":
        # Extraemos solo el texto para mostrarlo limpio en el historial
        contenido_mostrar = mensaje["content"]
        if isinstance(contenido_mostrar, list):
            contenido_mostrar = contenido_mostrar[0]["text"]
            
        with st.chat_message(mensaje["role"]):
            st.markdown(contenido_mostrar)

# 5. Logica de procesamiento
if peticion := st.chat_input("Escribe tu consulta..."):
    
    # Preparar el contenido del usuario. Si hay imagen y el modelo es vision, usamos un formato especial.
    contenido_usuario = peticion
    
    if imagen_adjunta is not None and modelo_seleccionado == "llama3.2-vision":
        # Convertir imagen a Base64
        base64_image = base64.b64encode(imagen_adjunta.read()).decode("utf-8")
        tipo_mime = imagen_adjunta.type
        
        # Formato multimodal requerido por la API
        contenido_usuario = [
            {"type": "text", "text": peticion},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{tipo_mime};base64,{base64_image}"}
            }
        ]
    elif imagen_adjunta is not None and modelo_seleccionado != "llama3.2-vision":
         st.warning("Has subido una imagen pero estas usando Llama 3.1. Cambia el modelo a Llama 3.2 Vision en el panel lateral para procesarla.")

    # Guardar y mostrar la peticion
    mensajes_actuales.append({"role": "user", "content": contenido_usuario})
    with st.chat_message("user"):
        st.markdown(peticion)

    contexto_adicional = ""

    if usar_internet:
        with st.spinner("Buscando en internet..."):
            try:
                resultados = DDGS().text(peticion, max_results=3)
                texto_resultados = " ".join([r['body'] for r in resultados])
                contexto_adicional += f"Informacion de internet:\n{texto_resultados}\n\n"
            except Exception:
                pass

    if archivo_texto is not None:
        texto_archivo = archivo_texto.getvalue().decode("utf-8")
        contexto_adicional += f"Contexto del documento:\n{texto_archivo}\n\n"

    mensajes_ia = mensajes_actuales.copy()
    
    if contexto_adicional:
         instruccion_sistema = f"Utiliza esta informacion para responder:\n\n{contexto_adicional}"
         mensajes_ia.insert(-1, {"role": "system", "content": instruccion_sistema})

    # Generar respuesta
    with st.chat_message("assistant"):
        contenedor_respuesta = st.empty()
        respuesta_completa = ""
        
        try:
            flujo = cliente.chat.completions.create(
                model=modelo_seleccionado,
                messages=mensajes_ia,
                stream=True,
            )
            
            for fragmento in flujo:
                if fragmento.choices[0].delta.content is not None:
                    respuesta_completa += fragmento.choices[0].delta.content
                    contenedor_respuesta.markdown(respuesta_completa + "▌")
            
            contenedor_respuesta.markdown(respuesta_completa)
            mensajes_actuales.append({"role": "assistant", "content": respuesta_completa})
            
        except Exception as e:
            st.error(f"Error procesando la solicitud. Asegurate de haber descargado el modelo {modelo_seleccionado}.")