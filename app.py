import streamlit as st
import os
from groq import Groq
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from PIL import Image
import labvision  # Suponiendo que este archivo contiene funciones para procesar imágenes médicas
from apikey import groq_apikey

# Configuración de la API key de Groq
os.environ['GROQ_API_KEY'] = groq_apikey


def main():
    # Personalización del logo y diseño
    st.image('MedIA_Logo.png', width=200)
    st.title("Bienvenido a MedAI Multimodal")
    st.write("¡Hola! Soy MedAI, un chatbot que maneja consultas de texto e imágenes médicas.")
    
    # Opciones de personalización en la barra lateral
    st.sidebar.title('Opciones de Personalización')
    system_prompt = st.sidebar.text_input("Prompt del Sistema:", value="¡Hola! Soy un chatbot de atención médica.")
    model = st.sidebar.selectbox(
        'Escoja un Modelo',
        ['llama3-8b-8192', 'llama3-70b-8192','mixtral-8x7b-32768', 'gemma-7b-it']
    )
    conversational_memory_length = st.sidebar.slider('Longitud de la memoria conversacional:', 1, 10, value=5)
    
    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    # Elegir el tipo de chatbot
    chatbot_type = st.sidebar.selectbox(
        "Seleccione el tipo de Chatbot:",
        ["Análisis de Medicamentos", "Generador de Quiz"]
    )

    # Cargar imagen
    image = st.file_uploader("Subir una imagen del medicamento (formato JPG/PNG):", type=["jpg", "png"])

    user_question = st.text_input("Escribe tu consulta o deja que el bot genere preguntas:")

    # Estado de la sesión
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Inicializar el objeto de chat para Groq con Langchain
    groq_chat = ChatGroq(
        groq_api_key=os.environ['GROQ_API_KEY'],
        model_name=model
    )

    # Procesar la imagen si está disponible
    if image:
        img = Image.open(image)
        st.image(img, caption="Imagen del Medicamento", use_column_width=True)
        
        # Aquí se llama a la función que procesa imágenes (debe estar definida en labvision.py)
        processed_image_data = labvision.process_image(img)

    if chatbot_type == "Análisis de Medicamentos" and image:
        st.write("**Análisis de la Imagen del Medicamento**")
        system_prompt = "Eres un asistente médico que analiza imágenes de medicamentos, proporcionando características, usos y contraindicaciones."

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}")
            ]
        )

        if user_question:
            user_input = f"Describe las características, usos y contraindicaciones de este medicamento: {processed_image_data}."
            conversation = LLMChain(
                llm=groq_chat,
                prompt=prompt,
                verbose=True,
                memory=memory,
            )
            response = conversation.predict(human_input=user_input)
            st.session_state.chat_history.append({'human': user_input, 'AI': response})
            st.write("MedAI:", response)

    elif chatbot_type == "Generador de Quiz" and image:
        st.write("**Generador de Preguntas de Quiz**")
        system_prompt = "Eres un asistente educativo que genera preguntas de quiz basadas en imágenes médicas para estudiantes."

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}")
            ]
        )

        if user_question:
            user_input = f"Genera preguntas relacionadas con la imagen de este medicamento: {processed_image_data}."
            conversation = LLMChain(
                llm=groq_chat,
                prompt=prompt,
                verbose=True,
                memory=memory,
            )
            response = conversation.predict(human_input=user_input)
            st.session_state.chat_history.append({'human': user_input, 'AI': response})
            st.write("MedAI:", response)


if __name__ == "__main__":
    main()
