import streamlit as st
import os
from apikey import groq_apikey
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Configurar la clave API de Groq
os.environ['GROQ_API_KEY'] = groq_apikey

def main():
    st.image('MedIA_Logo.png', width=200)  # Logo personalizado
    st.title("Bienvenido a MedAI")
    st.write("¡Hola! Soy MedAI, un chatbot diseñado para brindar atención médica virtual. ¿En qué puedo ayudarte hoy?")

    # Configuración de personalización en la barra lateral
    system_prompt = st.sidebar.text_input("Prompt del Sistema:", value="¡Hola! Soy un chatbot de atención médica.")
    model = st.sidebar.selectbox('Escoja un Modelo', ['llama3-8b-8192', 'llama3-70b-8192'])
    conversational_memory_length = st.sidebar.slider('Longitud de la memoria conversacional:', 1, 10, value=5)
    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    # Opción para seleccionar el modo de consulta (información o quiz)
    mode = st.sidebar.selectbox("Selecciona el tipo de consulta", ["Información sobre medicamento", "Generar Quiz"])

    user_question = st.text_input("Escribe tu consulta médica:")
    image_file = st.file_uploader("Sube una imagen de un medicamento", type=["jpg", "jpeg", "png"])

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Inicialización del objeto de chat para Groq con LangChain
    groq_chat = ChatGroq(
        groq_api_key=os.environ['GROQ_API_KEY'],
        model_name=model
    )

    if user_question or image_file:
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}")
            ]
        )

        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=True,
            memory=memory,
        )

        # Procesar imagen si está presente
        if image_file:
            st.image(image_file, caption='Imagen del medicamento subido', use_column_width=True)

            # Supón que se extrae el nombre del medicamento de la imagen
            medication_name = "Paracetamol"  # Este sería el resultado del análisis de imagen

            # Lógica para generar información o quiz
            if mode == "Información sobre medicamento":
                # Generar la respuesta usando LangChain y Groq para información del medicamento
                response = conversation.predict(human_input=f"Cuáles son las características, usos y contraindicaciones del medicamento {medication_name}?")
            else:
                # Generar preguntas tipo quiz usando LangChain y Groq
                response = conversation.predict(human_input=f"Genera 3 preguntas tipo quiz sobre el medicamento {medication_name} para estudiantes de medicina o farmacia.")
        else:
            # Si solo hay consulta en texto
            response = conversation.predict(human_input=user_question)

        # Mostrar la respuesta en Streamlit
        st.write("MedAI:", response)

if __name__ == "__main__":
    main()
