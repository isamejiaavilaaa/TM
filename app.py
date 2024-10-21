import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image
from keras.models import load_model
import platform

# Muestra la versión de Python junto con detalles adicionales
st.write("Versión de Python:", platform.python_version())

# Cargar el modelo entrenado
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("Reconocimiento de Imágenes")
st.subheader("Cierra y abre la mano para saber si el sistema lo detecta correctamente")

# Mostrar imagen de referencia
image = Image.open('OIG5.jpg')
st.image(image, width=350)

# Configuración de la barra lateral
with st.sidebar:
    st.subheader("Usando un modelo entrenado en Teachable Machine, puedes usar esta app para identificar gestos de la mano.")

# Captura de imagen desde la cámara
img_file_buffer = st.camera_input("Toma una Foto")

# Definir un umbral mínimo de confianza para las predicciones
confidence_threshold = 0.7

if img_file_buffer is not None:
    # Leer la imagen desde el buffer
    img = Image.open(img_file_buffer)

    # Redimensionar la imagen
    newsize = (224, 224)
    img = img.resize(newsize)

    # Convertir la imagen a un array numpy
    img_array = np.array(img)

    # Normalizar la imagen
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1

    # Cargar la imagen en el array
    data[0] = normalized_image_array

    # Realizar la predicción
    prediction = model.predict(data)

    # Mostrar las probabilidades para ambas predicciones
    prob_abierta = prediction[0][0]
    prob_cerrada = prediction[0][1]

    st.write(f"Probabilidad de Mano Abierta: {prob_abierta:.2f}")
    st.write(f"Probabilidad de Mano Cerrada: {prob_cerrada:.2f}")

    # Decisión basada en el umbral de confianza
    if prob_abierta > confidence_threshold:
        st.header(f'Mano Abierta detectada con alta confianza ({prob_abierta:.2f}) 😊')
    elif prob_cerrada > confidence_threshold:
        st.header(f'Mano Cerrada detectada con alta confianza ({prob_cerrada:.2f}) ✊')
    else:
        st.header('No se ha detectado con suficiente confianza si la mano está abierta o cerrada.')
