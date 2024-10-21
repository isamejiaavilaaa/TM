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
    st.subheader("Usando un modelo entrenado en Teachable Machine puedes usar esta app para identificar gestos de la mano.")

# Captura de imagen desde la cámara
img_file_buffer = st.camera_input("Toma una Foto")

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

    # Mostrar el resultado de la predicción
    if prediction[0][0] > 0.5:
        st.header('Mano Abierta, con probabilidad: ' + str(prediction[0][0]))
    elif prediction[0][1] > 0.5:
        st.header('Mano Cerrada, con probabilidad: ' + str(prediction[0][1]))
    else:
        st.header('No se ha detectado la mano correctamente.')
