import streamlit as st

st.set_page_config(page_title="Análisis de Opiniones", layout="wide")

import pandas as pd
import json
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import string
from transformers import pipeline

nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

# Modelos de Hugging Face
@st.cache_resource
def cargar_modelos():
    clasificador = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    resumidor = pipeline('summarization', model='facebook/bart-large-cnn')
    return clasificador, resumidor

clasificador, resumidor = cargar_modelos()

# Función para clasificar sentimiento
def clasifica_sentimiento(text):
    res = clasificador(text[:512])[0]['label']
    if res in ['1 star', '2 stars']:
        return 'Negativo'
    elif res == '3 stars':
        return 'Neutro'
    else:
        return 'Positivo'

# Función para resumen
def resumir(text):
    return resumidor(text[:1024], max_length=40, min_length=10, do_sample=False)[0]['summary_text']

# App UI
st.title("🫓 Análisis de Opiniones sobre Tienda de Arepas")

# Cargar archivo JSON
archivo = st.file_uploader("📄 Sube un archivo JSON con comentarios", type=["json"])

if archivo:
    data = json.load(archivo)
    df = pd.DataFrame(data)

    st.subheader("🔍 Comentarios cargados")
    st.dataframe(df)

    # Clasificar sentimientos
    st.subheader("💬 Clasificación de Sentimientos")
    df['sentimiento'] = df['comentario'].apply(clasifica_sentimiento)
    st.dataframe(df[['comentario', 'sentimiento']])

    # Gráfico de barras
    conteo = df['sentimiento'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.bar(conteo.index, conteo.values, color=["green", "gray", "red"])
    ax1.set_title("Cantidad de opiniones por clase de sentimiento")
    st.pyplot(fig1)

    # Gráfico de pastel
    fig2, ax2 = plt.subplots()
    ax2.pie(conteo, labels=conteo.index, autopct="%1.1f%%", startangle=140)
    ax2.axis("equal")
    st.pyplot(fig2)

    # Top 10 palabras
    st.subheader("📊 Top 10 Palabras Más Frecuentes y Nube de Palabras")
    all_text = ' '.join(df['comentario'].astype(str)).lower()
    all_text = all_text.translate(str.maketrans('', '', string.punctuation))
    palabras = all_text.split()
    palabras_filtradas = [p for p in palabras if p not in stop_words and len(p) > 2]
    conteo_palabras = Counter(palabras_filtradas).most_common(10)

    if conteo_palabras:
        palabras_top, frecuencias_top = zip(*conteo_palabras)
        fig3, ax3 = plt.subplots()
        ax3.bar(palabras_top, frecuencias_top)
        ax3.set_title("Top 10 palabras más frecuentes")
        plt.xticks(rotation=45)
        st.pyplot(fig3)

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(palabras_filtradas))
        st.image(wordcloud.to_array())

    # Análisis de nuevo comentario
    st.subheader("🧠 Analiza un nuevo comentario")
    nuevo = st.text_area("✍️ Escribe tu comentario aquí")

    if st.button("Analizar"):
        if nuevo.strip() == "":
            st.warning("Por favor, escribe un comentario primero.")
        else:
            sentimiento = clasifica_sentimiento(nuevo)
            resumen = resumir(nuevo)
            st.success(f"➡️ Sentimiento: {sentimiento}")
            st.info(f"📝 Resumen: {resumen}")
