import streamlit as st
import pandas as pd
import numpy as np

# Titre de l'app
st.title("🎉 Test de Streamlit")

# Texte simple
st.write("Ceci est une petite app Streamlit de test sur Pop!_OS.")

# Un dataframe aléatoire
df = pd.DataFrame(
    np.random.randn(10, 3),
    columns=["Colonne A", "Colonne B", "Colonne C"]
)

st.subheader("📊 Voici un DataFrame")
st.dataframe(df, width="stretch")  # version corrigée (pas use_container_width)

# Un graphique de ligne
st.subheader("📈 Graphique de test")
st.line_chart(df)

# Un bouton interactif
if st.button("Clique ici"):
    st.success("Bravo, ton bouton fonctionne ")
