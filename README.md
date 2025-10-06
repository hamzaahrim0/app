# 🏥 Détection du Cancer de la Prostate

Cette application **Streamlit** permet de prédire le niveau de PSA (Prostate-Specific Antigen) chez les hommes, un indicateur clé pour la détection du cancer de la prostate, en utilisant des modèles de régression linéaire.

---

## 📋 Fonctionnalités

- 📊 **Exploration des données** : visualisations, statistiques descriptives, corrélations.
- 🤖 **Entraînement de modèles** :
  - Régression Linéaire
  - Régression Ridge
  - Régression Lasso
  - Elastic Net
  - Comparaison de tous les modèles
- 🔮 **Prédictions personnalisées** pour de nouveaux patients.
- 📈 **Comparaison des performances** des modèles (R², MSE, RMSE, MAE).

---

## ⚡ Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/hamzaahrim0/app.git
cd app

    Créer un environnement virtuel (optionnel) :

python3 -m venv venv
source venv/bin/activate

    Installer les dépendances :

pip install streamlit pandas numpy matplotlib seaborn scikit-learn plotly

🚀 Lancer l’application

streamlit run app.py

Ouvrir ensuite l’URL fournie par Streamlit (généralement http://localhost:8501) dans votre navigateur.
📝 Utilisation

    Accueil : présentation du projet et des variables.

    Exploration des données : upload ou utilisation de données d’exemple, visualisations et analyses.

    Entraînement du modèle : sélection du modèle, réglage des paramètres, entraînement.

    Prédiction : saisie des valeurs des variables pour obtenir la prédiction de LPSA.

    Comparaison des modèles : visualisation et comparaison des performances des modèles.

⚠️ Avertissement

Cette application est à titre informatif uniquement et ne remplace pas un diagnostic médical professionnel.
🔗 Auteur

Hamza Ahrim
GitHub
