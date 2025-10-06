import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Détection Cancer Prostate",
    page_icon="🏥",
    layout="wide"
)


st.title("🏥 Détection du Cancer de la Prostate")
st.markdown("### Modèle de Régression Linéaire - Prédiction du PSA")
st.divider()


st.sidebar.title("📋 Navigation")
page = st.sidebar.radio(
    "Choisissez une section:",
    ["🏠 Accueil", "📊 Exploration des Données", "🤖 Entraînement du Modèle", "🔮 Prédiction", "📈 Comparaison des Modèles"]
)

st.sidebar.divider()
st.sidebar.info("👨‍⚕️ Application de ML pour la détection du cancer de la prostate basée sur les niveaux de PSA")


# ============================================================
# PAGE ACCUEIL
# ============================================================
if page == "🏠 Accueil":
    st.header("Bienvenue dans l'Application de Détection du Cancer de la Prostate")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📖 À propos du projet")
        st.markdown("""
        Cette application utilise des techniques de **régression linéaire** pour prédire 
        le niveau de PSA (Prostate-Specific Antigen) chez les hommes, un indicateur clé 
        pour la détection du cancer de la prostate.
        
        **Modèles disponibles:**
        - 🔵 Régression Linéaire Simple
        - 🟢 Régression Ridge
        - 🟡 Régression Lasso
        - 🟠 Elastic Net
        """)
    
    with col2:
        st.subheader("🎯 Variable cible")
        st.info("""
        **LPSA** : Log du Prostate-Specific Antigen
        
        Un niveau élevé de PSA peut indiquer:
        - Cancer de la prostate
        - Hyperplasie bénigne de la prostate
        - Inflammation de la prostate
        """)
    
    st.divider()
    
    st.subheader("🚀 Comment utiliser cette application ?")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### 1️⃣ Charger les données")
        st.write("Uploadez votre fichier de données (CSV ou TXT)")
    
    with col2:
        st.markdown("#### 2️⃣ Explorer")
        st.write("Analysez les corrélations et distributions")
    
    with col3:
        st.markdown("#### 3️⃣ Entraîner")
        st.write("Choisissez et entraînez votre modèle")
    
    with col4:
        st.markdown("#### 4️⃣ Prédire")
        st.write("Testez sur de nouveaux patients")


# ============================================================
# PAGE EXPLORATION DES DONNÉES
# ============================================================
elif page == "📊 Exploration des Données":
    st.header("📊 Exploration et Analyse des Données")
    
    # Upload du fichier
    fichier = st.file_uploader("📁 Uploadez votre fichier de données (CSV, TXT, Excel)", 
                                type=['csv', 'txt', 'xlsx'])
    
    # Option pour utiliser des données d'exemple
    use_example = st.checkbox("🧪 Utiliser des données d'exemple")
    
    if fichier is not None or use_example:
        try:
            if use_example:
                # Créer des données d'exemple similaires au dataset prostate
                np.random.seed(42)
                n_samples = 97
                df = pd.DataFrame({
                    'lcavol': np.random.normal(1.35, 1.18, n_samples),
                    'lweight': np.random.normal(3.63, 0.43, n_samples),
                    'age': np.random.randint(41, 80, n_samples),
                    'lbph': np.random.normal(0.1, 1.45, n_samples),
                    'svi': np.random.binomial(1, 0.22, n_samples),
                    'lcp': np.random.normal(-0.8, 1.4, n_samples),
                    'gleason': np.random.choice([6, 7, 8, 9], n_samples),
                    'pgg45': np.random.randint(0, 100, n_samples),
                    'lpsa': np.random.normal(2.48, 1.15, n_samples)
                })
                st.info("📌 Données d'exemple chargées (similaires au dataset prostate)")
            else:
                # Charger le fichier uploadé
                if fichier.name.endswith('.csv') or fichier.name.endswith('.txt'):
                    try:
                        df = pd.read_csv(fichier, sep='\t')
                    except:
                        df = pd.read_csv(fichier)
                else:
                    df = pd.read_excel(fichier)
                
                # Supprimer les colonnes 'col' et 'train' si elles existent
                if 'col' in df.columns:
                    df = df.drop('col', axis=1)
                if 'train' in df.columns:
                    df = df.drop('train', axis=1)
            
            # Sauvegarder dans session_state
            st.session_state['df'] = df
            st.success("✅ Données chargées avec succès!")
            
            # Informations de base
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📏 Lignes", df.shape[0])
            col2.metric("📊 Colonnes", df.shape[1])
            col3.metric("❌ Valeurs manquantes", df.isnull().sum().sum())
            col4.metric("🎯 Variable cible", "lpsa")
            
            st.divider()
            
            # Aperçu des données
            st.subheader("👀 Aperçu des données")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Statistiques descriptives
            st.subheader("📈 Statistiques descriptives")
            st.dataframe(df.describe(), use_container_width=True)
            
            # Corrélations avec lpsa
            st.subheader("🔗 Corrélations avec lpsa (Variable cible)")
            if 'lpsa' in df.columns:
                correlations = df.corr()['lpsa'].sort_values(ascending=False)
                
                fig = go.Figure(go.Bar(
                    x=correlations.values,
                    y=correlations.index,
                    orientation='h',
                    marker=dict(
                        color=correlations.values,
                        colorscale='RdBu',
                        showscale=True
                    )
                ))
                fig.update_layout(
                    title="Corrélations avec lpsa",
                    xaxis_title="Coefficient de corrélation",
                    yaxis_title="Variables",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Visualisations
            st.subheader("📊 Visualisations")
            
            tab1, tab2, tab3 = st.tabs(["📉 Distributions", "🔗 Matrice de Corrélation", "📦 Box Plots"])
            
            with tab1:
                colonnes_numeriques = df.select_dtypes(include=[np.number]).columns.tolist()
                col_dist = st.selectbox("Choisissez une variable", colonnes_numeriques, key="dist")
                fig = px.histogram(df, x=col_dist, nbins=30, 
                                  title=f"Distribution de {col_dist}",
                                  color_discrete_sequence=['#1f77b4'])
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                correlation_matrix = df.corr()
                fig = px.imshow(correlation_matrix, 
                               text_auto='.2f',
                               aspect="auto",
                               color_continuous_scale='RdBu_r',
                               title="Matrice de corrélation complète")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                col_box = st.selectbox("Choisissez une variable", colonnes_numeriques, key="box")
                fig = px.box(df, y=col_box, title=f"Box Plot de {col_box}",
                            color_discrete_sequence=['#2ca02c'])
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement: {e}")
    else:
        st.info("⬆️ Veuillez charger un fichier de données ou utiliser les données d'exemple")


# ============================================================
# PAGE ENTRAÎNEMENT DU MODÈLE
# ============================================================
elif page == "🤖 Entraînement du Modèle":
    st.header("🤖 Configuration et Entraînement du Modèle")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger les données dans la section 'Exploration des Données'")
    else:
        df = st.session_state['df']
        
        st.subheader("⚙️ Configuration du Modèle")
        
        # Paramètres d'entraînement
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_size = st.slider("Taille du set de test (%)", 10, 40, 20) / 100
        
        with col2:
            random_state = st.number_input("Random State", 0, 100, 42)
        
        with col3:
            normaliser = st.checkbox("Standardiser les données", value=True)
        
        # Type de modèle
        st.subheader("🎯 Choix du Modèle")
        type_modele = st.selectbox(
            "Sélectionnez le type de régression",
            ["Régression Linéaire Simple", "Ridge", "Lasso", "Elastic Net", "Tous les modèles (comparaison)"]
        )
        
        # Paramètres spécifiques
        params = {}
        if type_modele == "Ridge":
            col1, col2 = st.columns(2)
            with col1:
                params['alpha'] = st.number_input("Alpha (λ)", 0.01, 10.0, 2.14, 0.01)
            with col2:
                optimize = st.checkbox("Optimiser λ par validation croisée", value=False)
                if optimize:
                    params['optimize'] = True
        
        elif type_modele == "Lasso":
            col1, col2 = st.columns(2)
            with col1:
                params['alpha'] = st.number_input("Alpha (λ)", 0.001, 1.0, 0.08, 0.001)
            with col2:
                optimize = st.checkbox("Optimiser λ par validation croisée", value=False)
                if optimize:
                    params['optimize'] = True
        
        elif type_modele == "Elastic Net":
            col1, col2 = st.columns(2)
            with col1:
                params['alpha'] = st.number_input("Alpha (λ)", 0.001, 10.0, 0.1, 0.001)
            with col2:
                params['l1_ratio'] = st.slider("L1 Ratio (α)", 0.0, 1.0, 0.5, 0.1)
            optimize = st.checkbox("Optimiser les paramètres par validation croisée", value=False)
            if optimize:
                params['optimize'] = True
        
        st.divider()
        
        # Bouton d'entraînement
        if st.button("🚀 Entraîner le Modèle", type="primary", use_container_width=True):
            with st.spinner("⏳ Entraînement en cours..."):
                try:
                    # Préparation des données
                    X = df.drop('lpsa', axis=1)
                    y = df['lpsa']
                    
                    # Division train/test
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # Standardisation
                    if normaliser:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
                        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
                        st.session_state['scaler'] = scaler
                    else:
                        X_train_scaled = X_train
                        X_test_scaled = X_test
                    
                    results = {}
                    
                    # ============================================================
                    # ENTRAÎNEMENT DES MODÈLES
                    # ============================================================
                    
                    if type_modele == "Tous les modèles (comparaison)":
                        # Régression Linéaire
                        lr_model = LinearRegression()
                        lr_model.fit(X_train_scaled, y_train)
                        y_pred_lr = lr_model.predict(X_test_scaled)
                        results['Linéaire'] = {
                            'model': lr_model,
                            'y_pred': y_pred_lr,
                            'mse': mean_squared_error(y_test, y_pred_lr),
                            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
                            'mae': mean_absolute_error(y_test, y_pred_lr),
                            'r2': r2_score(y_test, y_pred_lr)
                        }
                        
                        # Ridge
                        ridge_model = Ridge(alpha=2.14)
                        ridge_model.fit(X_train_scaled, y_train)
                        y_pred_ridge = ridge_model.predict(X_test_scaled)
                        results['Ridge'] = {
                            'model': ridge_model,
                            'y_pred': y_pred_ridge,
                            'mse': mean_squared_error(y_test, y_pred_ridge),
                            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
                            'mae': mean_absolute_error(y_test, y_pred_ridge),
                            'r2': r2_score(y_test, y_pred_ridge)
                        }
                        
                        # Lasso
                        lasso_model = Lasso(alpha=0.08, max_iter=10000)
                        lasso_model.fit(X_train_scaled, y_train)
                        y_pred_lasso = lasso_model.predict(X_test_scaled)
                        results['Lasso'] = {
                            'model': lasso_model,
                            'y_pred': y_pred_lasso,
                            'mse': mean_squared_error(y_test, y_pred_lasso),
                            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
                            'mae': mean_absolute_error(y_test, y_pred_lasso),
                            'r2': r2_score(y_test, y_pred_lasso)
                        }
                        
                        # Elastic Net
                        elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
                        elastic_model.fit(X_train_scaled, y_train)
                        y_pred_elastic = elastic_model.predict(X_test_scaled)
                        results['Elastic Net'] = {
                            'model': elastic_model,
                            'y_pred': y_pred_elastic,
                            'mse': mean_squared_error(y_test, y_pred_elastic),
                            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_elastic)),
                            'mae': mean_absolute_error(y_test, y_pred_elastic),
                            'r2': r2_score(y_test, y_pred_elastic)
                        }
                        
                    else:
                        # Entraînement d'un seul modèle
                        if type_modele == "Régression Linéaire Simple":
                            model = LinearRegression()
                            model.fit(X_train_scaled, y_train)
                        
                        elif type_modele == "Ridge":
                            if params.get('optimize', False):
                                alphas_ridge = np.logspace(-3, 3, 100)
                                ridge_cv = GridSearchCV(
                                    Ridge(),
                                    param_grid={'alpha': alphas_ridge},
                                    cv=5,
                                    scoring='neg_mean_squared_error'
                                )
                                ridge_cv.fit(X_train_scaled, y_train)
                                best_alpha = ridge_cv.best_params_['alpha']
                                st.info(f"🎯 λ optimal trouvé: {best_alpha:.4f}")
                                model = Ridge(alpha=best_alpha)
                            else:
                                model = Ridge(alpha=params['alpha'])
                            model.fit(X_train_scaled, y_train)
                        
                        elif type_modele == "Lasso":
                            if params.get('optimize', False):
                                alphas_lasso = np.logspace(-4, 0, 100)
                                lasso_cv = GridSearchCV(
                                    Lasso(max_iter=10000),
                                    param_grid={'alpha': alphas_lasso},
                                    cv=5,
                                    scoring='neg_mean_squared_error'
                                )
                                lasso_cv.fit(X_train_scaled, y_train)
                                best_alpha = lasso_cv.best_params_['alpha']
                                st.info(f"🎯 λ optimal trouvé: {best_alpha:.4f}")
                                model = Lasso(alpha=best_alpha, max_iter=10000)
                            else:
                                model = Lasso(alpha=params['alpha'], max_iter=10000)
                            model.fit(X_train_scaled, y_train)
                        
                        elif type_modele == "Elastic Net":
                            if params.get('optimize', False):
                                param_grid_en = {
                                    'alpha': np.logspace(-4, 1, 50),
                                    'l1_ratio': np.linspace(0.1, 0.9, 9)
                                }
                                elastic_cv = GridSearchCV(
                                    ElasticNet(max_iter=10000),
                                    param_grid=param_grid_en,
                                    cv=5,
                                    scoring='neg_mean_squared_error',
                                    n_jobs=-1
                                )
                                elastic_cv.fit(X_train_scaled, y_train)
                                best_alpha = elastic_cv.best_params_['alpha']
                                best_l1 = elastic_cv.best_params_['l1_ratio']
                                st.info(f"🎯 Paramètres optimaux trouvés: λ={best_alpha:.4f}, α={best_l1:.4f}")
                                model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1, max_iter=10000)
                            else:
                                model = ElasticNet(alpha=params['alpha'], l1_ratio=params['l1_ratio'], max_iter=10000)
                            model.fit(X_train_scaled, y_train)
                        
                        y_pred = model.predict(X_test_scaled)
                        results[type_modele] = {
                            'model': model,
                            'y_pred': y_pred,
                            'mse': mean_squared_error(y_test, y_pred),
                            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                            'mae': mean_absolute_error(y_test, y_pred),
                            'r2': r2_score(y_test, y_pred)
                        }
                    
                    # Sauvegarder dans session_state
                    st.session_state['results'] = results
                    st.session_state['X_test'] = X_test_scaled
                    st.session_state['y_test'] = y_test
                    st.session_state['X_columns'] = X.columns
                    st.session_state['normaliser'] = normaliser
                    
                    # Affichage des résultats
                    st.success("✅ Modèle(s) entraîné(s) avec succès!")
                    
                    st.subheader("📊 Résultats")
                    
                    # Créer un tableau comparatif
                    comparison_df = pd.DataFrame({
                        'Modèle': list(results.keys()),
                        'R²': [results[m]['r2'] for m in results.keys()],
                        'MSE': [results[m]['mse'] for m in results.keys()],
                        'RMSE': [results[m]['rmse'] for m in results.keys()],
                        'MAE': [results[m]['mae'] for m in results.keys()]
                    })
                    st.dataframe(comparison_df.style.highlight_max(subset=['R²'], color='lightgreen')
                                                    .highlight_min(subset=['MSE', 'RMSE', 'MAE'], color='lightgreen'),
                                use_container_width=True)
                    
                    # Afficher les coefficients du meilleur modèle
                    if len(results) == 1:
                        model_name = list(results.keys())[0]
                        model = results[model_name]['model']
                        
                        if hasattr(model, 'coef_'):
                            st.subheader(f"📈 Coefficients du modèle {model_name}")
                            coef_df = pd.DataFrame({
                                'Variable': X.columns,
                                'Coefficient': model.coef_
                            }).sort_values('Coefficient', key=abs, ascending=False)
                            
                            fig = px.bar(coef_df, x='Coefficient', y='Variable',
                                        orientation='h',
                                        title=f"Importance des variables - {model_name}",
                                        color='Coefficient',
                                        color_continuous_scale='RdBu')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Afficher le nombre de variables sélectionnées pour Lasso
                            if 'Lasso' in model_name or 'Elastic' in model_name:
                                n_selected = np.sum(model.coef_ != 0)
                                st.info(f"🎯 Nombre de variables sélectionnées: {n_selected}/{len(X.columns)}")
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'entraînement: {e}")
                    st.exception(e)


# ============================================================
# PAGE PRÉDICTION
# ============================================================
elif page == "🔮 Prédiction":
    st.header("🔮 Faire des Prédictions")
    
    if 'results' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord entraîner un modèle dans la section 'Entraînement du Modèle'")
    else:
        results = st.session_state['results']
        X_columns = st.session_state['X_columns']
        
        # Sélection du modèle
        model_names = list(results.keys())
        selected_model_name = st.selectbox("📊 Sélectionnez le modèle à utiliser", model_names)
        selected_model = results[selected_model_name]['model']
        
        st.subheader("📝 Entrez les valeurs pour la prédiction")
        st.markdown("*Entrez les valeurs des variables explicatives pour prédire le niveau de lpsa*")
        
        # Créer des inputs pour chaque feature
        input_data = {}
        
        cols = st.columns(2)
        for i, feature in enumerate(X_columns):
            with cols[i % 2]:
                input_data[feature] = st.number_input(
                    f"**{feature}**",
                    value=0.0,
                    format="%.4f",
                    help=f"Entrez la valeur pour {feature}"
                )
        
        st.divider()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎯 Prédire", type="primary", use_container_width=True):
                try:
                    # Préparer les données
                    input_df = pd.DataFrame([input_data])
                    
                    # Normaliser si nécessaire
                    if st.session_state.get('normaliser', False) and 'scaler' in st.session_state:
                        input_df = st.session_state['scaler'].transform(input_df)
                    
                    # Prédiction
                    prediction = selected_model.predict(input_df)[0]
                    
                    # Afficher le résultat
                    st.success("✅ Prédiction effectuée avec succès!")
                    
                    # Affichage stylisé du résultat
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown("### 🎯 Résultat de la prédiction")
                        st.markdown(f"## **LPSA prédit: {prediction:.4f}**")
                        
                        # Interprétation
                        if prediction < 2.0:
                            st.info("📊 Niveau de PSA relativement bas")
                        elif prediction < 3.0:
                            st.warning("📊 Niveau de PSA modéré - surveillance recommandée")
                        else:
                            st.error("📊 Niveau de PSA élevé - consultation médicale recommandée")
                    
                    st.markdown("---")
                    st.caption("⚠️ Cette prédiction est fournie à titre informatif uniquement et ne remplace pas un diagnostic médical professionnel.")
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors de la prédiction: {e}")


# ============================================================
# PAGE COMPARAISON DES MODÈLES
# ============================================================
elif page == "📈 Comparaison des Modèles":
    st.header("📈 Évaluation et Comparaison des Modèles")
    
    if 'results' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord entraîner des modèles")
    else:
        results = st.session_state['results']
        y_test = st.session_state['y_test']
        
        # Comparaison des métriques
        st.subheader("📊 Comparaison des Performances")
        
        comparison_df = pd.DataFrame({
            'Modèle': list(results.keys()),
            'R²': [results[m]['r2'] for m in results.keys()],
            'MSE': [results[m]['mse'] for m in results.keys()],
            'RMSE': [results[m]['rmse'] for m in results.keys()],
            'MAE': [results[m]['mae'] for m in results.keys()]
        })
        
        # Graphique comparatif des R²
        fig_r2 = px.bar(comparison_df, x='Modèle', y='R²',
                       title="Comparaison des R² Scores",
                       color='R²',
                       color_continuous_scale='Viridis')
        st.plotly_chart(fig_r2, use_container_width=True)
        
        # Graphiques de performance pour chaque modèle
        st.subheader("📉 Analyse Détaillée par Modèle")
        
        for model_name in results.keys():
            with st.expander(f"🔍 {model_name}"):
                y_pred = results[model_name]['y_pred']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Prédictions vs Réalité
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=y_test, y=y_pred,
                        mode='markers',
                        name='Prédictions',
                        marker=dict(size=8, color='blue', opacity=0.6)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[y_test.min(), y_test.max()],
                        y=[y_test.min(), y_test.max()],
                        mode='lines',
                        name='Ligne parfaite',
                        line=dict(color='red', dash='dash')
                    ))
                    fig
