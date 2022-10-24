import streamlit as st
import os
import requests
import json
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import shap
from PIL import Image
import plotly.graph_objects as go


PORT = os.environ.get('PORT', 5000)
FLASK_URL = "http://127.0.0.1:5000/app"
##
model = pickle.load(open('xgb_classif.pkl', 'rb'))
X_test = pickle.load(open('data_md.pkl', 'rb'))


st.title('Implémenter un modèle de scoring')
st.subheader("OUMAR CISSOKO  - Etudiant Data Scientist")

X_t = X_test.drop(columns = ['TARGET'])
y_t = X_test['TARGET']

image = Image.open('logo.png')
st.sidebar.image(image)
def main():
    @st.cache
    def list_id():
        url_id = FLASK_URL + '/list_id/'
        response = requests.get(url_id)
        #
        content = json.loads(response.content)
        id_client = pd.Series(content['data']).values.tolist()
        return id_client
    ids_list = list_id()
    choosen_id = st.sidebar.selectbox("Choisir ID",ids_list)
    st.subheader('ID sélectionné')
    st.write(choosen_id)
    @st.cache
    def donnee_client():
        url_donnes_clt = FLASK_URL + '/donnees_clients?SK_ID_CURR='+str(choosen_id)
        response = requests.get(url_donnes_clt)
        content = json.loads(response.content.decode('utf-8'))
        donnee_client = pd.DataFrame(content['data'])
        return donnee_client
    X_client =donnee_client()
    st.subheader('Caractéristiques du client choisit')
    st.write(X_client)

    @st.cache
    def score_model(id_client):
        score_url = FLASK_URL + "/scoring_client?SK_ID_CURR=" + str(id_client)
        response = requests.get(score_url)
        content = json.loads(response.content.decode('utf-8'))
        score_model = float((content['score']))
        return score_model
    score = score_model(choosen_id)
    st.subheader("Score du client")
    st.write(round(score*100,2),"%")

    if score < 0.5:
        st.write("CREDIT ACCORDE")
    else:
        st.write("CREDIT NON ACCORDE")

    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = score,
        mode = "gauge+number+delta",
        title = {'text': "Score du client choisit"},
        delta = {'reference': 0.5},
        gauge = {'axis': {'range': [None, 1]},
                 'steps' : [
                     {'range': [0, 0.5], 'color': "lightgray"},
                     {'range': [0.5, 1], 'color': "gray"}],
                 'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.5}}))

    st.write(fig)

    @st.cache
    def get_feat_importance():
        feat_imp_url = FLASK_URL + '/feature_importance'
        response = requests.get(feat_imp_url)
        content = json.loads(response.content)
        feat_imp = content['data']
        return feat_imp
    feat_imp = pd.DataFrame(get_feat_importance())

    ######
    @st.cache
    def get_shap_value(id_client):
        shap_url = FLASK_URL + "/shap_value?SK_ID_CURR=" + str(id_client)
        response = requests.get(shap_url)
        content = json.loads(response.content.decode('utf-8'))
        shap_val_dict = (content)
        return shap_val_dict
    shap_val_dict = get_shap_value(choosen_id)
    X_neigh = pd.DataFrame(shap_val_dict['X_neigh'])

    #######
    @st.cache
    def values_shap(selected_id):
        # URL of the sk_id API
        shap_values_api_url = FLASK_URL + "/shap_value?SK_ID_CURR=" + str(selected_id)
        # Requesting the API and saving the response
        response = requests.get(shap_values_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        shapvals = pd.DataFrame(content['shap_vals_cust'].values())
        expec_vals = pd.DataFrame(content['expected_vals'].values())
        return shapvals, expec_vals

    @st.cache
    def waterfall_plot(nb, ft, expected_val, shap_val):
        return shap.plots._waterfall.waterfall_legacy(expected_val, shap_val[0,:],
                                                        max_display=nb, feature_names=ft)

    if st.checkbox('Display waterfall local interpretation',key = 23):
        with st.spinner('SHAP waterfall plots displaying ...'):
            shap_vals,expected_vals = values_shap(choosen_id)
            features = X_t.columns
            Nb_features = st.slider("Nombre de features à afficher",2,25,10)
            waterfall_plot(Nb_features,features,expected_vals[0][0],shap_vals.values)
            plt.gcf()
            st.pyplot(plt.gcf())


    def feature_importances(df):
        # Ordonner les features par importance dans la construction du modèle
        df = pd.DataFrame(feat_imp)
        df = df.sort_values('importance', ascending=False).reset_index()

        plt.figure(figsize=(8, 6))
        fig, ax = plt.subplots()

        Nombre_features = st.slider('Nombre de features à afficher', 0, 40, 10)

        # Ranger les features par l'index le plus important en haut
        ax.barh(list(reversed(list(df.index[:Nombre_features]))),
                df['importance_normalisee'].head(Nombre_features),
                align='center', edgecolor='k')

        # paramétrer format et labels
        ax.set_yticks(list(reversed(list(df.index[:Nombre_features]))))
        ax.set_yticklabels(df['feature'].head(Nombre_features))

        # Afficher les résultats
        plt.xlabel('Normalized Importance')
        plt.title('Feature Importances')
        st.pyplot(fig)

        return df
    feature_importances(pd.DataFrame(model.feature_importances_))

    st.write('Boxplot des principales features')
    if st.checkbox("Show boxplot with selected columns"):
        # get the list of columns
        Nb_features = st.slider('Nombre de features à afficher', 0, 10, 3)
        columns = X_neigh.columns[:Nb_features].tolist()
        st.write("#### Select the columns to display:")
        selected_cols = st.multiselect("", columns)
        if len(selected_cols) > 0:
            fig, ax = plt.subplots(1,len(columns))
            for icol,col in enumerate(selected_cols):
                selected_df = X_neigh[col]
                data_client = X_client[col]
                data_globale = pd.DataFrame(X_t)[col]
                dt = [selected_df,data_client,data_globale]
                sns.boxplot(dt,ax=ax[icol])
            plt.title("Boxplot")
            plt.legend(labels=['X_neigh','data_client','data_globale'],loc=2,bbox_to_anchor=(1,1))
            st.pyplot(fig)


if __name__ == '__main__':
    main()
