from flask import Flask, request, jsonify
import json
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import shap
import pickle

model = pickle.load(open('xgb_classif.pkl','rb'))

X_test = pickle.load(open('data_md.pkl', 'rb'))

X_t = X_test.drop(columns=['TARGET'])
y_t = X_test['TARGET']

app = Flask(__name__)

@app.route('/')
def home():
    return "Charger les données, le modèle, app .."


@app.route('/app/list_id/')
def list_id():
    id_list_client = pd.Series(list(X_t.index.sort_values()))
    id_list_client_js = json.loads(id_list_client.to_json())
    return jsonify({'data': id_list_client_js})


@app.route('/app/donnees_clients')
def donnees_clients():
    id_client = int(request.args.get('SK_ID_CURR'))
    X_client = X_t.loc[id_client: id_client]

    X_client = X_client[X_client.index==id_client]

    X_client_js = json.loads(X_client.to_json())
    return jsonify({'data': X_client_js})

def get_df_knn(id_client):
    NN = NearestNeighbors(n_neighbors=20)
    NN.fit(X_t)
    X_cust = X_t.loc[id_client:id_client]  # X_test
    idx = NN.kneighbors(X=X_cust,
                        n_neighbors=20,
                        return_distance=False).ravel()
    nearest_cust_idx = list(X_t.iloc[idx].index)
    # ----------------------------
    x_neigh = X_t.loc[nearest_cust_idx, :]
    y_neigh = y_t.loc[nearest_cust_idx]

    return x_neigh, y_neigh

@app.route('/app/clients_voisins')
def clients_voisins():
    id_client = int(request.args.get('SK_ID_CURR'))
    data_knn, y_knn = get_df_knn(id_client)
    data_knn_js = json.loads(data_knn.to_json())
    y_knn_js = json.loads(y_knn.to_json())
    return jsonify({
        'data_knn': data_knn_js,
        'y_knn': y_knn_js
    })


######
@app.route('/app/shap_value')
def shap_value():
    id_client = int(request.args.get('SK_ID_CURR'))
    X_neigh, y_neigh = get_df_knn(id_client)
    X_cust = X_t.loc[id_client:id_client]
    shap.initjs()

    explainer = shap.TreeExplainer(model)

    expected_vals = pd.Series(list(explainer.expected_value))
    shap_vals_cust = pd.Series(list(explainer.shap_values(X_cust)))
    shap_vals_neigh = pd.Series(list(explainer.shap_values(X_neigh)))
    X_neigh_js = json.loads(X_neigh.to_json())
    y_neigh_js = json.loads(y_neigh.to_json())

    expected_vals_js = json.loads(expected_vals.to_json())
    shap_vals_cust_js = json.loads(shap_vals_cust.to_json())
    shap_vals_neigh_js = json.loads(shap_vals_neigh.to_json())

    return jsonify({
        'X_neigh': X_neigh_js,
        'y_neigh': y_neigh_js,
        'expected_vals': expected_vals_js,
        'shap_vals_cust': shap_vals_cust_js,
        'shap_vals_neigh': shap_vals_neigh_js,

    })

@app.route('/app/scoring_client')
def scoring_client():
    id_client = int(request.args.get('SK_ID_CURR'))
    X_client = X_t.loc[id_client:id_client]
    score_client = model.predict_proba(X_client)[:, 1][0]
    score_client = json.dumps(float(score_client))
    return jsonify({
        'SK_ID_CURR': id_client,
        'score': score_client,
    })


@app.route('/app/feature_importance')
def feature_importance():
    feature_importance_values = model.feature_importances_

    feat_imp = pd.DataFrame({'feature': X_t.columns,
                           'importance': feature_importance_values})
    feat_imp = feat_imp.sort_values('importance',ascending=False).reset_index()
    feat_imp['importance_normalisee'] = feat_imp['importance']/feat_imp['importance'].sum()

    feat_imp_js = json.loads(feat_imp.to_json())
    return jsonify({
        'data': feat_imp_js
    })


if __name__ == "__main__":
    import os
    PORT = os.environ.get('PORT', 5000)
    app.run(host = '0.0.0.0', port=PORT)
# app.run(host="localhost", port=5000, debug=True)
