import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pydot
from io import StringIO
from sklearn.tree import export_graphviz


def data_prep():
    # read the dataset
    df = pd.read_csv('D3.csv')
    
    #correct insurance to bool type by mapping
    insurance_map = {'no': False, 'yes': True}
    df['insurance'] = df['insurance'].map(insurance_map)

    #correct immigrant to bool type by mapping
    immigrant_map = {'native': False, 'immigrant': True}
    df['immigrant'] = df['immigrant'].map(immigrant_map)

    df['insurance'].fillna(df['insurance'].mode()[0], inplace=True)
    df['immigrant'].fillna(df['immigrant'].mode()[0], inplace=True)

    #correct the rest binary variable to bool type
    df['covid19_positive'] = df['covid19_positive'].astype(bool)
    df['covid19_symptoms'] = df['covid19_symptoms'].astype(bool)
    df['covid19_contact'] = df['covid19_contact'].astype(bool)
    df['asthma'] = df['asthma'].astype(bool)
    df['kidney_disease'] = df['kidney_disease'].astype(bool)
    df['liver_disease'] = df['liver_disease'].astype(bool)
    df['compromised_immune'] = df['compromised_immune'].astype(bool)
    df['heart_disease'] = df['heart_disease'].astype(bool)
    df['lung_disease'] = df['lung_disease'].astype(bool)
    df['diabetes'] = df['diabetes'].astype(bool)
    df['hiv_positive'] = df['hiv_positive'].astype(bool)
    df['hypertension'] = df['hypertension'].astype(bool)
    df['other_chronic'] = df['other_chronic'].astype(bool)
    df['nursing_home'] = df['nursing_home'].astype(bool)
    df['health_worker'] = df['health_worker'].astype(bool)


    df['income'].replace(['blank'], np.nan, inplace=True)
    df['race'].replace(['blank'], np.nan, inplace=True)
    df['income'].fillna(df['income'].mode()[0], inplace=True)
    df['race'].fillna(df['race'].mode()[0], inplace=True)

    df = df.drop(['region'], axis=1)
    
    # one-hot encoding
    df = pd.get_dummies(df)
    
    y = df['covid19_positive']
    X = df.drop(['covid19_positive'], axis=1)

    # setting random state
    rs = 10

    X_mat = X.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.3, stratify=y, random_state=rs)

    return df,X,y,X_train, X_test, y_train, y_test

def analyse_feature_importance(dm_model, feature_names, n_to_display=20):
    # grab feature importances from the model
    importances = dm_model.feature_importances_
    
    # sort them out in descending order
    indices = np.argsort(importances)
    indices = np.flip(indices, axis=0)

    # limit to 20 features, you can leave this out to print out everything
    indices = indices[:n_to_display]

    for i in indices:
        print(feature_names[i], ':', importances[i])

def visualize_decision_tree(dm_model, feature_names, save_name):
    dotfile = StringIO()
    export_graphviz(dm_model, out_file=dotfile, feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dotfile.getvalue())
    graph.write_png(save_name) # saved in the following file
