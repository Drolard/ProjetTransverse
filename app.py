# -*- coding: utf-8 -*-
import time

import figures as figs

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
from dash.dependencies import Input, Output, State
import dash_reusable_components as drc

import pandas as pd

import numpy as np

import itertools

import sklearn
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm, model_selection
from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import datasets
from sklearn import metrics


import io

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objs as go

import sys

import colorlover as cl



#On charge toutvia un csv donné et on change certaines données
dataset = pd.read_csv("student-mat.csv", sep=";")

#On transforme tout les yes/no en boolean
dataset['schoolsup'] = np.where(dataset['schoolsup'] == 'yes', True, False)
dataset['famsup'] = np.where(dataset['schoolsup'] == 'yes', True, False)
dataset['paid'] = np.where(dataset['schoolsup'] == 'yes', True, False)
dataset['activities'] = np.where(dataset['schoolsup'] == 'yes', True, False)
dataset['nursery'] = np.where(dataset['schoolsup'] == 'yes', True, False)
dataset['higher'] = np.where(dataset['schoolsup'] == 'yes', True, False)
dataset['internet'] = np.where(dataset['schoolsup'] == 'yes', True, False)
dataset['romantic'] = np.where(dataset['schoolsup'] == 'yes', True, False)

#On crée une colonne avec la moyenne de l'élève et une colonne target
dataset['moyenne_G1G2G3'] = (dataset.G1+dataset.G2+dataset.G3)/3

dataset['target'] = ((dataset.G1 + dataset.G2 + dataset.G3)/3) >= 10

#Infos
infos = dict()

infos['shape'] = dataset.shape

buffer = io.StringIO()
dataset.info(buf=buffer)
infos['info'] = buffer.getvalue()

infos['isnull'] = dataset.isnull().sum()

infos['describe'] = dataset.describe()
infos['describe_dict'] = infos['describe'].to_dict('records')
infos['describe_dict'][0]['info'] = 'count'
infos['describe_dict'][1]['info'] = 'mean'
infos['describe_dict'][2]['info'] = 'std'
infos['describe_dict'][3]['info'] = 'min'
infos['describe_dict'][4]['info'] = '25%'
infos['describe_dict'][5]['info'] = '50%'
infos['describe_dict'][6]['info'] = '75%'
infos['describe_dict'][7]['info'] = 'max'
infos['describe_columns'] = [{"name": i, "id": i} for i in infos['describe']]
infos['describe_columns'].insert(0, {'name': 'Infos', 'id': 'info'}) 

infos['describe_object'] = dataset.describe(exclude=np.number)
infos['describe_object_dict'] = infos['describe_object'].to_dict('records')
infos['describe_object_dict'][0]['info'] = 'count'
infos['describe_object_dict'][1]['info'] = 'unique'
infos['describe_object_dict'][2]['info'] = 'top'
infos['describe_object_dict'][3]['info'] = 'freq'
infos['describe_object_columns'] = [{"name": i, "id": i} for i in infos['describe_object']]
infos['describe_object_columns'].insert(0, {'name': 'Infos', 'id': 'info'}) 

infos['describe_only_target'] = dataset.target.describe(exclude=np.number)
infos['describe_only_target_dict'] = [infos['describe_only_target'].to_dict()]
infos['describe_only_target_columns'] = [{"name": i, "id": i} for i in infos['describe_only_target'].keys()]
#Box plots 

box_plot = dict()

box_plot['G1'] = px.box(dataset, y="G1")
box_plot['G2'] = px.box(dataset, y="G2")
box_plot['G3'] = px.box(dataset, y="G3")

box_plot['moyenne_G1G2G3'] = px.box(dataset, y="moyenne_G1G2G3")

#FEATURE ENGEERING 

# on supprime les colonnes en trop
cleanDataset = dataset
cleanDataset = cleanDataset.drop(columns=['Mjob', 'Fjob', 'reason', 'guardian', 'famsup', 'address', 'Pstatus'])

#On supprime les colonnes qui ont la même valeur dans toutes les row
nunique = cleanDataset.apply(pd.Series.nunique)
cols_to_drop = nunique[nunique == 1].index
cleanDataset = cleanDataset.drop(cols_to_drop, axis=1)


#On change les colonnes qui ont 2 valeurs en 0 et 1

cleanDataset['sex'] = cleanDataset.apply(lambda row: 0 if row['sex'] == 'F' else 1, axis=1)
cleanDataset['famsize'] = cleanDataset.apply(lambda row: 0 if row['famsize'] == 'LE3' else 1, axis=1)
cleanDataset['school'] = cleanDataset.apply(lambda row: 0 if row['school'] =='GP' else 1, axis=1)

#On change tout les booleens en 0 et 1
cleanDataset.replace(False, 0, inplace=True)
cleanDataset.replace(True, 1, inplace=True)

infos['clean_dataset'] = cleanDataset

#On affiche

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 'https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
             
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Téléversement et analyse descriptive du dataset sélectionné ', children=[
            dbc.Container([
                dbc.Row([
                    html.H2(children='Informations sur le dataset :'),
                    
                    html.Ul(children=[
                        
                        html.Li(children='Dataset utilisé = Student-Performane disponible ici : http://archive.ics.uci.edu/ml/datasets/Student+Performance'),
                        
                        html.Li(children='Shape du dataset = ' + str(infos['shape'])),
                        
                        html.Li(children='Infos du dataset = ' + str(infos['info'])),
                        
                        html.Li(children=[
                            html.Span(children='Intitulés des colonnes = '),
                            dash_table.DataTable(
                                id='table_intitule',
                                columns=[{"name": i, "id": i} for i in dataset.columns],
                            )
                        ]),
                        
                        html.Li(children=
                                '''Informations sur les colonnes (en anglais car récupérer de http://archive.ics.uci.edu/ml/datasets/Student+Performance  = \n
                                1 school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira) \n
                                2 sex - student's sex (binary: 'F' - female or 'M' - male) \n
                                3 age - student's age (numeric: from 15 to 22) \n
                                4 address - student's home address type (binary: 'U' - urban or 'R' - rural) \n
                                5 famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3) \n
                                6 Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart) \n
                                7 Medu - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education) \n
                                8 Fedu - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education) \n
                                9 Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other') \n
                                10 Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other') \n
                                11 reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other') \n
                                12 guardian - student's guardian (nominal: 'mother', 'father' or 'other') \n
                                13 traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour) \n
                                14 studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours) \n
                                15 failures - number of past class failures (numeric: n if 1<=n<3, else 4) \n
                                16 schoolsup - extra educational support (binary: yes or no) \n
                                17 famsup - family educational support (binary: yes or no) \n
                                18 paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no) \n
                                19 activities - extra-curricular activities (binary: yes or no) \n
                                20 nursery - attended nursery school (binary: yes or no) \n
                                21 higher - wants to take higher education (binary: yes or no) \n
                                22 internet - Internet access at home (binary: yes or no) \n
                                23 romantic - with a romantic relationship (binary: yes or no) \n
                                24 famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent) \n
                                25 freetime - free time after school (numeric: from 1 - very low to 5 - very high) \n
                                26 goout - going out with friends (numeric: from 1 - very low to 5 - very high) \n
                                27 Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high) \n
                                28 Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high) \n
                                29 health - current health status (numeric: from 1 - very bad to 5 - very good) \n
                                30 absences - number of school absences (numeric: from 0 to 93) \n'''),
                                
                        html.Li(children='Nombre de valeurs nulles par colonnes = ' + str(infos['isnull'])),
                        
                        html.Li(children=[
                            html.Span(children='Moyenne, max, min, et percentiles pour les colonnes numériques = '),
                            dash_table.DataTable(
                                id='table_describe',
                                columns=infos['describe_columns'],
                                data=infos['describe_dict'],
                            )
                        ]),
                        
                        html.Li(children=[
                            html.Span(children='Count, unique, top et freq pour les colonnes non-numériques = '),
                            dash_table.DataTable(
                                id='table_describe_object',
                                columns=infos['describe_object_columns'],
                                data=infos['describe_object_dict'],
                            )
                        ])
                    ]),
                ]),
               
                html.Hr(),
                
                dbc.Row([
                
                    dbc.Col(
                    html.H2(children='Box plots des différentes valeurs :'),
                    width={"size": 12},
                    ),
                    
                    dbc.Col([
                        html.H4(children='Box plots des moyennes de math au premier trimestre :'),
                        dcc.Graph(figure=box_plot['G1']),
                    ]),
                    
                    dbc.Col([
                        html.H4(children='Box plots des moyennes de math au deuxième trimestre :'),
                        dcc.Graph(figure=box_plot['G2']),
                    ]),
                    
                    dbc.Col([
                        html.H4(children='Box plots des moyennes de math au troisième trimestre :'),
                        dcc.Graph(figure=box_plot['G3']),
                    ]),
    
                ]),
                
                dbc.Row([
                
                    dbc.Col([
                        html.H2(children='Box plots de la moyenne et étude du target :'),
                        html.Div(children='La valeur target représente si l\'élève a réussi son année, c\'est à dire qu\'il a eu plus de 10 de moyenne aux trois semestres')
                    ],
                    width={"size": 12},
                    ),
                    
                    dbc.Col([
                        html.H4(children='Box plots de la moyenne des éleves sur les 3 semestres:'),
                        dcc.Graph(figure=box_plot['moyenne_G1G2G3']),
                    ]),
                    
                    dbc.Col([
                        html.Div(children='Count, unique, top et freq pour la colonne target, c\'est à dire le nombre d\'élèves, combien il y a de valeurs unique (2 car vrai ou faux), quelle est la plus fréquente (false = plus d\'élèves ont pas leur année, true = plus d\'élèves ont leur année), et la fréquence de cette valeur la plus fréquente '),
                        dash_table.DataTable(
                            id='table_describe_only_target',
                            columns=infos['describe_only_target_columns'],
                            data=infos['describe_only_target_dict'],
                        ),
                    ]),
    
                ]),
                
                dbc.Row([
                
                    dbc.Col([
                        html.H2(children='Table après feature engeneering (10 premières lignes) :'),
                        html.Div(children='Les colonnes non numériques ont été enlevés, les colonnes ayant qu\'une seule valeur ont été enlevé et les colonnes booléens ont été transformé en numérique')
                    ],
                    width={"size": 12},
                    ),
                    
                    dbc.Col([
                        dash_table.DataTable(
                            id='table_clean_dataset',
                            columns=[{"name": i, "id": i} for i in infos['clean_dataset'].columns],
                            data=infos['clean_dataset'].head(10).to_dict('records'),
                        )
                    ]),
    
                ]),
                
            ],fluid=True),
            
            
        ]),

        dcc.Tab(label='Analyse du comportement et des performances du modèle', children=[
            html.Div(
                id="app-container",
                children=[
                    dbc.Row([
                        dbc.Col([
                            html.H1(children='Analyse du dataset via SVC (Support Vector Classification) :'),
                            html.Br(),
                            html.Br(),
                            
                            html.Div(
                                id="left-column",
                                children=[
                                    drc.Card(
                                        id="first-card",
                                        children=[
                                                
                                            html.H2(children='Parametre généraux du modele SVC :'),
                                            
                                            drc.NamedDropdown(
                                                name="Kernel",
                                                id="dropdown-svm-parameter-kernel",
                                                options=[
                                                    {
                                                        "label": "Radial basis function (RBF)",
                                                        "value": "rbf",
                                                    },
                                                    {
                                                        "label": "Linéaire", 
                                                        "value": "linear"
                                                    },
                                                    {
                                                        "label": "Polynomial",
                                                        "value": "poly",
                                                    },
                                                    {
                                                        "label": "Sigmoid",
                                                        "value": "sigmoid",
                                                    },
                                                ],
                                                value="rbf",
                                                clearable=False,
                                                searchable=False,
                                            ),
                                            
                                            drc.NamedSlider(
                                                name="Pourcentage de donnée utilisée pour le train (les autres données seront utilisés pour le test)",
                                                id="slider-dataset-sample-size",
                                                min=10,
                                                max=90,
                                                marks={
                                                    10: {'label': '10%'},
                                                    20: {'label': '20%'},
                                                    30: {'label': '30%'},
                                                    40: {'label': '40%'},
                                                    50: {'label': '50%'},
                                                    60: {'label': '60%'},
                                                    70: {'label': '70%'},
                                                    80: {'label': '80%'},
                                                    90: {'label': '90%'},
                                                },
                                                value=40,
                                            ),
                                            
                                            drc.NamedSlider(
                                                name="Parametre de régularisation (Parametre 'C' du modele SVC, voir https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)",
                                                id="slider-svm-parameter-C",
                                                min=-2,
                                                max=4,
                                                value=0,
                                                marks={
                                                    i: "{}".format(10 ** i)
                                                    for i in range(-2, 5)
                                                },
                                            ),
                                            html.Div(
                                                id="shrinking-container",
                                                children=[
                                                    html.P(children="Shrinking"),
                                                    dcc.RadioItems(
                                                        id="radio-svm-parameter-shrinking",
                                                        labelStyle={
                                                            "margin-right": "7px",
                                                            "display": "inline-block",
                                                        },
                                                        options=[
                                                            {
                                                                "label": " Activé",
                                                                "value": "True",
                                                            },
                                                            {
                                                                "label": " Désactivé",
                                                                "value": "False",
                                                            },
                                                        ],
                                                        value="True",
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Br(),
                                    html.Br(),
                                    drc.Card(
                                        id="button-card",
                                        children=[
                                            
                                            html.H2(children='Parametre du modele si kernel "rbf", "poly" ou "sigmoid" :'),
                                            
                                            drc.NamedSlider(
                                                name="Gamma",
                                                id="slider-svm-parameter-gamma",
                                                min=-4,
                                                max=0,
                                                value=-1,
                                                marks={
                                                    i: "{}".format(10 ** i)
                                                   for i in range(-5, 1)
                                                },
                                            ),

                                        ],
                                    ),
                                    html.Br(),
                                    html.Br(),
                                    drc.Card(
                                        id="last-card",
                                        children=[
                                            
                                            html.H2(children='Parametre du modele si kernel "poly" :'),
                                            
                                            drc.NamedSlider(
                                                name="Degree",
                                                id="slider-svm-parameter-degree",
                                                min=2,
                                                max=10,
                                                value=3,
                                                step=1,
                                                marks={
                                                    str(i): str(i) for i in range(2, 11, 2)
                                                },
                                            ),
                                            
                                        ],
                                    ),
                                ],
                            ),
                        ]),
                        dbc.Col([
                            html.Div(
                                id="div-graphs",
                                children=dcc.Graph(
                                    id="graph-sklearn-svm",
                                    figure=dict(
                                        layout=dict(
                                            plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                                        )
                                    ),
                                ),
                            ),
                        ]),
                        
                        dbc.Col([
                            html.Div(
                                id="div-graphs-random-forest",
                                children=dcc.Graph(
                                    id="graph-sklearn-random-forest",
                                    figure=dict(
                                        layout=dict(
                                            plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                                        )
                                    ),
                                ),
                            ),
                        ]),
                        
                        dbc.Col([
                            html.H1(children='Analyse du dataset via RandomForest :'),
                            html.Br(),
                            html.Br(),
                            
                            
                            html.Div(
                                id="left-column-random-forest",
                                children=[
                                    drc.Card(
                                        id="first-card-random-forest",
                                        children=[
                                                
                                            html.H2(children='Parametre généraux du modele :'),
                                            
                                            drc.NamedSlider(
                                                name="Pourcentage de donnée utilisée pour le train (les autres données seront utilisés pour le test)",
                                                id="slider-dataset-sample-size-random-forest",
                                                min=10,
                                                max=90,
                                                marks={
                                                    10: {'label': '10%'},
                                                    20: {'label': '20%'},
                                                    30: {'label': '30%'},
                                                    40: {'label': '40%'},
                                                    50: {'label': '50%'},
                                                    60: {'label': '60%'},
                                                    70: {'label': '70%'},
                                                    80: {'label': '80%'},
                                                    90: {'label': '90%'},
                                                },
                                                value=40,
                                            ),
                                            
                                            
                                            drc.NamedSlider(
                                                name="Nombre d'arbres dans la forêt : ",
                                                id="slider-random-forest-n-estimators",
                                                min=10,
                                                max=1000,
                                                marks={
                                                    str(i): str(i)
                                                    for i in [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
                                                },
                                                value=100,
                                            ),
                                            
                                            drc.NamedSlider(
                                                name="Profondeur max d'un arbre (Si à 'None' les arbres ne s'arrêteront que quand toutes les feuilles sont pures ou qu'elles contiennent moins que 'Minimum de sample requis pour split', plus bas): ",
                                                id="slider-random-forest-max-depth",
                                                min=0,
                                                max=100,
                                                step=1,
                                                marks={
                                                    0: {'label': 'None'},
                                                    10: {'label': '10'},
                                                    20: {'label': '20'},
                                                    30: {'label': '30'},
                                                    40: {'label': '40'},
                                                    50: {'label': '50'},
                                                    60: {'label': '60'},
                                                    70: {'label': '70'},
                                                    80: {'label': '80'},
                                                    90: {'label': '90'},
                                                    100: {'label': '100'},
                                                },
                                                value=0,
                                            ),
                                            
                                            drc.NamedSlider(
                                                name="Minimum de sample requis pour split :",
                                                id="slider-random-forest-min-samples-split",
                                                min=2,
                                                max=50,
                                                step = 1,
                                                marks={
                                                    str(i): str(i)
                                                    for i in [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
                                                },
                                                value=2,
                                            ),
                                            
                                            drc.NamedSlider(
                                                name="Minimum de sample requis pour créer une feuille' :",
                                                id="slider-random-forest-min-samples-leaf",
                                                min=1,
                                                max=50,
                                                step = 1,
                                                marks={
                                                    str(i): str(i)
                                                    for i in [1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
                                                },
                                                value=1,
                                            ),
                                            
                                            drc.NamedSlider(
                                                name="Pourcentage minimum du total de sample requis pour créer une feuille :",
                                                id="slider-random-forest-min-weight-fraction-leaf",
                                                min=0,
                                                max=20,
                                                step=1,
                                                marks={
                                                    0: {'label': '0%'},
                                                    1: {'label': '1%'},
                                                    2: {'label': '2%'},
                                                    4: {'label': '4%'},
                                                    6: {'label': '6%'},
                                                    8: {'label': '8%'},
                                                    10: {'label': '10%'},
                                                    12: {'label': '12%'},
                                                    14: {'label': '14%'},
                                                    16: {'label': '16%'},
                                                    18: {'label': '18%'},
                                                    20: {'label': '20%'},
                                                },
                                                value=0,
                                            ),
                                            
                                            drc.NamedSlider(
                                                name="Nombre de features maximale à considérer pour un split('auto' pour que min_weight_fraction_leaf soit à auto) :",
                                                id="slider-random-forest-max-feature",
                                                min=0,
                                                max=20,
                                                step=1,
                                                marks={
                                                    0: {'label': 'auto'},
                                                    1: {'label': '1'},
                                                    2: {'label': '2'},
                                                    4: {'label': '4'},
                                                    6: {'label': '6'},
                                                    8: {'label': '8'},
                                                    10: {'label': '10'},
                                                    12: {'label': '12'},
                                                    14: {'label': '14'},
                                                    16: {'label': '16'},
                                                    18: {'label': '18'},
                                                    20: {'label': '20'},
                                                },
                                                value=0,
                                            ),

                                        ],
                                    ),
                                ],
                            ),
                        ]),
                        
                    ]),
                ],
            )
        ]),
        dcc.Tab(label='Dataset (attention, peut être très long)', children=[
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in dataset.columns],
                data=dataset.to_dict('records'),
            )
        ]),
    ])
])
              
                    
#########################################################################################################
#########################################################################################################
######################################        MODELE SVC           ######################################
#########################################################################################################
#########################################################################################################
                                
# Disable Sliders if kernel not in the given list
@app.callback(
    Output("slider-svm-parameter-degree", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_degree(kernel):
    return kernel != "poly"


@app.callback(
    Output("slider-svm-parameter-gamma", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_gamma_coef(kernel):
    return kernel not in ["rbf", "poly", "sigmoid"]

                       
@app.callback(
    Output("div-graphs", "children"),
    [
        Input("dropdown-svm-parameter-kernel", "value"),
        Input("slider-svm-parameter-degree", "value"),
        Input("slider-svm-parameter-C", "value"),
        Input("slider-svm-parameter-gamma", "value"),
        Input("radio-svm-parameter-shrinking", "value"),
        Input("slider-dataset-sample-size", "value"),
    ],
)
def update_svm_graph(
    kernel,
    degree,
    C,
    gamma,
    shrinking,
    sample_size,
):
    
    # Data Pre-processing
    clean_dataset = infos['clean_dataset']
    clean_target = clean_dataset['target']
    clean_dataset = clean_dataset.drop('target', axis = 1)   
    
    X, y = clean_dataset, clean_target
  
    # On fait le train_test_split en fonction des sliders
    train_size = sample_size/100
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=42
    )

    if shrinking == "True":
        flag = True
    else:
        flag = False
        
    gamma = 10 ** gamma
    C = 10 ** C

    # Train SVM
    modele = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, shrinking=flag)
    modele.fit(X_train, y_train)
    
    #On récupère le y_pred
    y_pred = modele.predict(X_test)
    
    
    #Courbe ROC
    roc_figure = figs.serve_roc_curve(model=modele, X_test=X_test, y_test=y_test)

    #On génère un camembert pour la matrice de confusion
    matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)
    tn, fp, fn, tp = matrix.ravel()

    values = [tp, fn, fp, tn]
    label_text = ["True Positive", "False Negative", "False Positive", "True Negative"]
    labels = ["TP", "FN", "FP", "TN"]
    blue = cl.flipper()["seq"]["9"]["Blues"]
    red = cl.flipper()["seq"]["9"]["Reds"]
    colors = ["#13c6e9", blue[1], "#ff916d", "#ff744c"]

    trace0 = go.Pie(
        labels=label_text,
        values=values,
        hoverinfo="label+value+percent",
        textinfo="text+value",
        text=labels,
        sort=False,
        marker=dict(colors=colors),
        insidetextfont={"color": "white"},
        rotation=90,
    )

    layout = go.Layout(
        title="Confusion Matrix",
        margin=dict(l=50, r=50, t=100, b=10),
        legend=dict(bgcolor="#282b38", font={"color": "#a5b1cd"}, orientation="h"),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )

    data = [trace0]
    figure = go.Figure(data=data, layout=layout)

    confusion_figure = figure


    return [
        html.Div(
            id="graphs-container",
            children=[
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(id="graph-line-roc-curve", figure=roc_figure),
                ),
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(
                        id="graph-pie-confusion-matrix", figure=confusion_figure
                    ),
                ),
            ],
        ),
    ]


#########################################################################################################
#########################################################################################################
######################################        RANDOM FOREST           ###################################
#########################################################################################################
#########################################################################################################

# Disable Sliders if kernel not in the given list                
@app.callback(
    Output("div-graphs-random-forest", "children"),
    [     
        Input("slider-random-forest-n-estimators", "value"),
        Input("slider-random-forest-max-depth", "value"),
        Input("slider-random-forest-min-samples-split", "value"),
        Input("slider-random-forest-min-samples-leaf", "value"),
        Input("slider-random-forest-min-weight-fraction-leaf", "value"),
        Input("slider-random-forest-max-feature", "value"),
        Input("slider-dataset-sample-size-random-forest", "value"),
    ],
)
def update_svm_graph(
    n_estimators,
    max_depth,
    min_samples_split,
    min_samples_leaf,
    min_weight_fraction_leaf,
    max_features,
    sample_size
):
    
    # Data Pre-processing
    clean_dataset = infos['clean_dataset']
    clean_target = clean_dataset['target']
    clean_dataset = clean_dataset.drop('target', axis = 1)   
    
    X, y = clean_dataset, clean_target
  
    # On fait le train_test_split en fonction des sliders
    train_size = sample_size/100
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=42
    )
    
    if max_depth == 0:
        max_depth = None
        
    min_weight_fraction_leaf = min_weight_fraction_leaf/100
        
    if max_features == 0:
        max_features = 'auto'

    # Train SVM
    modele = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features)
    modele.fit(X_train, y_train)
    
    #On récupère le y_pred
    y_pred = modele.predict(X_test)
    
    
    #Courbe ROC
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)

    # AUC Score
    auc_score = metrics.roc_auc_score(y_true=y_test, y_score=y_pred)

    trace0 = go.Scatter(
        x=fpr, y=tpr, mode="lines", name="Test Data", marker={"color": "#13c6e9"}
    )

    layout = go.Layout(
        title=f"ROC Curve (AUC = {auc_score:.3f})",
        xaxis=dict(title="False Positive Rate", gridcolor="#2f3445"),
        yaxis=dict(title="True Positive Rate", gridcolor="#2f3445"),
        legend=dict(x=0, y=1.05, orientation="h"),
        margin=dict(l=100, r=10, t=25, b=40),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )

    data = [trace0]
    
    figure = go.Figure(data=data, layout=layout)
     
    roc_figure = figure

    #On génère un camembert pour la matrice de confusion
    matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)
    tn, fp, fn, tp = matrix.ravel()

    values = [tp, fn, fp, tn]
    label_text = ["True Positive", "False Negative", "False Positive", "True Negative"]
    labels = ["TP", "FN", "FP", "TN"]
    blue = cl.flipper()["seq"]["9"]["Blues"]
    red = cl.flipper()["seq"]["9"]["Reds"]
    colors = ["#13c6e9", blue[1], "#ff916d", "#ff744c"]

    trace0 = go.Pie(
        labels=label_text,
        values=values,
        hoverinfo="label+value+percent",
        textinfo="text+value",
        text=labels,
        sort=False,
        marker=dict(colors=colors),
        insidetextfont={"color": "white"},
        rotation=90,
    )

    layout = go.Layout(
        title="Confusion Matrix",
        margin=dict(l=50, r=50, t=100, b=10),
        legend=dict(bgcolor="#282b38", font={"color": "#a5b1cd"}, orientation="h"),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )

    data = [trace0]
    figure = go.Figure(data=data, layout=layout)

    confusion_figure = figure


    return [
        html.Div(
            id="graphs-container-random-forest",
            children=[
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(id="graph-line-roc-curve-random-forest", figure=roc_figure),
                ),
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(
                        id="graph-pie-confusion-matrix-random-forest", figure=confusion_figure
                    ),
                ),
            ],
        ),
    ]

if __name__ == '__main__':
    app.run_server(debug=False)
    
    
