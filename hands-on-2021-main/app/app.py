import base64
import io
import yaml

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go

import numpy as np
import tensorflow as tf
from PIL import Image

from constants import CLASSES

with open('app.yaml') as yaml_data:
    
    params = yaml.safe_load(yaml_data)
    

IMAGE_WIDTH = params[1]['IMAGE_WIDTH']
IMAGE_HEIGHT = params[2]['IMAGE_HEIGHT']

# Load DNN model
classifier = tf.keras.models.load_model(params[0]['keras_path'])

def classify_image(path, model, image_box=None):
  """Classify image by model

  Parameters
  ----------
  path: filepath to image
  model: tf/keras classifier

  Returns
  -------
  class id with highest probability is returned
  """
  images_list = []
  image = path.resize((IMAGE_WIDTH, IMAGE_HEIGHT), box=image_box) # box argument clips image to (x1, y1, x2, y2)
  image = np.array(image)
  images_list.append(image)
  
  return [np.argmax(model.predict(np.array(images_list))), model.predict(np.array(images_list))]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash('Traffic Signs Recognition', external_stylesheets=external_stylesheets)



#styles dictionaries
layout = dict(
    margin=dict(l=40, r=40, b=180, t=40),
    hovermode="closest",
    legend=dict(font=dict(color='#7f7f7f'), orientation="h"),
    title= 'Probabilités estimées',
    font=dict(
        color="#7f7f7f"
    ),
)

title_style = dict(
    textAlign='center',
    backgroundColor='#0665A4',
    color='#EAF1E8')


# Define application layout
app.layout = html.Div(style = dict(textAlign='center'), children=[
    html.H1('Sign Classifier', 
           style = title_style),
    html.Hr(),
    html.H3('Reconnaissance de panneaux de circulation grâce à un réseau de neurones entrainé'),
    html.H6("Un modèle de réseau de neurones artificiels a été entrainé sur une base de données de panneaux de circulation existante, il permet d'afficher la probabilité d'appartenir à une catégorie de panneau."),
    html.Hr(),
    html.H6("Téléversez une image de panneau à reconnaître"),
    dcc.Upload(
        id='bouton-chargement',
        children=html.Div([
            'Cliquer-déposer ou ',
                    html.A('sélectionner une image')
        ]),
        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'backgroundColor': '#E7F2FA',
            'display': 'inline-block',
            'margin': '10px'
        }
    ),
    html.Div(id='mon-image'),
                      ])

@app.callback(Output('mon-image', 'children'),
              [Input('bouton-chargement', 'contents')])
def update_output(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        if 'image' in content_type:
            image = Image.open(io.BytesIO(base64.b64decode(content_string)))
            predicted_class = classify_image(image, classifier)[0]
            estimated_probabilities = classify_image(image, classifier)[1][0]
            max_proba = max(estimated_probabilities)
            estimated_probabilities, classes_list = (list(t) for t in zip(*sorted(zip(estimated_probabilities, CLASSES.values()), reverse=True)))
            return html.Div(style = dict(textAlign='center'), children = [
                html.Hr(),
                html.Img(src=contents, style={'height':'25%', 'width':'25%', 'margin':'20px'}),
                html.H3('Classe prédite : {}, avec une probabilité de : {:.5f}'.format(CLASSES[predicted_class], max_proba)),
                html.Hr(),
                html.Div([
                    dcc.Graph(id='horizon_proba', 
                             figure={
                                 'data': [go.Bar(x=classes_list,
                                                 y=estimated_probabilities)],
                                 "layout": layout
                             })
                ]),
                html.Hr(),
                #html.Div('Raw Content'),
                #html.Pre(contents, style=pre_style)
            ])
        else:
            try:
                # Décodage de l'image transmise en base 64 (cas des fichiers ppm)
                # fichier base 64 --> image PIL
                image = Image.open(io.BytesIO(base64.b64decode(content_string)))
                # image PIL --> conversion PNG --> buffer mémoire 
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                # buffer mémoire --> image base 64
                buffer.seek(0)
                img_bytes = buffer.read()
                content_string = base64.b64encode(img_bytes).decode('ascii')
                # Appel du modèle de classification
                predicted_class = classify_image(image, classifier)[0]
                estimated_probabilities = classify_image(image, classifier)[1][0]
                max_proba = max(estimated_probabilities)
                estimated_probabilities, classes_list = (list(t) for t in zip(*sorted(zip(estimated_probabilities, CLASSES.values()), reverse=True)))
                # Affichage de l'image
                return html.Div([
                    html.Hr(),
                    html.Img(src='data:image/png;base64,' + content_string, style={'height':'25%', 'width':'25%'}),
                    html.H3('Classe prédite : {}, avec une probabilité de : {:.5f}'.format(CLASSES[predicted_class], max_proba)),
                    html.Hr(),
                    html.Div([
                        dcc.Graph(id='horizon_proba', 
                                 figure={
                                     'data': [go.Bar(x=classes_list,
                                                     y=estimated_probabilities)],
                                     "layout": layout
                                 })
                    ]),
                    html.Hr(),
                ])
            except:
                return html.Div([
                    html.Hr(),
                    html.Div('Uniquement des images svp : {}'.format(content_type)),
                    html.Hr(),                
                    html.Div('Raw Content'),
                    html.Pre(contents, style=pre_style)
                ])
            


# Start the application
if __name__ == '__main__':
    app.run_server(debug=True)