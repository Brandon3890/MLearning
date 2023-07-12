from flask import Flask, request, jsonify

app = Flask(__name__)

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# Cargar los datos desde un archivo CSV
datos = pd.read_csv('Anexo_ET_demo_round_traces_1.csv', sep=';')

# Crear una instancia del codificador de etiquetas
label_encoder = LabelEncoder()

# Codificar las características categóricas en variables numéricas
datos['Team_codificado'] = label_encoder.fit_transform(datos['Team'])

# Seleccionar las columnas de variables independientes
X = datos[['Team_codificado', 'RoundHeadshots', 'RoundStartingEquipmentValue']]

# Seleccionar la columna de la variable objetivo
y = datos['RoundWinner_codificado']

# Crear el modelo de Regresión Logística
modelo = LogisticRegression(C=3, solver='liblinear')

# Entrenar el modelo con todos los datos
modelo.fit(X, y)

@app.route('/api', methods=['POST'])
def predict():
    # Obtener los datos de la solicitud POST
    data = request.json
    
    # Realizar la predicción utilizando tu modelo
    prediction = modelo.predict([data['input']])
    
    # Decodificar la predicción numérica a los valores originales
    prediction_decoded = label_encoder.inverse_transform(prediction)
    
    # Devolver la predicción como respuesta en formato JSON
    return jsonify({'prediction': prediction_decoded[0]})

if __name__ == '__main__':
    app.run()