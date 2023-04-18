import tensorflow as tf
import pandas as pd
from tensorflow.keras import regularizers
from tensorflow.keras import layers



# Charger les données d'entrée
input_train = pd.read_csv("InputTrain.csv")
input_test = pd.read_csv("InputTest.csv")
label_train = pd.read_csv("StepOne_LabelTrain.csv")
final_input = pd.read_csv("Sample_LabelTest.csv")
input_train = input_train.iloc[:, 2:]
label_train = label_train.iloc[:, 1:]
input_test = input_test.iloc[:, 2:]

# Créer le modèle
model = tf.keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(2158,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(5, activation='softmax')
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(input_train.values[:, 2:], label_train.values[:, 1:],
          epochs=220, batch_size=128)

# Prédire les sorties pour les données de test
predictions = model.predict(input_test.values[:, 2:])

# Enregistrer les prédictions dans un fichier CSV

final = pd.DataFrame(predictions.astype(int), columns=["Washing Machine", "Dishwasher", "Tumble Dryer", "Microwave", "Kettle"])
final.insert(0, "Index", final_input["Index"])
final.to_csv("./test_kaggle.csv", index=False)