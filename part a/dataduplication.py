import pandas as pd
from sklearn.model_selection import train_test_split

# Charger le fichier CSV
df1 = pd.read_csv('StepOne_LabelTrain.csv')
print("number of line in StepOne_LabelTrain :" ,len(df1))
# Conserver les lignes qui n'ont pas toutes les colonnes de l'étiquette à 0 dans df_labels_train
df2= df1.loc[~(df1[['Washing Machine', 'Dishwasher', 'Tumble Dryer', 'Microwave', 'Kettle']] == 0).all(axis=1)]
print("number of line in StepOne_LabelTrain with only 0 in each column :", len(df2))


# Fusionner df1 et df2_20_percent
df1 = pd.concat([df1, df2], ignore_index=True)
df1 = pd.concat([df1, df2], ignore_index=True)
df1 = pd.concat([df1, df2], ignore_index=True)




df1['Index'] = range(1, len(df1) + 1)

# Enregistrer le dataframe combiné en tant que fichier CSV
df1.to_csv('df_StepOne.csv', index=False)


####################input train


# Charger le fichier CSV inputtrain
inputtrain = pd.read_csv('InputTrain.csv')

# Conserver les lignes de inputtrain qui ont les mêmes index que df1
df3 = inputtrain.loc[inputtrain['Index'].isin(df2['Index'])]
# Fusionner 
df4 = pd.concat([inputtrain, df3], ignore_index=True)
df4 = pd.concat([df4, df3], ignore_index=True)
df4 = pd.concat([df4, df3], ignore_index=True)

df4['Index'] = range(1, len(df4) + 1)
# Enregistrer le dataframe modifié en tant que CSV
df4.to_csv('df_train.csv', index=False)

print("number of line in the new changed dataset StepOne_LabelTrain.csv:", len(df1))
print("number of line in the new changed dataset InputTrain.csv :",len(df4))




