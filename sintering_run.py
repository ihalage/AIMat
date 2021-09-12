import os
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import TransformerMixin #gives fit_transform method for free

DATASET_PATH = "/home/avin/QM/material_data/CPM/data"

def load_material_data(dataset_path=DATASET_PATH):

	csv_path = os.path.join(dataset_path, "4000_dielectric_materials.csv")
	return pd.read_csv(csv_path)


material_data = load_material_data()

material_data = material_data.apply(pd.to_numeric, errors='coerce')
material_data = material_data.dropna(subset=['Sintering', 'er', 'Qf', 'Tcf'])
material_data = material_data[['Sintering', 'er', 'Qf', 'Tcf']]
# print material_data.info()

train_set, test_set = train_test_split(material_data, test_size=0.2, random_state=42)

train_data = train_set.drop('Sintering', axis=1)
train_labels = train_set['Sintering'].copy()
test_data = test_set.drop('Sintering', axis=1)
test_labels = test_set['Sintering'].copy()



#To handle fit_transform() problem
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)



class DataFrameSelector(BaseEstimator, TransformerMixin):
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names
	def fit(self, X, y=None):
		return self
	def transform(self, X):
		return X[self.attribute_names].values

attribs = ['er', 'Qf', 'Tcf']
pipeline = Pipeline([
						# ('selector', DataFrameSelector(attribs)),
						('std_scaler', StandardScaler())
						])

pipeline.fit_transform(train_data)

#trained model
pkl_filename = "saved_model_sintering/pickle_model.pkl"

# Load from file
with open(pkl_filename, 'r') as file:  
    sintering_model = pickle.load(file)

def predict_sintering(input_df):
	
	sintering_prediction = sintering_model.predict(pipeline.transform(input_df))
	return sintering_prediction[0]

# predict_sintering(np.array([65, 20000, 35]).reshape(1,-1))
# predict_sintering(np.array([45, 20000, 30]).reshape(1,-1))