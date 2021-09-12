
import os
import pandas as pd 
import matplotlib.pyplot as plt 
from six.moves import urllib
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import pickle
np.set_printoptions(threshold=np.nan)


DATASET_PATH = "data/"

def load_material_data(dataset_path=DATASET_PATH):

	csv_path = os.path.join(dataset_path, "4000_dielectric_materials.csv")
	return pd.read_csv(csv_path)


material_data = load_material_data()

material_data = material_data.apply(pd.to_numeric, errors='coerce')
material_data = material_data.dropna(subset=['Sintering', 'er', 'Qf', 'Tcf'])
material_data = material_data[['Sintering', 'er', 'Qf', 'Tcf']]
print material_data.info()

train_set, test_set = train_test_split(material_data, test_size=0.2, random_state=42)

train_data = train_set.drop('Sintering', axis=1)
train_labels = train_set['Sintering'].copy()
test_data = test_set.drop('Sintering', axis=1)
test_labels = test_set['Sintering'].copy()


# plt.plot(material_data['er'], material_data['Sintering'], 'ro')
# plt.show()


#Feature scaling and pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import TransformerMixin #gives fit_transform method for free

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

attribs = list(train_data)
print 'attribs', attribs
pipeline = Pipeline([
						# ('selector', DataFrameSelector(attribs)),
						('std_scaler', StandardScaler())
						])

prepared_train_data = pipeline.fit_transform(train_data)
prepared_test_data = pipeline.transform(test_data)
param_grid = [
{'n_estimators': [3, 10, 100], 'max_features': [3]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [3]},
]

rand_forest = RandomForestRegressor()

grid_search = GridSearchCV(rand_forest, param_grid, cv=5,
scoring='neg_mean_squared_error')


grid_search.fit(prepared_train_data, train_labels)


some_data = test_data
some_labels = test_labels
some_data_prepared = pipeline.transform(some_data)

predictions = grid_search.predict(some_data_prepared)

mse = mean_squared_error(some_labels, predictions)
rmse = np.sqrt(mse)

print predictions
print 'RMSE: ', rmse

# Save to file in the current working directory
pkl_filename = "saved_model_sintering/pickle_model.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(grid_search, file)


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print "Mean absolute percentage error:", mean_absolute_percentage_error(some_labels, predictions),"%"

print grid_search.predict(pipeline.transform(np.array([12, 90000, -35]).reshape(1,-1)))
print grid_search.predict(pipeline.transform(np.array([45, 12000, 15]).reshape(1,-1)))
print grid_search.predict(pipeline.transform(np.array([47, 14000, 19]).reshape(1,-1)))
print grid_search.predict(pipeline.transform(np.array([49, 20000, -5]).reshape(1,-1)))




# def sintering_dnn(in_data):

# 	dense1 = tf.layers.dense(in_data, 28, activation=tf.nn.relu)
# 	dense2 = tf.layers.dense(dense1, 28, activation=tf.nn.relu)
# 	dense3 = tf.layers.dense(dense2, 8, activation=tf.nn.relu)
# 	out = tf.layers.dense(dense3, 1, activation=tf.nn.relu)

# 	return out

# def train(learn_rate, batch_size):

# 	inputs = tf.placeholder(tf.float32, (None,3), name='inputs')
# 	labels = tf.placeholder(tf.float32, (None,1), name='labels')

# 	predictions = sintering_dnn(inputs)
# 	predictions = tf.identity(predictions, name='predictions')

# 	loss = tf.losses.mean_squared_error(predictions, labels)

# 	opt = tf.train.AdamOptimizer(learn_rate).minimize(loss)

# 	def do_report(e):

# 		r = sess.run([predictions, loss], feed_dict={inputs: prepared_test_data, labels: np.array(test_labels).reshape(-1,1)})

# 		# print np.c_[r[0],test_labels]
# 		print 'Loss at the moment: ', np.sqrt(r[1])



# 	init = tf.global_variables_initializer()
# 	saver = tf.train.Saver()

# 	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
# 	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		
# 		sess.run(init)

# 		try:
# 			epochs = 100000
# 			for e in range(epochs):
# 				print 'epoch :', e
# 				for i in range(len(train_data) / batch_size):
					
# 					index=i*batch_size
# 					batch_xs = prepared_train_data[index:index + batch_size]
# 					batch_ys = np.array(train_labels[index:index + batch_size]).reshape(-1,1)
# 					# print batch_ys.shape
				
# 					sess.run(opt, feed_dict={inputs: batch_xs, labels: batch_ys})

# 				do_report(e)


# 		except KeyboardInterrupt:
# 			print 'Training stopped manually'
# 			save_path = saver.save(sess, "/home/avin/QM/material_data/CPM/saved_model_sintering/cpm_sintering.ckpt")
#   	# 		print("Model saved in path: %s" % save_path)



# if __name__ == "__main__":

# 	train(learn_rate=0.001,
# 		  batch_size=10)
