import os
import pandas as pd 
import matplotlib.pyplot as plt 
from six.moves import urllib
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle

import tensorflow as tf
import tensorflow.contrib.layers as lays

# tf.logging.set_verbosity(tf.logging.INFO)

DATASET_PATH = "/home/avin/QM/material_data/CPM/4000_filtered_composites_final.csv"


def load_material_data(dataset_path):

	# csv_path = os.path.join(dataset_path, "4000_filtered_composites_final.csv")
	return pd.read_csv(dataset_path)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)


compound_data = load_material_data(DATASET_PATH)

compound_data = compound_data.dropna(subset=['Tcf', 'Tcf1', 'Tcf2', 'V1', 'V2'])

compound_data = compound_data[['er', 'Qf', 'Tcf', 'er1', 'Qf1', 'Tcf1', 'er2', 'Qf2', 'Tcf2', 'V1', 'V2']]

# print compound_data
# corr_matrix = compound_data.corr()
# print(corr_matrix["er"].sort_values(ascending=False))

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def denormalize(df_normalized,df):
	result = df_normalized.copy()
	for feature_name in df_normalized.columns:
		max_value = df[feature_name].max()
		min_value = df[feature_name].min()
		result[feature_name] = df_normalized[feature_name] * (max_value - min_value) + min_value
	return result


compound_data_normalized = normalize(compound_data)

compound_data_normalized = shuffle(compound_data_normalized,random_state=12)

train_set, test_set = train_test_split(compound_data_normalized, test_size=0.15, random_state=42)


train_xs_er = train_set[['er1', 'er2', 'V1', 'V2']]
train_ys_er = train_set[['er']]

test_xs_er = test_set[['er1', 'er2', 'V1', 'V2']]
test_ys_er = test_set[['er']]


train_xs_Qf = train_set[['Qf1', 'Qf2', 'V1', 'V2']]
train_ys_Qf = train_set[['Qf']]

test_xs_Qf = test_set[['Qf1', 'Qf2', 'V1', 'V2']]
test_ys_Qf = test_set[['Qf']]


train_xs_Tcf = train_set[['Tcf1', 'Tcf2', 'V1', 'V2']]
train_ys_Tcf = train_set[['Tcf']]

test_xs_Tcf = test_set[['Tcf1', 'Tcf2', 'V1', 'V2']]
test_ys_Tcf = test_set[['Tcf']]



def CPM_DNN(er_in, Qf_in, Tcf_in):

	#DNN for permittivity prediction

	dense1 = tf.layers.dense(er_in, 24, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
	dense2 = tf.layers.dense(dense1, 48, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
	dense3 = tf.layers.dense(dense2, 12, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
	er_out = tf.layers.dense(dense3, 1, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))


	#DNN for Q*f prediction

	#concatenate er_out with Qf_in
	Qf_input = tf.concat([er_out, Qf_in], 1)
	# Qf_input2 = tf.concat([er_out, Qf_input1], 1)
	# Qf_input3 = tf.concat([er_out, Qf_input2], 1)
	# Qf_input = tf.concat([er_out, Qf_input3], 1)

	dense5 = tf.layers.dense(Qf_input, 24, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
	dense6 = tf.layers.dense(dense5, 48, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
	dense7 = tf.layers.dense(dense6, 12, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
	Qf_out = tf.layers.dense(dense7, 1, activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))


	#DNN for Tcf prediction

	#concatenate er_out with Tcf_in
	Tcf_input = tf.concat([er_out, Tcf_in], 1)
	# Tcf_input2 = tf.concat([er_out, Tcf_input1], 1)
	# Tcf_input3 = tf.concat([er_out, Tcf_input2], 1)
	# Tcf_input = tf.concat([er_out, Tcf_input3], 1)

	dense9 = tf.layers.dense(Tcf_input, 24, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
	dense10 = tf.layers.dense(dense9, 48, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
	dense11 = tf.layers.dense(dense10, 12, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
	Tcf_out = tf.layers.dense(dense11, 1, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))


	return er_out, Qf_out, Tcf_out



def train(learn_rate, batch_size):

	#placeholders for inputs and targets
	er_in = tf.placeholder(tf.float32, (None, 4), name='er_in')
	er_labels = tf.placeholder(tf.float32, (None, 1), name='er_labels')

	Qf_in = tf.placeholder(tf.float32, (None, 4), name='Qf_in')
	Qf_labels = tf.placeholder(tf.float32, (None, 1), name='Qf_labels')

	Tcf_in = tf.placeholder(tf.float32, (None, 4), name='Tcf_in')
	Tcf_labels = tf.placeholder(tf.float32, (None, 1), name='Tcf_labels')

	predicted_er, predicted_Qf, predicted_Tcf = CPM_DNN(er_in, Qf_in, Tcf_in)
	predicted_er = tf.identity(predicted_er, name='predicted_er')
	predicted_Qf = tf.identity(predicted_Qf, name='predicted_Qf')
	predicted_Tcf = tf.identity(predicted_Tcf, name='predicted_Tcf')

	er_loss = tf.losses.mean_squared_error(predicted_er, er_labels)
	Qf_loss = tf.losses.mean_squared_error(predicted_Qf, Qf_labels)
	Tcf_loss = tf.losses.mean_squared_error(predicted_Tcf, Tcf_labels)

	total_loss = er_loss + Qf_loss + Tcf_loss
	loss = tf.reduce_mean(total_loss, name='loss')

	opt = tf.train.AdamOptimizer(learn_rate).minimize(loss)

	def do_report(e):

		r = sess.run([predicted_er, predicted_Qf, predicted_Tcf, loss], 
			feed_dict={er_in: test_xs_er, er_labels: test_ys_er, Qf_in: test_xs_Qf, Qf_labels: test_ys_Qf, Tcf_in: test_xs_Tcf, Tcf_labels: test_ys_Tcf})

		predicted_outputs_pd = pd.DataFrame({'er': r[0][:,0], 'Qf': r[1][:,0], 'Tcf': r[2][:,0]}, columns=['er', 'Qf', 'Tcf'])

		denormalized_labels = np.array(denormalize(test_ys_Tcf,compound_data))
		denormalized_outputs = np.array(denormalize(predicted_outputs_pd,compound_data))
		
		print np.c_[denormalized_outputs[:,2].reshape(len(test_set),1), denormalized_labels.reshape(len(test_set),1)]
		print 'loss at the moment: ', np.sqrt(r[3])


	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		
		sess.run(init)

		try:
			epochs = 100000
			for e in range(epochs):
				print 'epoch :', e
				for i in range(len(train_xs_er) / batch_size):
					
					index=i*batch_size
					batch_xs_er = train_xs_er[index:index + batch_size]
					batch_ys_er = train_ys_er[index:index + batch_size]

					batch_xs_Qf = train_xs_Qf[index:index + batch_size]
					batch_ys_Qf = train_ys_Qf[index:index + batch_size]

					batch_xs_Tcf = train_xs_Tcf[index:index + batch_size]
					batch_ys_Tcf = train_ys_Tcf[index:index + batch_size]
					# if (i<(len(test_ys)/batch_size)):
					# 	test_batch_xs = test_xs[index:index + batch_size]
					# 	test_batch_ys = test_ys[index:index + batch_size]


					sess.run(opt, feed_dict={er_in: batch_xs_er, er_labels: batch_ys_er, Qf_in: batch_xs_Qf, Qf_labels: batch_ys_Qf, Tcf_in: batch_xs_Tcf, Tcf_labels: batch_ys_Tcf})

				do_report(e)


		except KeyboardInterrupt:
			print 'Training stopped manually'
			save_path = saver.save(sess, "/home/avin/QM/material_data/CPM/saved_model/cpm_dnn.ckpt")
  	# 		print("Model saved in path: %s" % save_path)



if __name__ == "__main__":

	train(learn_rate=0.001,
		  batch_size=10)
