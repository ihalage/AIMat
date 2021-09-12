import os
import pandas as pd 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def load_material_data(dataset_path):
	return pd.read_csv(dataset_path)


def normalize(df,df_ref):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df_ref[feature_name].max()
        min_value = df_ref[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def denormalize(df_normalized,df):
	result = df_normalized.copy()
	for feature_name in df_normalized.columns:
		max_value = df[feature_name].max()
		min_value = df[feature_name].min()
		result[feature_name] = df_normalized[feature_name] * (max_value - min_value) + min_value
	return result

DATASET_PATH = "/home/avin/QM/material_data/CPM/4000_filtered_composites_final.csv"
compound_data = load_material_data(DATASET_PATH)
compound_data = compound_data.dropna(subset=['Tcf', 'Tcf1', 'Tcf2', 'V1', 'V2'])
compound_data = compound_data[['er', 'Qf', 'Tcf', 'er1', 'Qf1', 'Tcf1', 'er2', 'Qf2', 'Tcf2', 'V1', 'V2']]



NEW_DATA = '/home/avin/QM/material_data/CPM/new_compound_predict_inputs.csv'
new_inputs = load_material_data(NEW_DATA)

new_inputs = new_inputs[['er', 'Qf', 'Tcf', 'er1', 'Qf1', 'Tcf1', 'er2', 'Qf2', 'Tcf2', 'V1', 'V2']]

new_inputs_normalized = normalize(new_inputs,compound_data)

new_er = new_inputs_normalized[['er1', 'er2', 'V1', 'V2']]
new_Qf = new_inputs_normalized[['Qf1', 'Qf2', 'V1', 'V2']]
new_Tcf = new_inputs_normalized[['Tcf1', 'Tcf2', 'V1', 'V2']]


graph = tf.get_default_graph()  
imported_meta = tf.train.import_meta_graph("saved_model/cpm_dnn.ckpt.meta")

with tf.Session() as sess:

	imported_meta.restore(sess, "/home/avin/QM/material_data/CPM/saved_model/cpm_dnn.ckpt")
	print 'Model restored'

	predicted_er = graph.get_tensor_by_name('predicted_er:0')
	predicted_Qf = graph.get_tensor_by_name('predicted_Qf:0')
	predicted_Tcf = graph.get_tensor_by_name('predicted_Tcf:0')

	er_in = graph.get_tensor_by_name('er_in:0')
	Qf_in = graph.get_tensor_by_name('Qf_in:0')
	Tcf_in = graph.get_tensor_by_name('Tcf_in:0')

	r = sess.run([predicted_er, predicted_Qf, predicted_Tcf], feed_dict={er_in: new_er, Qf_in: new_Qf, Tcf_in: new_Tcf})

	# print r[0][:,0]

	predicted_outputs_pd = pd.DataFrame({'er': r[0][:,0], 'Qf': r[1][:,0], 'Tcf': r[2][:,0]}, columns=['er', 'Qf', 'Tcf'])
 	# print predicted_outputs_pd
 	# denormalized_labels = np.array(denormalize(new_inputs_normalized[['er', 'Qf', 'Tcf']],compound_data))
	denormalized_outputs = np.array(denormalize(predicted_outputs_pd,compound_data))

	# print denormalized_outputs
	print np.c_[denormalized_outputs[:,0].reshape(-1,1), denormalized_outputs[:,1].reshape(-1,1), denormalized_outputs[:,2].reshape(-1,1)]
	# x=new_inputs['V2']
	# plt.plot(x[0:101], denormalized_outputs[:,0][:101].reshape(-1,1), label='LaAlO3 - TiO2')
	# plt.plot(x[0:101], denormalized_outputs[:,0][101:202].reshape(-1,1), label='Bi2O3 - TiO2')
	# plt.plot(x[0:101], denormalized_outputs[:,0][202:303].reshape(-1,1), label='ZnAl2O4 - TiO2')
	# plt.title("Relative Permittivity")
	# plt.ylabel("Relative Permittivity")
	# plt.xlabel("TiO2 ratio")
	# plt.legend()
	# plt.savefig('effective permittivity',dpi=500)
	# plt.show()

	# plt.plot(x[0:101], denormalized_outputs[:,1][:101].reshape(-1,1), label='LaAlO3 - TiO2')
	# plt.plot(x[0:101], denormalized_outputs[:,1][101:202].reshape(-1,1), label='Bi2O3 - TiO2')
	# plt.plot(x[0:101], denormalized_outputs[:,1][202:303].reshape(-1,1), label='ZnAl2O4 - TiO2')
	# plt.title("Qf")
	# plt.ylabel("Qf")
	# plt.xlabel("TiO2 ratio")
	# plt.legend()
	# plt.savefig('effective Qf',dpi=500)
	# plt.show()

	# plt.plot(x[0:101], denormalized_outputs[:,2][:101].reshape(-1,1), label='LaAlO3 - TiO2')
	# plt.plot(x[0:101], denormalized_outputs[:,2][101:202].reshape(-1,1), label='Bi2O3 - TiO2')
	# plt.plot(x[0:101], denormalized_outputs[:,2][202:303].reshape(-1,1), label='ZnAl2O4 - TiO2')
	# plt.title("Tcf")
	# plt.ylabel("Tcf")
	# plt.xlabel("TiO2 ratio")
	# plt.legend()
	# plt.savefig('effective Tcf',dpi=500)
	# plt.show()
	# plt.plot(x, denormalized_outputs[:,1].reshape(-1,1), label='Qf')
	# plt.title("ZnAl2O4 - TiO2")
	# plt.ylabel("Qf")
	# plt.xlabel("TiO2 ratio")
	# plt.legend()
	# plt.savefig('ZnAl2O4-TiO2 Qf',dpi=500)
	# plt.show()
	# plt.plot(x, denormalized_outputs[:,2].reshape(-1,1), label='Tcf')
	# plt.title("ZnAl2O4 - TiO2")
	# plt.ylabel("Tcf")
	# plt.xlabel("TiO2 ratio")
	# plt.legend()
	# plt.savefig('ZnAl2O4-TiO2 Tcf',dpi=500)
	# plt.show()