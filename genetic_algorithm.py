import sys
import numpy as np
import pandas as pd
# import os
# import time
import predict_properties_genetic
from sklearn.metrics import mean_squared_error
import random
import time
import sys
from PyQt4 import QtGui
from Tkinter import *
import Tkinter
import tkMessageBox


np.set_printoptions(threshold=np.nan)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)



def run_GUI():

	window = Tk()

	window.title("Dielectric Material Design")
	 
	lbl1 = Label(window, text="Expected Permittivity")
	lbl1.grid(column=0, row=0)

	txt1 = Entry(window,width=10)
	txt1.grid(column=1, row=0)

	lbl2 = Label(window, text="Expected Q*f")
	lbl2.grid(column=0, row=1)

	txt2 = Entry(window,width=10)
	txt2.grid(column=1, row=1)

	lbl3 = Label(window, text="Expected Tcf")
	lbl3.grid(column=0, row=2)

	txt3 = Entry(window,width=10)
	txt3.grid(column=1, row=2)

	lbl4 = Label(window, text="Population Size")
	lbl4.grid(column=0, row=3)

	txt4 = Entry(window,width=10)
	txt4.grid(column=1, row=3)

	lbl5 = Label(window, text="No of Generations")
	lbl5.grid(column=0, row=4)

	txt5 = Entry(window,width=10)
	txt5.grid(column=1, row=4)
	# txt3.focus()

	# lbl4 = Label(window, width=10)
	# lbl4.grid(column=1, row=16)

	def clicked():
		# res = "Expected Tcf " + txt3.get()
		# lbl4.configure(text= res)
		global expected_er
		global expected_Qf
		global expected_Tcf
		global population_size
		global no_of_generations
		expected_er = float(txt1.get())
		expected_Qf = float(txt2.get())
		expected_Tcf = float(txt3.get())
		population_size = int(txt4.get())
		no_of_generations = int(txt5.get())
		window.destroy()
		# run_GA(10)
		# return expected_er, expected_Qf, expected_Tcf

	btn = Button(window, text="Calculate", bg="green", fg="black", command=clicked)

	btn.grid(column=1, row=5)

	window.geometry('350x150')
	 
	window.mainloop()

run_GUI()

start_time = time.time()

# print(sys.argv[2])

# expected_er = 80
# expected_Qf = 20000
# expected_Tcf = 10.0

expected_properties = np.array([expected_er, expected_Qf, expected_Tcf]).reshape(1,3)

SC_CRYSTALS_PATH = "filtered_single_crystals.csv"

def load_material_data(dataset_path):
	return pd.read_csv(dataset_path)

single_crystals = load_material_data(SC_CRYSTALS_PATH)

# print single_crystals.info()


# population_size = 10
k_size = int(population_size*0.3)
elite_size = int(population_size*0.1)
mating_pool_size = int(population_size*0.4)

ID = np.array(single_crystals['ID'])

# print ID

#create population by mixing materials randomly
POPULATION = np.zeros(shape=(population_size,2))

# expected_df = pd.DataFrame(columns=['er', 'Qf', 'Tcf'])
expected_df = pd.DataFrame({'er':expected_properties[:,0],'Qf':expected_properties[:,1], 'Tcf':expected_properties[:,2]})
# expected_df['er'] = expected_er
# expected_df['Qf'] = expected_Qf
# expected_df['Tcf'] = expected_Tcf
# print expected_df

expected_df_normalized = predict_properties_genetic.normalize(expected_df,predict_properties_genetic.compound_data)
norm_exp_er = expected_df_normalized.iloc[0]['er']
norm_exp_Qf = expected_df_normalized.iloc[0]['Qf']
norm_exp_Tcf = expected_df_normalized.iloc[0]['Tcf']

#generate a new population
def generate_population(population):
	for i in range (population_size):

		choice1 = np.random.choice(ID)

		index = np.argwhere(ID==choice1)
		ID2 = np.delete(ID, index)

		choice2 = np.random.choice(ID2)

		population[i][0] = choice1
		population[i][1] = choice2

	return population 


# print generate_population(POPULATION)


def generate_dataframe(sc_pair):

	composite = pd.DataFrame(columns=['er1', 'Qf1', 'Tcf1', 'er2', 'Qf2', 'Tcf2', 'V1', 'V2'])
	composite['V1'] = np.linspace(1,0,101)
	composite['V2'] = np.linspace(0,1,101)

	er1 = single_crystals.loc[single_crystals.ID == sc_pair[0], 'er'].to_string(index=False)
	composite['er1'] =  np.repeat(er1, 101)

	Qf1 = single_crystals.loc[single_crystals.ID == sc_pair[0], 'Qf'].to_string(index=False)
	composite['Qf1'] =  np.repeat(Qf1, 101)

	Tcf1 = single_crystals.loc[single_crystals.ID == sc_pair[0], 'Tcf'].to_string(index=False)
	composite['Tcf1'] =  np.repeat(Tcf1, 101)

	er2 = single_crystals.loc[single_crystals.ID == sc_pair[1], 'er'].to_string(index=False)
	composite['er2'] =  np.repeat(er2, 101)

	Qf2 = single_crystals.loc[single_crystals.ID == sc_pair[1], 'Qf'].to_string(index=False)
	composite['Qf2'] =  np.repeat(Qf2, 101)

	Tcf2 = single_crystals.loc[single_crystals.ID == sc_pair[1], 'Tcf'].to_string(index=False)
	composite['Tcf2'] =  np.repeat(Tcf2, 101)
	# print composite
	return composite.apply(pd.to_numeric)

# print predict_properties_genetic.predict_properties(generate_dataframe([106,4]))

# selected_df = predict_properties_genetic.predict_properties(generate_dataframe([64,4]))

# print generate_dataframe([61,89]).info()
#call the cpm script and determine the fitness here

def determine_fitness(df,sc_pair):
	max_fitness = 0
	max_fitness_row = 0
	V1_at_max = 0
	predictions_at_max = 0
	for index, row in df.iterrows():
		# print row
		V1 = 1 - index*0.01
		denormalized_predictions = np.array(predict_properties_genetic.denormalize(pd.DataFrame(row).transpose(), predict_properties_genetic.compound_data))
		predicted_er = row['er']
		predicted_Qf = row['Qf']
		predicted_Tcf = row['Tcf']

		er_mse = mean_squared_error([predicted_er], [norm_exp_er])
		Qf_mse = mean_squared_error([predicted_Qf], [norm_exp_Qf])
		Tcf_mse = mean_squared_error([predicted_Tcf], [norm_exp_Tcf])

		fitness = 1.0/(er_mse + Qf_mse + Tcf_mse)

		if (fitness > max_fitness):
			max_fitness = fitness
			max_fitness_row = row
			V1_at_max = V1
			predictions_at_max = denormalized_predictions
			# predicted_er_at_max = predicted_er
			# predicted_Qf_at_max = predicted_Qf
			# predicted_Tcf_at_max = predicted_Tcf

	# print max_fitness
	return np.array([np.array(sc_pair), max_fitness, V1_at_max, predictions_at_max])

# print determine_fitness(selected_df)

#input to the generate_mating_pool function will be an array of [[sc1,sc2], fitness, V1] sorted according to the fitness
#apply elitism here
#sc_pair_fitness must be a numpy array
def generate_mating_pool(sc_pair_fitness, k_size, elite_size):

	Sc_pair_fitness = np.array(sc_pair_fitness)

	sc_pairs_sorted = Sc_pair_fitness[Sc_pair_fitness[:,1].argsort()[::-1]]

	mating_pool = []
	mating_pool.append(sc_pairs_sorted[0][0])
	j=1
	while True:
		if not ((np.array(mating_pool) == np.array(sc_pairs_sorted[j][0])).all(1).any()):
			mating_pool.append(sc_pairs_sorted[j][0])
		if len(mating_pool)==elite_size:
			break
		j+=1
		
	# print mating_pool

	

	for i in range(mating_pool_size - elite_size):
		#tournament selection, select k number of sc_pairs randomly
		tournament_pairs = np.array(random.sample(sc_pairs_sorted[elite_size:],k_size))

		# sort tournament pairs
		tournament_pairs_sorted = tournament_pairs[tournament_pairs[:,1].argsort()[::-1]]
		j=0
		while True:
			if not ((np.array(mating_pool) == np.array(tournament_pairs_sorted[j][0])).all(1).any()):
				mating_pool.append(tournament_pairs_sorted[j][0])

		#delete the added pair having row
				sc_pairs_sorted = np.delete(sc_pairs_sorted, np.argwhere(sc_pairs_sorted[:,1]==tournament_pairs_sorted[j][1]),0)
				break
			# else:
			# 	print 'pair inside', tournament_pairs_sorted[j][0]
			j+=1

	# print mating_pool
	return mating_pool


def breed(parent1, parent2):

	while True:
		child = [random.choice(parent1), random.choice(parent2)]
		if child[0] != child[1]:
			break

	return child


def next_generation(mating_pool, population_size, elite_size):

	children = []
	# rest_length = population_size - elite_size

	for i in range(elite_size):
		children.append(mating_pool[i])


	while True:

		[parent1, parent2] = random.sample(mating_pool[elite_size:],2)

		child = breed(parent1, parent2)
		if not ((np.array(children) == np.array(child)).all(1).any()):
			children.append(child)
		# else:
		# 	print 'child inside', child
		if (len(children) == population_size):
			break

	return children

def run_GA(no_of_generations):
	print 'Discovering Materials....'
	current_generation = generate_population(POPULATION)
	counter =0
	sc_pair_fitness = []
	final_fitness_array = []
	for j in range(no_of_generations):
		print 'Generation: ', j+1
		for i in current_generation:
			#generate dataframe from current single crystal pair (composite)
			generated_dataframe = generate_dataframe(i)
			#predict properties of the current composite using the forward DNN
			predicted_properties = predict_properties_genetic.predict_properties(generated_dataframe)
			#i is the current composite
			single_crystal_pair = i
			#find the fitness of the current composite / output of the determine fitness function is not only fitness but also [composite, fitness, V1, predictions]
			current_fitness = determine_fitness(predicted_properties,single_crystal_pair)
			#append the current results, so that finally we have 
			sc_pair_fitness.append(current_fitness)
			
			counter+=1
			# print counter
		# print np.array(sc_pair_fitness)[:,1]
		current_mating_pool = generate_mating_pool(sc_pair_fitness,k_size,elite_size)
		current_generation = next_generation(current_mating_pool,population_size,elite_size)
	
	for i in current_generation:
		final_fitness_array.append(determine_fitness(predict_properties_genetic.predict_properties(generate_dataframe(i)),i))

	final_fitness_array = np.array(final_fitness_array)
	sorted_final_fitness_array = final_fitness_array[final_fitness_array[:,1].argsort()[::-1]]
	# print sorted_final_fitness_array.shape
	#filter single crystals
	single_crystal1 = []
	single_crystal2 = []
	#get the IDs of the single crystals
	sc1_encoded = np.array(sorted_final_fitness_array[:,0].tolist())[:,0].astype(int)
	sc2_encoded = np.array(sorted_final_fitness_array[:,0].tolist())[:,1].astype(int)

	for i,j in zip(sc1_encoded, sc2_encoded):
		#get the real material related to the ID of the single crystal
		single_crystal1.append(single_crystals.loc[single_crystals['ID'] == i, 'Material'].iloc[0])
		single_crystal2.append(single_crystals.loc[single_crystals['ID'] == j, 'Material'].iloc[0])
	
	#getting V1 and converting object type to float to do rounding, otherwise it gives an error
	V1 = np.around(sorted_final_fitness_array[:,2].astype(np.double), 2)
	V2 = np.around((1.000 - V1), 2)
	# print np.array(sorted_final_fitness_array[:,3].tolist())
	predicted_er = np.array(sorted_final_fitness_array[:,3].tolist()).reshape(population_size,3)[:,0]
	predicted_Qf = np.array(sorted_final_fitness_array[:,3].tolist()).reshape(population_size,3)[:,1]
	predicted_Tcf = np.array(sorted_final_fitness_array[:,3].tolist()).reshape(population_size,3)[:,2]


	print '\n\nExpected Properties: '
	print '\t Expected Permittivity  : ', expected_er
	print '\t Expected Qf\t\t: ', expected_Qf
	print '\t Expected Tcf\t\t: ', expected_Tcf
	# print predicted_er, predicted_Qf, predicted_Tcf
	# print V1,V2
	final_results = pd.DataFrame({'ID1': sc1_encoded,'single_crystal1': single_crystal1, 'ID2': sc2_encoded, 'single_crystal2': single_crystal2, 'V1': V1, 'V2': V2, 'Predicted Permittivity': predicted_er, 'Predicted Qf': predicted_Qf, 'Predicted Tcf': predicted_Tcf}, columns=['ID1', 'single_crystal1', 'ID2', 'single_crystal2', 'V1', 'V2', 'Predicted Permittivity', 'Predicted Qf', 'Predicted Tcf'])
	print '\n\n',final_results.head(50).to_string(index=False)

	print("\n\n--- %s minutes taken to run ---\n" % ((time.time() - start_time)/60.0))


	
if __name__ == "__main__":
	run_GA(no_of_generations)
	# run_GUI()