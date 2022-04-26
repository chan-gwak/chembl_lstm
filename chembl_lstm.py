# chembl_lstm.py
# by Chan Gwak for DAT112 Neural Network Programming Assignment 3
# due 8 Jun 2021 Tuesday

# A script that uses LSTM (long short term memory) network 
# to predict the AlogP value (a measure of hydrophobicity)
# of a chemical, given its SMILES-format formula.
# Uses a dataset obtained from ChEMBL
# (https://www.ebi.ac.uk/chembl/).

# The file data.filtered.csv contains 909690 data rows, each of
# which represents a chemical and its properties.
# Of the 42 columns, columns 6 and 8 (counting from 0) contain 
# the AlogP and Smiles data, respectively.
# AlogP is a float like 22.57 or -8.86. Some Smiles strings are:
#   CC(=O)Oc1cc(S(=O)(=O)Nc2n[nH]c(Nc3ccc(C)cc3)n2)c(S)cc1Cl
#   O=C1N[C@H](c2c[nH]c3cc(Br)ccc23)CN=C1c1c[nH]c2cc(Br)ccc12

# To process each string as a "sentence", identify "words".
# To help the network understand the sentences, treat certain
# sequences as words:
#  * Cl, Br (multi-letter atoms in the "organic subset")
#  * =O (double-bonded oxygen, can only attach to 1 atom)
#  * Any unit in square brackets (e.g. [nH], [C@H], [N+], [O-])
#  * NOT letter-number combinations such as C1 or c23
# The position-labeling numbers are kept separate so that the
# network still understands c23 as an aromatic carbon 
# (connecting at bonds labeled 2 and 3). 
# All other characters are treated as words.
# Assumption: Square brackets will not contain square brackets.
#             i.e. [[Se]O4] is impossible.



import pandas as pd
import numpy as np
import sklearn.model_selection as skms
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
from matplotlib import pyplot as plt


# Hyperparameters
data_size = 20000 # how much of the data set to be used; max 909,690
# Max datasize on my computer: 70,000 ~ 80,000
spec_words = ['Cl','Br','=O'] # special words that network should identify
train_set_size = 0.8 # train_size; Use 0.2 of data for testing
num_lstm = 512 # number of lstm nodes
num_hidden = 24 # number of hidden layer neurons
drop_rate = 0.2 # rate of dropout
opt_batch_size = 128 # batch_size for sgd & variants; default 32
num_epochs = 120 # number of epochs for training

# Settings
verbosity = 1 # decides the verbosity of model.fit
make_plots = True # whether to make plots of loss

# Name of file containing the dataset.
# The file should be colocated with this script.
chem_fname = 'data.filtered.csv'


# Use the pandas package to read the file into a DataFrame.
# Columns 6 and 8 contain the AlogP and Smiles data, respectively.
print("\nReading ChEMB data.")
chem_df_full = pd.read_csv(chem_fname, usecols=[6, 8], 
	dtype={'AlogP': np.float64, 'Smiles': str})

# Take only a random portion of the data (determined by data_size).
chem_df = chem_df_full.sample(n=data_size)

# Separate out the inputs (the SMILES strings) and targets (the AlogP data).
# Convert the AlogP data to a numpy array of floats.
smi = chem_df['Smiles'] # Smiles
tgt = chem_df['AlogP'].to_numpy() # AlogP


# Define a function to parse the SMILES strings into lists of "words".
def parse_smiles(smiles_str):

	# Pad the left square brackets with a space on the left,
	# and the right brackets with a space on the right.
	proc_str = smiles_str.replace('[',' [')
	proc_str =   proc_str.replace(']','] ')

	# Parsing Stage 1: Split out the items like ' [***] '.
	proc_list = proc_str.split()

	# Initialize the finalized processed list.
	fproc_list = []

	# Parsing Stage 2: Split out the special words, and split the 
	# remaining characters.
	# Add the square-bracketed pieces to the final list.
	# For each special word (not already found in square brackets), 
	# pad it with spaces on both sides. 
	# After padding all such words, split to separate them.
	# Finally, completely split all non-special words.
	for piece in proc_list:
		if ('[' in piece) or (']' in piece):
			fproc_list += [piece]
		else:
			for word in spec_words:
				piece = piece.replace(word,' '+word+' ')
			proc_piece = piece.split()
			for subpiece in proc_piece:
				if subpiece in spec_words:
					fproc_list += [subpiece]
				else:
					fproc_list += list(subpiece)

	return fproc_list


# Use the above function to begin preprocessing the data.
parsed_smi = []
max_length = 0
for chem_str in smi:

	# Parse each SMILES string.
	parsed_str = parse_smiles(chem_str)

	# Update the maximum length.
	if len(parsed_str) > max_length:
		max_length = len(parsed_str)

	# Accumulate parsed strings.
	parsed_smi += [parsed_str]

# Pad each parsed string with '' to reach max length.
for parsed_str in parsed_smi:
	parsed_str += [''] * (max_length - len(parsed_str))

# Create a list of all "words".
flat_parsed_smi = [word for parsed_str in parsed_smi for word in parsed_str]
words = sorted(list(set(flat_parsed_smi)))
n_words = len(words)

# Create word-index encodings.
encoding = {w: i for i, w in enumerate(words)}

# One-hot encode the data.
# First set up an empty numpy array:
inp = np.zeros((data_size, max_length, n_words), dtype = np.bool)
# Then populate it:
for i, parsed_str in enumerate(parsed_smi):
	for j, word in enumerate(parsed_str):
		inp[i, j, encoding[word]] = 1


# Split the data into training and testing data sets.
train_inp, test_inp, train_tgt, test_tgt = skms.train_test_split(inp, tgt, 
	train_size = train_set_size)


# A function to build the model.
# After the input layer is a layer of LSTMs 
# to process the inputs as "sentences".
# This is followed by a dense hidden layer.
# The final layer is a single node which returns the output,
# the predicted AlogP value (which could be any float).
# Dropouts occur between the layers to reduce overfitting.
def build_model(numlstm, numnodes, dropout):

	model = km.Sequential()

	model.add(kl.LSTM(num_lstm, input_shape = (max_length, n_words)))
	
	model.add(kl.Dropout(dropout))

	model.add(kl.Dense(numnodes, activation = 'tanh'))
	
	model.add(kl.Dropout(dropout))

	model.add(kl.Dense(1, activation = 'linear'))

	return model

# Build the network.
print("Building the network.\n")
model = build_model(num_lstm, num_hidden, drop_rate)

# Compile the network.
# As the target is a float, the loss (cost) is mean squared error.
# Note: As this is a regression problem (prediction of a quantity),
#       'accuracy' is meaningless; the model's goal is not to
#       predict exact values.
print("\nCompiling the model.")
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

print("Printing a summary of the model:\n")
model.summary()


# Train the network on the training data.
print("\nTraining the network.\n")
fit = model.fit(train_inp, train_tgt, batch_size = opt_batch_size,
	epochs = num_epochs, verbose = verbosity)


# Print out the final training accuracy and loss value.
print("")
score = model.evaluate(train_inp, train_tgt)
print("The training loss is:", score)

# Evaluate the network on the test data, 
# and print out the test loss value.
print("")
score = model.evaluate(test_inp, test_tgt)
print("The testing loss is:", score)

print("\nHere is a summary of the model again:\n")
model.summary()


# Plot the accuracy and loss over time.
if make_plots:
	print("\nPlotting loss.")
	plt.plot(fit.history['loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.savefig('loss.png')
