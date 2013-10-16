
# CSV
import csv

# PyBrain imports
from pybrain.datasets import SupervisedDataSet, UnsupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import TanhLayer

dataset_dir = "datasets/"
output_dir = "output/"

# training_data_filename = dataset_dir + "tgmctrain.csv"
# evaluation_data_filename = dataset_dir + "tgmcevaluation.csv"

training_data_filename = dataset_dir + "training_clipped.csv"
evaluation_data_filename = dataset_dir + "evaluation_clipped.csv"

output_filename = output_dir + "output.csv"

nFeatures = 318   #obtained from previous runs

# Configs

nHiddenLayers = 4
training_iterations = 1
learningRate = 0.01
momentum = 0.99

def createSupervisedDataSetFromCSVFile(input_file):
	# init the dataset
	print "Creating a dataset from", input_file

	ds = SupervisedDataSet(nFeatures, 1)

	with open(input_file) as training_data:
	    reader = csv.reader(training_data)

	    for row in reader:
	        row_data = []
	        correct = None
	        for data in row:
	            try:
	                row_data.append(float(data))
	            except ValueError:
	                if data == "true":
	                    correct = 1
	                elif data == "false":
	                    correct = 0
	                else:
	                    print "Something weird appeared"

	        ds.addSample(tuple(row_data[2:]), (correct,)) # drop the Qid and Aid

	print "Dataset created with size", len(ds), "and", ds.indim, "features."

	return ds


def createUnsupervisedDataSetFromCSVFile(input_file):
	# init the dataset
	print "Creating a dataset from", input_file

	ds = UnsupervisedDataSet(nFeatures)

	with open(input_file) as training_data:
	    reader = csv.reader(training_data)

	    for row in reader:
	        row_data = []
	        for data in row:
	            try:
	                row_data.append(float(data))
	            except ValueError:
	            	print "Non-floatable value!"

	        ds.addSample(tuple(row_data[2:])) # drop the Qid and Aid

	print "Dataset created with size", len(ds), "and", ds.dim, "features."

	return ds

def main():

	print "Loading the training dataset..."
	ds = createSupervisedDataSetFromCSVFile(training_data_filename)

	with open("training.p", 'w') as p:
		ds.save_pickle(p)

	print "Building Network..."
	net = buildNetwork(ds.indim, nHiddenLayers, ds.outdim, recurrent=True)

	print "Create trainer..."
	# print "learningRate:", learningRate
	# print "momentum:", momentum
	trainer = BackpropTrainer(net, ds)

	print "Training network", training_iterations, "times..."
	for _ in range(training_iterations):
		trainer.train()

	print "Loading the evaluation dataset..."
	test_data = createUnsupervisedDataSetFromCSVFile(evaluation_data_filename)





if __name__ == '__main__':
	main()