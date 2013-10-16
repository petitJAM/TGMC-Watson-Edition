
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
evaluation_data_filename = dataset_dir + "tgmcevaluation.csv" #"evaluation_clipped.csv"

output_filename = output_dir + "output.csv"

nFeatures = 318   #obtained from previous runs

# Configs

nHiddenLayers = 4
training_iterations = 1
learningRate = 0.05
momentum = 0.99
lrDecay = 0
weightDecay = 0

def createSupervisedDataSetFromCSVFile(input_file):
	# init the dataset
	print "Creating a dataset from", input_file

	ds = SupervisedDataSet(nFeatures, 1)

	with open(input_file) as training_data:
	    reader = csv.reader(training_data)

	    for row in reader:
	        row_data = []
	        correct = 0 
	        for data in row:
	            try:
	                row_data.append(float(data))
	            except ValueError:
	                if data == "true":
	                    correct = 100
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

def openEvaluationData(input_file):
	# init the dataset
	print "Creating a dataset from", input_file

	ds = [] 

	with open(input_file) as training_data:
	    reader = csv.reader(training_data)

	    for row in reader:
	        row_data = []
	        for data in row:
	            try:
	                row_data.append(float(data))
	            except ValueError:
	            	print "Non-floatable value!"

	        ds.append(tuple(row_data[2:])) # drop the Qid and Aid

	return ds


def main():

	print "Loading the training dataset..."
	ds = createSupervisedDataSetFromCSVFile(training_data_filename)

	with open("training.p", 'w') as p:
		ds.save_pickle(p)

	print "Building Network..."
	net = buildNetwork(ds.indim, nHiddenLayers, ds.outdim, recurrent=True)

	print "Create trainer..."
	print "learningRate:", learningRate
	print "momentum:", momentum
	print "lrDecay:", lrDecay
	print "weightDecay:", weightDecay
	trainer = BackpropTrainer(net, ds, learningrate=learningRate, lrdecay=lrDecay, momentum=momentum, batchlearning=False, weightdecay=weightDecay)

	print "Training network", training_iterations, "times..."
	for _ in range(training_iterations):
		trainer.train()

	print "Loading the evaluation dataset..."
	test_data = openEvaluationData(evaluation_data_filename)

	test_results = []

	for d in test_data:
		test_results.append(net.activate(d))

	print sum(test_results) / len(test_results)
	highest = max(test_results)
	print highest, test_results.index(highest)
	print min(test_results)

	#trainer.testOnData(test_data, verbose=True)



if __name__ == '__main__':
	main()
