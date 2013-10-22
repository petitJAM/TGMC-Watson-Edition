#!/usr/bin/python

import csv, cPickle as pickle, time, random

# PyBrain imports
from pybrain.datasets import SupervisedDataSet, UnsupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import TanhLayer

dataset_dir = "datasets/"
output_dir = "output/"

td_fn = "tgmctrain.csv"
# td_fn = "training_clipped.csv"
ed_fn = "tgmcevaluation.csv"

training_data_filename = dataset_dir + td_fn
evaluation_data_filename = dataset_dir + ed_fn

training_data_pickle_name = td_fn + ".p"
evaluation_data_pickle_name = ed_fn + ".p"

output_filename = output_dir + "output.csv"

nFeatures = 318   #obtained from previous runs

# Configs

nHiddenLayers = 2
training_iterations = 1
learningRate = 0.05
momentum = 0.99
lrDecay = 0
weightDecay = 0

topN = 100

# makes it smaller by randomly sampling the answers for each question
def createBetterSupervisedDataSet(input_file):
	print "Creating a BETTER supervised dataset from", input_file

	ds = SupervisedDataSet(nFeatures, 1)

	try:
		with open(training_data_pickle_name, 'rb') as p:
			print "Loading from pickle"
			answers_by_question = pickle.load(p)
			print "Load successful"

	except IOError:
		answers_by_question = loadAnswersByQuestion(input_file)

		print "Saving to a pickle..."
		with open(training_data_pickle_name, 'wb') as p:
			pickle.dump(answers_by_question, p)
		print "Saved to", training_data_pickle_name

	# loop to load stuff into ds

	return None

def loadAnswersByQuestion(input_file):
	print "Loading from CSV"
	with open(input_file) as training_data:
		reader = csv.reader(training_data)

		answers_by_question = {}
		for row in reader:

			row_data = []
			correct = 100 if row[-1] == "true" else 0

			for data in row:
				try:
					row_data.append(float(data))
				except ValueError:
					if data == 'true':
						correct = 100

					elif data == 'false':
						correct = 0

					else:
						print "Something weird appeared"

			current_qid = int(row_data[1])

			if current_qid not in answers_by_question:
				answers_by_question[current_qid] = []

			# need some random sampling
			answers_by_question[current_qid].append((correct, row_data[2:]))

	for qid in answers_by_question.keys():
		print "Question:", qid
		print "Number of trues:", sum([1 if answer[0] else 0 for answer in answers_by_question[qid]])

	print "Total Number of questions:", len(answers_by_question.keys())

	return answers_by_question


def createSupervisedDataSetFromCSVFile(input_file):
	# init the dataset
	print "Creating a supervised dataset from", input_file

	ds = SupervisedDataSet(nFeatures, 1)

	correct_count = 0

	with open(input_file) as training_data:
	    reader = csv.reader(training_data)

	    for row in reader:
	        row_data = []
	        correct = 100 if row[-1] == "true" else 0

	        if correct > 0 or random.random() < .9999: # speed up
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

	        if correct:
	        	correct_count += 1

	        if (not correct) and (random.random() > 0.9999):
	        	break

	        ds.addSample(tuple(row_data[2:]), (correct,)) # drop the Qid and Aid

	print "Dataset created with size", len(ds), "and", ds.indim, "features."
	print "Found", correct_count, "correct answers"

	return ds


def createUnsupervisedDataSetFromCSVFile(input_file):
	# init the dataset
	print "Creating an unsupervised dataset from", input_file

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

	        ds.append((row_data[:2], tuple(row_data[2:]))) # drop the Qid and Aid

	return ds


def main():
	startTime()
	####################

	print "Loading the training dataset..."

	try: # Try to load the saved object from a pickle
		with open(training_data_pickle_name) as p:
			print "Loading from pickle"
			ds = pickle.load(p)

	except IOError: # Pickle file didn't exists
		print "Loading from CSV"
		ds = createSupervisedDataSetFromCSVFile(training_data_filename)

		# with open(training_data_pickle_name, 'w') as p:
		# 	print "Saving pickle..."
		# 	ds.save_pickle(p)

	print "Loading took", nextTime(), "seconds"
	print

	####################

	print "Building Network..."
	net = buildNetwork(ds.indim, nHiddenLayers, ds.outdim, recurrent=True)

	# print "Build Network took", nextTime(), "seconds"

	####################

	print "Create trainer..."
	print "learningRate:", learningRate
	print "momentum:", momentum
	print "lrDecay:", lrDecay
	print "weightDecay:", weightDecay
	trainer = BackpropTrainer(net, ds, learningrate=learningRate, lrdecay=lrDecay, momentum=momentum, batchlearning=False, weightdecay=weightDecay)

	# print "Create Trainer took", nextTime(), "seconds"

	####################

	print "Training network", training_iterations, "times..."
	for _ in range(training_iterations):
		print "Error in training:", trainer.train() # Save this error and use as a confidence?

	print "Training took", nextTime(), "seconds"

	####################

	print "Loading the evaluation dataset..."
	test_data = openEvaluationData(evaluation_data_filename)

	print "Testing took", nextTime(), "seconds"

	test_results = []

	for d in test_data:
		test_results.append((d[0], net.activate(d[1])[0]))

	print "Average score:", sum([t[1] for t in test_results]) / len(test_results)

	highest = max(test_results)
	smallest = min(test_results)
	test_results.sort(key=lambda x: x[1])
	print "Top 10", test_results[-10:]
	print "Max:", highest, "  Min:", smallest

	# clear file
	open("answers.txt", 'w').close()

	# write to answers.txt
	with open("answers.txt", 'w') as ans:
		for a in [int(a[0][0]) for a in test_results[-topN:]]:
			print a
			ans.write(str(a))
			ans.write("\n")

	#trainer.testOnData(test_data, verbose=True)

####################

last_time = None
def startTime():
	global last_time
	last_time = time.time()

# returns the amount of time passed since the last call to this
# must call startTime before this
def nextTime():
	global last_time
	if last_time is None:
		return -1

	old_time = last_time
	last_time = time.time()

	return (last_time - old_time)

####################

def other_main():
	startTime()
	####################

	print "IN OTHER_MAIN"
	ds = createBetterSupervisedDataSet(training_data_filename)

	print "Loading took", nextTime(), "seconds"
	print

if __name__ == '__main__':
	other_main()
