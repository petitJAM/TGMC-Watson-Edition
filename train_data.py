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

correctness_lo_val = 0
correctness_hi_val = 1

# could save some data in the pickle to check if these numbers have changed to auto-update

false_removal_factor = 1.1

# Configs

recurrent = True
nHiddenLayers = 318
training_iterations = 3
learningRate = 0.3
momentum = 0.9
lrDecay = .1
weightDecay = 0

topN = 500

def createNNforEachQuestionAnswerSet(input_file):
	pass

# makes it smaller by randomly sampling the answers for each question
def createBetterSupervisedDataSet(input_file):
	print "Creating a BETTER supervised dataset from", input_file

	ds = SupervisedDataSet(nFeatures, 1)
	answers_by_question = {}

	try:
		with open(training_data_pickle_name, 'rb') as p:
			print "Loading from pickle"
			answers_by_question = pickle.load(p)
			print "Load successful"
			print "Size of answers_by_question:", len(answers_by_question.keys())

	except IOError:
		answers_by_question = loadAnswersByQuestion(input_file)

		print "Saving to a pickle..."
		with open(training_data_pickle_name, 'wb') as p:
			pickle.dump(answers_by_question, p)
		print "Saved to", training_data_pickle_name

	# loop to load stuff into ds
	for qid in answers_by_question:
		for aid in answers_by_question[qid]:
			if aid != 'info':
				ds.addSample( tuple(answers_by_question[qid][aid]['data']), (answers_by_question[qid][aid]['target'], ) )
				# ds.addSample(tuple(ans[1]), (ans[0],))

	return ds

def loadAnswersByQuestion(input_file):
	print "Loading from CSV"
	with open(input_file) as training_data:
		reader = csv.reader(training_data)

		answers_by_question = {}
		for row in reader:

			row_data = []
			correct = correctness_lo_val

			if row[-1] == "true":
				correct = correctness_hi_val

			for data in row:
				try:
					row_data.append(float(data))
				except ValueError:
					pass

			# Question ID and Answer ID
			Q = int(row_data[1])
			A = int(row_data[0])

			if Q not in answers_by_question:
				answers_by_question[Q] = {}

			answers_by_question[Q][A] = { 'target': correct, 'data': row_data[2:] }

	for qid in answers_by_question.keys():
		# Do something using the number of trues in each set
		ntrue = sum([1 for a in answers_by_question[qid] if answers_by_question[qid][a]['target'] > correctness_lo_val])

		# happens before 'info' is added so it doesn't 
		nanswers = len(answers_by_question[qid])

		if ntrue < 1:
			del answers_by_question[qid]
			continue

		else:
			# we want about this many falses, so that true is roughly 1/2 of the sample
			n_false_to_remove = (nanswers - ntrue) - ntrue * 2
			print "Removing", n_false_to_remove, "false values from", nanswers

			false_keys = [fk for fk in answers_by_question[qid].keys() if answers_by_question[qid][fk]['target'] == correctness_lo_val]

			print "Removing", n_false_to_remove, "/", len(false_keys), "falses"

			for _ in range(n_false_to_remove):
				candidate = false_keys.pop(int(random.random() * 1000) % len(false_keys)) #random.choice(false_keys)
				del answers_by_question[qid][candidate]
				# false_keys.remove(candidate)

		nanswers = len(answers_by_question[qid])

		answers_by_question[qid]['info'] = {}
		answers_by_question[qid]['info']['length'] = nanswers
		answers_by_question[qid]['info']['nTrue'] = ntrue

		print qid, "True:", ntrue, "out of", nanswers
		# answers_by_question[qid] = [a for a in answers_by_question[qid] if a[0] > correctness_lo_val or random.random() > .75]


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
	        correct = correctness_hi_val if row[-1] == "true" else correctness_lo_val

	        if correct > correctness_lo_val or random.random() < .9999: # speed up
		        for data in row:
		            try:
		                row_data.append(float(data))
		            except ValueError:
		                if data == "true":
		                    correct = correctness_hi_val
		                elif data == "false":
		                    correct = correctness_lo_val
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

	print "Evaluation set opened"

	return ds


def main():
	startTime()
	####################

	print "Loading the training dataset..."

	ds = createBetterSupervisedDataSet(training_data_filename)

	# try: # Try to load the saved object from a pickle
	# 	with open(training_data_pickle_name) as p:
	# 		print "Loading from pickle"
	# 		ds = pickle.load(p)

	# except IOError: # Pickle file didn't exists
	# 	print "Loading from CSV"
	# 	ds = createSupervisedDataSetFromCSVFile(training_data_filename)

	print "Loading took", nextTime(), "seconds"
	print

	####################

	print "Building a", "Recurrent" if recurrent else "Feedforward", "Network"
	print "With", nHiddenLayers, "hidden layers"
	net = buildNetwork(ds.indim, nHiddenLayers, ds.outdim, recurrent=recurrent) # False for feedforward network

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
		print "   Iteration took:", nextTime(), "seconds"

	####################

	print "Loading the evaluation dataset..."
	test_data = openEvaluationData(evaluation_data_filename)

	test_results = []

	for d in test_data:
		# if random.random() > .75:
		test_results.append((d[0], net.activate(d[1])[0]))

	# print trainer.testOnData()

	print "Testing took", nextTime(), "seconds"

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
			ans.write(str(a))
			ans.write("\n")

	#trainer.testOnData(test_data, verbose=True)

####################

__last_time = None
def startTime():
	global __last_time
	__last_time = time.time()

# returns the amount of time passed since the last call to this
# must call startTime before this
def nextTime():
	global __last_time
	if __last_time is None:
		return -1

	old_time = __last_time
	__last_time = time.time()

	return (__last_time - old_time)

####################

def other_main():
	startTime()
	####################

	print "IN OTHER_MAIN"
	ds = createBetterSupervisedDataSet(training_data_filename)

	print "Loading took", nextTime(), "seconds"
	print

if __name__ == '__main__':
	main()
