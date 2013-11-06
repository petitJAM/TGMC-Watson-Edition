#!/usr/bin/python

import csv, random

# PyBrain imports
from pybrain.datasets import SupervisedDataSet, UnsupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

correctness_lo_val = 0
correctness_hi_val = 1

nFeatures = 318
recurrent = True
nHiddenLayers = 4
training_iterations = 3
learningRate = 0.3
momentum = 0.9
lrDecay = .1
weightDecay = 0

def loadAnswersByQuestion(input_file="datasets/tgmctrain.csv"):
	print "Loading from CSV..."
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

	print "Filtering answers by question..."
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
			n_false_to_remove = 0
			# print "Removing", n_false_to_remove, "false values from", nanswers

			false_keys = [fk for fk in answers_by_question[qid].keys() if answers_by_question[qid][fk]['target'] == correctness_lo_val]

			# print "Removing", n_false_to_remove, "/", len(false_keys), "falses"

			for _ in range(n_false_to_remove):
				candidate = false_keys.pop(int(random.random() * 1000) % len(false_keys)) #random.choice(false_keys)
				del answers_by_question[qid][candidate]
				# false_keys.remove(candidate)

		nanswers = len(answers_by_question[qid])

		# answers_by_question[qid]['info'] = {}
		# answers_by_question[qid]['info']['length'] = nanswers
		# answers_by_question[qid]['info']['nTrue'] = ntrue

		# print qid, "True:", ntrue, "out of", nanswers
		# answers_by_question[qid] = [a for a in answers_by_question[qid] if a[0] > correctness_lo_val or random.random() > .75]


	print "Total Number of questions:", len(answers_by_question.keys())

	return answers_by_question

def openEvaluationData(input_file="datasets/tgmcevaluation.csv"):
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

def trainNetworkFromDataset(ds):

	# print "Building a", "Recurrent" if recurrent else "Feedforward", "Network"
	# print "With", nHiddenLayers, "hidden layers"
	net = buildNetwork(ds.indim, nHiddenLayers, ds.outdim, recurrent=True)

	# print "Create trainer..."
	# print "learningRate:", learningRate
	# print "momentum:", momentum
	# print "lrDecay:", lrDecay
	# print "weightDecay:", weightDecay
	trainer = BackpropTrainer(net, ds, learningrate=learningRate, lrdecay=lrDecay, momentum=momentum, batchlearning=False, weightdecay=weightDecay)

	for _ in range(training_iterations):
		trainer.train()

	return net, trainer

def createDataset(answers_row_data):
	ds = SupervisedDataSet(nFeatures, 1)
	for aid in answers_row_data.keys():
		ds.addSample( tuple(answers_row_data[aid]['data']), (answers_row_data[aid]['target'], ) )
	return ds

def main():

	abq = loadAnswersByQuestion()

	nets = {}

	print "Training", len(abq), "networks"
	for q_key in abq.keys():
		answers = abq[q_key]
		answers_dataset = createDataset(answers)
		net, trainer = trainNetworkFromDataset(answers_dataset)
		nets[q_key] = net

	eval_ds = openEvaluationData()

	for a_id, d in eval_ds:
		r = [nets[n_id].activate(d)[0] for n_id in nets]
		print "For", a_id,
		print "max:", max(r),
		print "min:", min(r)






if __name__ == '__main__':
	main()