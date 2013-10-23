import csv, random
from sklearn import svm
from sklearn.preprocessing import StandardScaler

def load_train_csv(input_file='datasets/tgmctrain.csv'):
	data = []
	ans = []
	ids = []

	with open(input_file) as training_data:
		reader = reader = csv.reader(training_data)

		for row in reader:
			if random.random() > 0.01 and row[-1] != 'true':
				continue
			ans.append(row[-1])
			row = [float(x) for x in row[:-1]]
			ids.append(row[0])
			data.append(row[1:])
			
		

	return ids, data, ans

def load_eval_csv(input_file='datasets/tgmcevaluation.csv'):
	data = []
	ids = []

	with open(input_file) as training_data:
		reader = reader = csv.reader(training_data)

		for row in reader:

			row = [float(x) for x in row]
			ids.append(row[0])
			data.append(row[1:])
			

	return ids, data


if __name__ == '__main__':
	print 'Loading csv...'
	ids, data, ans = load_train_csv()
	print 'CSV loaded. %d rows' % len(data)
	
	print 'Start PreProcessing...'
	pre = StandardScaler()
	data = pre.fit_transform(data, ans)
	print 'PreProcessing Done.'

	clf = svm.SVC(gamma=0.0, verbose=True)

	print 'Fitting SVC...'
	print clf.fit(data,ans)
	print 'Training Done'

	print 'Load Eval CSV...'
	eval_ids, eval_data = load_eval_csv()
	print 'Eval Loaded'

	print 'Start Predicting'
	answers = clf.predict(eval_data)
	print 'Predicting Done'
	
	# clear file
	open("answers.txt", 'w').close()

	# write to answers.txt
	with open("answers.txt", 'w') as ans:
		for i in range(len(answers)):
			if answers[i] == 'true':
				ans.write(str(eval_ids[i]))
				ans.write("\n")

	trues = [x for x in answers if x == 'true']
	falses = [x for x in answers if x == 'false']

	print 'Number True: %d' % len(trues)
	print 'Number Falses: %d' % len(falses)

