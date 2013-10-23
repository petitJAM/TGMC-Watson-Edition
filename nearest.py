import csv, math
from sklearn.neighbors import KNeighborsClassifier

dataset_dir = "datasets/"
output_dir = "output/"

training_data_filename = dataset_dir + "tgmctrain.csv"
evaluation_data_filename = dataset_dir + "tgmcevaluation.csv"

output_filename = output_dir + "output.csv"
X = []
y = []
eval = []

with open(training_data_filename) as training_data:
  reader = csv.reader(training_data)

  print "Opened training CSV"

  for row in reader:
    row_data = []
    for data in row[2:-1]:
      row_data.append(float(data))

    X.append(row_data)
      
    if row[-1] == "true":
      y.append(1)
    else:
      y.append(0)

  print "Training data has been read"
  
neigh = KNeighborsClassifier(n_neighbors=2, weights='distance')
neigh.fit(X, y)

print "Data has been fitted"

with open(evaluation_data_filename) as eval_data:
  reader = csv.reader(eval_data)
  
  print "Opened evaluation CSV"
  
  for row in reader:
    row_data = []
    for data in row[2:]:
      row_data.append(float(data))
      
    eval.append(row_data)
    
  print "Evaluation data has been read"
  
ans = neigh.predict(eval)

for i, num in enumerate(ans):
  if num == 1:
    print i + 400001