import csv, math

import pprint
pp = pprint.PrettyPrinter()

dataset_dir = "datasets/"
output_dir = "output/"

training_data_filename = dataset_dir + "tgmctrain.csv"
evaluation_data_filename = dataset_dir + "tgmcevaluation.csv"

output_filename = output_dir + "output.csv"

with open(training_data_filename) as training_data:
    reader = csv.reader(training_data)

    print "Opened CSV"
    csv_data = []
    true_rows = {}
    false_rows = {}

    for i in range(0, 318):
        csv_data.append([0, float("-inf"), float("inf")])

    for row in reader:

        row_data = []
        for data in row:
            try:
                row_data.append(float(data))
            except ValueError:
                if data == "true":
                    row_data.append(True)
                elif data == "false":
                    row_data.append(False)
                else:
                    print "Something weird appeared"
        csv_data.append(row_data)


    print "Reading Data complete"