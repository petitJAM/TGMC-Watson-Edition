import csv

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
    for i in range(0, 318):
        csv_data.append([0, float("-inf"), float("inf")])
    for row in reader:
        true_rows = {}
        false_rows = {}

        row_num = int(row[0]) - 1
        q_num = int(row[1])

        validity = row[-1]
        for i, data in enumerate(row[2:-1]):
            num = float(data)
            if validity == "true":
                csv_data[i][0] += num
                if q_num in true_rows:
                    true_rows[q_num].append(row_num)
                else:
                    true_rows[q_num] = [row_num]

            elif validity == "false":
                csv_data[i][0] -= num
                if q_num in false_rows:
                    false_rows[q_num].append(row_num)
                else:
                    false_rows[q_num] = [row_num]

            if num > csv_data[i][1]:
                csv_data[i][1] = num
            if num < csv_data[i][2]:
                csv_data[i][2] = num

        if row_num % 10000 == 0:
            print row_num
    row_length = len(csv_data)

    print "There are", len(csv_data), "rows with length", row_length, "and", row_length - 3, "features"

    print "Master row: ", csv_data

    print "Reading Data complete"