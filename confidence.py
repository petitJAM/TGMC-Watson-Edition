import csv, math

import pprint
pp = pprint.PrettyPrinter()

dataset_dir = "datasets/"
output_dir = "output/"

training_data_filename = dataset_dir + "tgmctrain.csv"
evaluation_data_filename = dataset_dir + "tgmcevaluation.csv"

output_filename = output_dir + "output.csv"

def determination(x):
    b = math.fabs(x[0]) > 300 * x[1]
    b = b and x[1] != x[2]
    return b

with open(training_data_filename) as training_data:
    reader = csv.reader(training_data)

    print "Opened CSV"
    csv_data = []
    true_rows = {}
    false_rows = {}

    for i in range(0, 318):
        csv_data.append([0, float("-inf"), float("inf")])

    for row in reader:

        row_num = int(row[0]) - 1
        q_num = int(float(row[1]))

        validity = row[-1]
        if validity == "true":
            if q_num in true_rows:
                true_rows[q_num].append(row_num)
            else:
                true_rows[q_num] = [row_num]
            for i, data in enumerate(row[2:-1]):
                num = float(data)

                csv_data[i][0] += num

                if num > csv_data[i][1]:
                    csv_data[i][1] = num
                if num < csv_data[i][2]:
                    csv_data[i][2] = num
        else:
            if q_num in true_rows and q_num in false_rows:
                if len(false_rows[q_num]) < (len(true_rows[q_num]) * 1):
                    false_rows[q_num].append(row)
            elif q_num in true_rows and not q_num in false_rows:
                false_rows[q_num] = [row] 

    for q_num in false_rows:
        for row in false_rows[q_num]:
            for i, data in enumerate(row[2:-1]):
                num = float(data)

                csv_data[i][0] -= num

                if num > csv_data[i][1]:
                    csv_data[i][1] = num
                if num < csv_data[i][2]:
                    csv_data[i][2] = num

    row_length = len(csv_data)

    for i, col in enumerate(csv_data):
        if determination(col):
            print i

    print "Reading Data complete"