# Read the training sample

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

    row_length = len(csv_data[0])

    print "There are", len(csv_data), "rows with length", row_length, "and", row_length - 3, "features"

    print "Reading Data complete"


data_by_qid = []
i, j = 0, 0

while i < len(csv_data):
    qid = csv_data[i][1]
    data_by_qid.append([])
    while i < len(csv_data) and csv_data[i][1] == qid:
        data_by_qid[j].append(csv_data[i])
        i += 1
    j += 1

print "Num questions:", len(data_by_qid)


# with open(output_filename, 'w') as output:
#     writer = csv.writer(output, delimiter=' ')

#     writer.writerow(sum_row)