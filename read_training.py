# Read the training sample

import csv
import pprint

training_data_filename = "TGMC-training-sample.csv"
output_filename = "output.csv"
pp = pprint.PrettyPrinter()

with open(training_data_filename) as training_data:
    # get the length of the row
    row_len_reader = csv.reader(training_data)
    row_length = len(row_len_reader.next())
    del row_len_reader

    reader = csv.reader(training_data)
    
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