import csv
import gzip
import numpy as np
from sklearn import linear_model, cross_validation, datasets, svm
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

train_filename = 'train.csv.gz'
test_filename  = 'test.csv.gz'
pred_filename  = 'example_mean.csv'

# Load the training file.
train_data = []
train_features = []
target = []

ds = SupervisedDataSet(256, 1)

with gzip.open(train_filename, 'r') as train_fh:

    # Parse it as a CSV file.
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    
    # Skip the header row.
    next(train_csv, None)

    # Load the data.
    for row in train_csv:
        smiles   = row[0]
        features = np.array([float(x) for x in row[1:257]])
        gap      = float(row[257])
        
        train_data.append({ 'smiles':   smiles,
                            'features': features,
                            'gap':      gap })
        
        # Get a features matrix
        train_features.append(features)
 
        # Create the target vector
        target.append(gap)

        ds.addSample(features, (gap))

# Cross validation split
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(
#    train_features, target, test_size = 0.2, random_state = 0)

print "Begin building network"
net = buildNetwork(256, 256, 1)

print "begin building Trainer"
trainer = BackpropTrainer(net, ds)

print "Training network"
trainer.trainUntilConvergence()

# Fit to a linear model
#clf = linear_model.Lasso(alpha = .1)
#clf.fit(X_train, y_train)
#print "Training score:", clf.score(X_test,y_test)

# # Load the test file.
# test_data = []
# test_features = []
# with gzip.open(test_filename, 'r') as test_fh:

#     # Parse it as a CSV file.
#     test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    
#     # Skip the header row.
#     next(test_csv, None)

#     # Load the data.
#     for row in test_csv:
#         id       = row[0]
#         smiles   = row[1]
#         features = np.array([float(x) for x in row[2:258]])
        
#         test_data.append({ 'id':       id,
#                            'smiles':   smiles,
#                            'features': features })
        
#         test_features.append(features)

# # Write a prediction file.
# the_predictions = np.array([])
# with open(pred_filename, 'w') as pred_fh:

#     # Produce a CSV file.
#     pred_csv = csv.writer(pred_fh, delimiter=',', quotechar='"')

#     # Write the header row.
#     pred_csv.writerow(['Id', 'Prediction'])
    
#     the_predictions = clf.predict(test_features)

#     counter = 0
#     for datum in test_data:
#         pred_csv.writerow([datum['id'], the_predictions[counter]])
#         counter = counter + 1



