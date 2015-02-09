
# coding: utf-8

# In[ ]:

import csv
import gzip
import numpy as np
from sklearn import linear_model

train_filename = 'train.csv.gz'
test_filename  = 'test.csv.gz'
pred_filename  = 'example_mean.csv'

# Load the training file.
train_data = []
train_features = []
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
        
        train_features.append(features)
        
target = np.array([datum['gap'] for datum in train_data])
        
clf = linear_model.Lasso(alpha = .1)
clf.fit(train_features, target)

# Load the test file.
test_data = []
test_features = []
with gzip.open(test_filename, 'r') as test_fh:

    # Parse it as a CSV file.
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    
    # Skip the header row.
    next(test_csv, None)

    # Load the data.
    for row in test_csv:
        id       = row[0]
        smiles   = row[1]
        features = np.array([float(x) for x in row[2:258]])
        
        test_data.append({ 'id':       id,
                           'smiles':   smiles,
                           'features': features })
        
        test_features.append(features)

# Write a prediction file.
the_predictions = np.array([])
with open(pred_filename, 'w') as pred_fh:

    # Produce a CSV file.
    pred_csv = csv.writer(pred_fh, delimiter=',', quotechar='"')

    # Write the header row.
    pred_csv.writerow(['Id', 'Prediction'])
    
    the_predictions = clf.predict(test_features)

    counter = 0
    for datum in test_data:
        pred_csv.writerow([datum['id'], the_predictions[counter]])
        counter = counter + 1


# In[ ]:



