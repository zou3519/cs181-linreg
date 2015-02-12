
# coding: utf-8

# In[6]:

#Only need to run 1st and 3rd cell once each. To run a test, just run 2nd and 4th cell(very fast). 
#Then go into the Excel document and compute =SQRT(0.00002*SUM(C2:C50001))

#You also need to make a copy of the train file and make a csv file with an appropriate name (see below)

import csv
import gzip
import numpy as np
from sklearn import linear_model

train_filename = 'train.csv.gz'
test_filename  = 'train_copy.csv.gz'
pred_filename  = 'Some_Predictions.csv'

# Load the training file.
train_data = []
train_features = []
with gzip.open(train_filename, 'r') as train_fh:

    # Parse it as a CSV file.
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    
    # Skip the header row.
    next(train_csv, None)
    
    # Load the data, but only the first 50000 rows!
    increment = 0
    for row in train_csv:
        if (increment < 50000):
            smiles   = row[0]
            features = np.array([float(x) for x in row[1:257]])
            gap      = float(row[257])

            train_data.append({ 'smiles':   smiles,
                                'features': features,
                                'gap':      gap })

            train_features.append(features)
            
            increment = increment + 1

    target = np.array([datum['gap'] for datum in train_data])


# In[42]:

#Insert whatever regression technique here.

clf = linear_model.Ridge(alpha=5)
clf.fit(train_features, target)


# In[14]:

# Load the test file.
test_data = []
test_features = []
with gzip.open(test_filename, 'r') as test_fh:

    # Parse it as a CSV file.
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    
    # Skip the header row.
    next(test_csv, None)

    # Load the data.
    theCount = 0
    for row in test_csv:
        if (theCount < 50000):
            pass
        elif (theCount >= 50000):
            if (theCount < 100000):
                smiles   = row[0]
                features = np.array([float(x) for x in row[1:257]])
                gap      = float(row[257])

                test_data.append({ 'smiles':   smiles,
                                'features': features,
                                'gap':      gap })

                test_features.append(features)
        
        theCount = theCount + 1


# In[44]:

# Write a prediction file.
the_predictions = np.array([])
with open(pred_filename, 'w') as pred_fh:

    # Produce a CSV file.
    pred_csv = csv.writer(pred_fh, delimiter=',', quotechar='"')

    # Write the header row.
    pred_csv.writerow(['Actual Value', 'Prediction', 'Squared Difference'])
    
    the_predictions = clf.predict(test_features)

    counter = 0
    for datum in test_data:
        pred_csv.writerow([datum['gap'], the_predictions[counter], (datum['gap']-the_predictions[counter])**2])
        counter = counter + 1


# In[ ]:



