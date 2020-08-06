#!/usr/bin/env python

import os
import json
import pickle
import sys
import traceback

import pandas as pd
import hdbscan


# These are the paths to where SageMaker mounts interesting things in your container.

prefix = '/opt/ml/'

input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

# This algorithm has a single channel of input data called 'training'. Since we run in
# File mode, the input files are copied to the directory specified here.
channel_name='training'
training_path = os.path.join(input_path, channel_name)

# The function to execute the training.
def train():
    print('Starting the training.')
    try:
        # Read in any hyperparameters that the user passed with the training job
        with open(param_path, 'r') as tc:
            trainingParams = json.load(tc)
            
        print("trainig params: ", trainingParams)
        # Take the set of files and read them all into a single pandas dataframe
        input_files = [os.path.join(training_path, file) for file in os.listdir(training_path)]
        print(input_files)
        if len(input_files) == 0:
            raise ValueError(('There are no files in {}.\n' +
                              'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                              'the data specification in S3 was incorrectly specified or the role specified\n' +
                              'does not have permission to access the data.').format(training_path, channel_name))
        print('Training complete.')
        raw_data = [pd.read_csv(file, header=None) for file in input_files]
        train_data = pd.concat(raw_data)
        
        # Here we only support a single hyperparameter. Note that hyperparameters are always passed in as
        # strings, so we need to do any necessary conversions.
        min_cluster_size = trainingParams.get('min_cluster_size', None)
        if min_cluster_size is not None:
            min_cluster_size = int(min_cluster_size)
        
        core_dist_n_jobs = trainingParams.get('core_dist_n_jobs', None)
        if core_dist_n_jobs is not None:
            core_dist_n_jobs = int(core_dist_n_jobs)
        else:
            core_dist_n_jobs = 4
            
        clust = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                                cluster_selection_method='eom',
                                core_dist_n_jobs=core_dist_n_jobs)
        print("Start HDBSCAN clustering...")
        clust = clust.fit(train_data)
        
        labels = clust.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print("Clustering finished, found", n_clusters_, 'clusters')
        print(n_noise_, "samples marked as noise (not in any cluster)")
        
        # save the model
        with open(os.path.join(model_path, 'hdbscan-model.pkl'), 'wb') as out:
            pickle.dump(clust, out)
        
        print("model {} saved.".format('hdbscan-model.pkl'))
            
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)

if __name__ == '__main__':
    train()

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)