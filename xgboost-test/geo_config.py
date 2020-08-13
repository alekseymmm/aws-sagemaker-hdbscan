# [general settings]
classifier = "xgboost"                   # "xgboost" or "random_forest"
training_set_path = "./training_set.txt" # path to the file describing training set
#training_set_path = "./training_set_old.txt" # path to the file describing training set
# list of test files
test_files_list = [
#'/home/alex/geo/Marov/HF_MIX_T/6h-MIX-1.csv',
'/home/alex/geo/Marov/HF_MIX_T/47h-MIX-1.csv',
#'/home/alex/geo/Marov/HF_MIX_T/55h-MIX-1.csv',
'/home/alex/geo/Marov/HF_MIX_T/71h-MIX-1.csv',
]

#oil_path = "/home/alex/geo/Aleksey_noise/B6_oil-2(1).csv"
oil_path = "/home/alex/geo/Marov/oil-0.csv"
oil_files_list = ["oil/oil-0.csv",
                  "oil/oil-0(1).csv"]
oil_classifier = 'SVM' # or 'IsolationForest'
max_oil_samples = 200000


model_name = "model_name.pkl"  # trained module will be saved (and loaded) with this name
#features_to_train = ["FSC-H","VL445-H","VL530-H","VL585-H","VL615-H","VL675-H","VL780-H","BL530-H","BL585-H","BL615-H","BL675-H","BL780-H", "RL675-H", "RL780-H"]
#features_to_train = ["BL530-H","RL675-H","VL445-H"] # only colors in features
features_to_train = ["BL-530-H", "BL-586-H", "BL-615-H", "RL-660-H", "RL-695-H", "VL-445-H","VL-530-H","YL-586-H","YL-615-H"]
#features_to_train = ["BL530-H", "RL675-H", "VL445-H", "VL530-H"] # only colors in features
start_time = 0                          # drop all events before this Time
ignore_time = True

# [classifier settings]
num_estimators = 1000 # 500 is better    # max number of trees in forest
xgb_max_depth = 4                      # max depth of tree for xgboost classifier
training_threads = 4                    # number of parallel threads for training

# [filtering settings]
outliers_filtering_enabled = True   # filter or not outliers
remove_negative_signals = True
#lower_bound_filtering_enabled = True
lower_bound_filtering_enabled = False
signal_lower_bound = 5  # measurements with signal lower
                        # than 10^(signal_lower_bound) are removed

preprocess_clusterization_enabled = True # True is muuuuch better
classify_noise = False

dbscan_eps = 0.25 # deprecated

# [3d plot settings]
x_axis = features_to_train[0]
y_axis = features_to_train[1]
z_axis = features_to_train[2]

marker_colors =  {"marker201": (0, 0, 1),
                  "marker202": (0, 1, 0),
                  "marker203": (1, 0, 0),
                  "marker204": (0, 1, 1),
                  "marker205": (1, 0, 1),
                  "marker206": (1, 1, 0),
                  "marker209": (1, 1, 1),
                  "marker210": (0, 0, 0.5),
                  "marker212": (0, 0.5, 0),
                  "marker213": (0.5, 0, 0),
                  "marker214": (0, 0.5, 0.5),
                  "marker218": (0.5, 0, 0.5),
                  "oil": (0, 0, 0)
                  }

plot_train_data_in_tests = True
