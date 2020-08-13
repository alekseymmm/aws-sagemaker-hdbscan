from geo_config import *
#new code 
import pandas as pd
import numpy as np
import joblib

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn import svm

#from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier


def filter_outliers_in_features(X):
    # clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf = EllipticEnvelope(support_fraction=1, contamination=0.2)
    clf.fit(X)
    # r = clf.predict(X)
    X = X[clf.predict(X) == 1]
    return X


def filter_outliers_in_data(X, y):
    # clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf = EllipticEnvelope(support_fraction=1, contamination=0.2)
    clf.fit(X)
    # r = clf.predict(X)
    y = y[clf.predict(X) == 1]
    X = X[clf.predict(X) == 1]

    return X, y


def normalize_features(X):
    normalizer = preprocessing.Normalizer().fit(X)
    X = normalizer.transform(X)
    return X, normalizer


def scale_features(X):
    scaler = preprocessing.MinMaxScaler().fit(X)
    X = scaler.transform(X)
    return X, scaler


def train_model(df, outliers_fraction=0.1):
    if oil_classifier == 'SVM':
        algorithm = svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                    gamma='scale')

    elif oil_classifier == 'IsolationForest':
        algorithm = IsolationForest(n_estimators=400,
                                    contamination=outliers_fraction,
                                    random_state=42, n_jobs=4)
    else:
        print("Unsupported oil classifier")

    algorithm.fit(df.values)

    return algorithm

def train_model2(df):
    outliers_fraction = 0.05
    #algorithm = EllipticEnvelope(contamination=outliers_fraction)

    algorithm = IsolationForest(n_estimators=400,
                                contamination=outliers_fraction,
                                random_state=42, n_jobs=4)

    X = np.array(df)
    algorithm.fit(X)

    return algorithm

# TODO : use params here not hardcode
def lower_bound_filter(df):
    ind = None
    for feature in features_to_train:
        if ind is None:
            ind = df[feature] >= signal_lower_bound
        else:
            ind |= df[feature] >= signal_lower_bound

    return df[ind]

def filter_marker(df, features, clf):
    #plot_density3(df, features, None, 2)

    if lower_bound_filtering_enabled:
        df = lower_bound_filter(df)

    #plot_density3(df, features, None, 2)

    pred = clf.predict(np.array(df))
    ind = (pred != 1)
    df = df[ind]
    #plot_density3(df, features, None, 2)

    clf = train_model2(df)
    pred = clf.predict(np.array(df))
    ind = (pred == 1)
    df = df[ind]
    #plot_density3(df, features, None, 1)
    #plot_density3(df, features, None, 2)

    return df


def load_markers(filename, oil_clf):
    df = pd.read_csv(filename)
    print("From file ", filename, " loaded dataframe of size: ", df.shape)

    # drop events before start_time
    if not ignore_time and "Time" in df.keys():
        ind = df["Time"] > start_time
        df = df[ind]

    # only interesting features
    df = filter_data(df, features_to_train)

    df = filter_marker(df, features_to_train, oil_clf)
    print("After filtering marker data frame size: ", df.shape)

    return df


def create_filtered_trainig_set(filenames, marker_names):
    X_df = load_markers(filenames[0])
    X = np.asarray(X_df)
    # X, normalizer = normalize_features(X)
    X = filter_outliers_in_features(X)

    (nsamples, nfeatures) = X.shape
    y = np.full((nsamples, 1), marker_names[0])

    for i in range(1, len(filenames)):
        X1_df = load_markers(filenames[i])
        X1 = np.asarray(X1_df)
        # X1, normalizer = normalize_features(X1)
        X1 = filter_outliers_in_features(X1)

        (nsamples, nfeatures) = X1.shape
        y1 = np.full((nsamples, 1), marker_names[i])

        X = np.vstack((X, X1))
        y = np.vstack((y, y1))

    c, r = y.shape
    y = y.reshape(c, )
    return (X, y)


def create_unfiltered_trainig_set(filenames, marker_names):
    X_df = load_markers(filenames[0])
    X = np.asarray(X_df)

    (nsamples, nfeatures) = X.shape
    y = np.full((nsamples, 1), marker_names[0])

    for i in range(1, len(filenames)):
        X1_df = load_markers(filenames[i])
        X1 = np.asarray(X1_df)
        (nsamples, nfeatures) = X1.shape
        y1 = np.full((nsamples, 1), marker_names[i])

        X = np.vstack((X, X1))
        y = np.vstack((y, y1))

    c, r = y.shape
    y = y.reshape(c, )
    return (X, y)

def create_trainig_set():
    print("load oil markers...")
    oil_clf, oil_data = train_oil_model(oil_files_list)
    joblib.dump(oil_clf, "oil_model.pkl")

    filenames, marker_names = load_training_set(training_set_path)
    print(filenames)

    X_df = load_markers(filenames[0], oil_clf)
    X = X_df.values
    
    label = 0

    (nsamples, nfeatures) = X.shape
    y = np.full((nsamples, 1), label)

    label += 1
    for i in range(1, len(filenames)):
        X_df = load_markers(filenames[i], oil_clf)
        (nsamples, nfeatures) = X_df.shape
        y1 = np.full((nsamples, 1), label)

        X = np.vstack((X, X_df.values))
        y = np.vstack((y, y1))

        i += 1
        label += 1


    # add_oil_to_train_set
    nsamples = len(oil_data)
    y1 = np.full((nsamples, 1), label)

    X = np.vstack((X, oil_data.values))
    y = np.vstack((y, y1))

    c, r = y.shape
    y = y.reshape(c, )

    return X, y


def predict_with_confidence(clf, X_values, threshold):
    probs = clf.predict_proba(X_values)
    ind = np.where(np.max(probs, axis=1) >= threshold)

    y_pred = clf.predict(X_values)

    return ind, y_pred[ind]


def print_res(markers, counts):
    sum = 0
    for k, v in counts.items():
        if k != 'oil':
            sum += v

    for i in range(len(markers)):
        marker_name = markers[i]
        if markers[i] == 'oil':
            continue

        cnt = 0
        if marker_name in counts.keys():
            cnt = counts[marker_name]

        percent = 0
        if sum != 0:
            percent = cnt * 100 / sum
        print("%s %d (%.2f %%)" % (marker_name + " : ", cnt,
                                   percent))
    print("Total", sum, " samples classified")
    return


def print_counts(markers, df_dict):
    sum = 0
    for k, v in df_dict.items():
        if k != 'oil':
            sum += len(v)
    df = pd.DataFrame(columns=['marker', 'count'])
    for i in range(len(markers)):
        marker_name = markers[i]
        if markers[i] == 'oil':
            continue

        cnt = 0
        if marker_name in df_dict.keys():
            cnt = len(df_dict[marker_name])
        df = df.append({'marker': marker_name, 'count': cnt}, ignore_index=True)

        percent = 0
        if sum != 0:
            percent = cnt * 100 / sum
        print("%s %d (%.2f %%)" % (marker_name + " : ", cnt,
                                   percent))
    print("Total", sum, " samples classified")
    return df


def load_training_set(path):
    df = pd.read_csv(path)

    arr = np.asarray(df)
    filenames = arr[:, 0]
    marker_names = arr[:, 1]
    return filenames, marker_names


def filter_data(df, features_list):
    df = df[features_list]

    if remove_negative_signals:
        for feature in features_list:
            df = df[df[feature] > 0]
    else:
        for feature in features_list:
            df = df[df[feature] != 0]


    X = np.array(df)
    negative_mask = X < 0
    X = np.abs(X)
    X = np.log10(X)
    X[negative_mask] *= -1

    #X = np.cbrt(X)

    df = pd.DataFrame(X, columns=features_list)
    return df


def train_oil_model(oil_files=oil_files_list):
    print("Load oil data from: ", oil_files)
    oil = pd.concat((pd.read_csv(f) for f in oil_files))
    print("Total {} oil samples was loaded".format(len(oil)))

    if len(oil) > max_oil_samples:
        oil = oil.sample(n=max_oil_samples)
        print("Size of oil dataset was reduced to %d samples" % max_oil_samples)

    print("Start train model for oil classification...")
    oil = filter_data(oil, features_to_train)
    clf = train_model(oil, outliers_fraction=0.05)
    pred = clf.predict(np.array(oil))
    ind = (pred == 1)
    result = oil[ind]
    print("Finish train model for oil classification. "
          "Filtered data frame size: ", result.shape)

    return clf, result


def add_oil_to_train_set(X_train, y_train, oil_data):
    nsamples = len(oil_data)
    y1 = np.full((nsamples, 1), "oil")

    X_train = np.vstack((X_train, oil_data.values))
    y_train = np.vstack((y_train, y1))

    return X_train, y_train


def main():
    X, y = create_trainig_set()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    if (classifier == "random_forest"):
        clf = RandomForestClassifier(n_estimators=num_estimators, max_features=None, n_jobs=training_threads)
    else:
        clf = XGBClassifier(n_estimators=num_estimators, nthread=training_threads, max_depth=xgb_max_depth)

    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)
    print("Training accuracy score: ", score)

    for feat, importance in zip(features_to_train, clf.feature_importances_):
        print('feature: {f}, importance: {i}'.format(f=feat, i=importance))

    joblib.dump(clf, model_name)

    return 0


if __name__ == "__main__":
    pd.set_option('display.width', 1000)

    # run main script
    main()

    print("done")
