import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from sklearn import svm, preprocessing, model_selection, metrics, pipeline, decomposition
from tqdm import tqdm

def stack_features(features, layer):
    num_reps = len(features[ids[0]])  # 100
    X, y, groups = [], [], []
    for rep in range(num_reps):
        for cls_id, cls_name in enumerate(ids):
            vec = np.asarray(features[cls_name][rep][layer]).ravel()
            X.append(vec)
            y.append(cls_id)
            groups.append(rep)  # group = triplet id
    return np.vstack(X), np.array(y), np.array(groups)

fname = os.path.join('features', 'mnist_features.pkl')
with open(fname, 'rb') as handle:
    features = pickle.load(handle)
    
ids = list(features.keys())

layer = "fc7"

act, trainCat, groups = stack_features(features, layer)
num_reps = len(np.unique(groups))
num_ids = len(ids)
num_samples = act.shape[0]

clf = pipeline.make_pipeline(
    preprocessing.StandardScaler(),
    decomposition.PCA(n_components=256, whiten=True, random_state=0),
    svm.LinearSVC(dual=True, C=0.25, tol=1e-3, max_iter=50000,
              class_weight='balanced', random_state=0)
)

logo = model_selection.LeaveOneGroupOut()
perf_fold = np.zeros(shape=(num_reps,))
y_true_all, y_pred_all = [], []

for fold_idx, (indTrain, indTest) in tqdm(enumerate(logo.split(act, trainCat, groups))):
    clf.fit(act[indTrain, :], trainCat[indTrain])
    dec = clf.predict(act[indTest, :])

    perf_fold[fold_idx] = metrics.accuracy_score(trainCat[indTest], dec)
    y_true_all.extend(trainCat[indTest])
    y_pred_all.extend(dec)

overall_acc = perf_fold.mean()
cm = metrics.confusion_matrix(y_true_all, y_pred_all, labels=[0,1,2])

print(f"Overall accuracy: {overall_acc:.3f}")
print("Per-fold accuracy (one triplet per fold):", perf_fold.shape, "folds")
print("Confusion matrix (rows=true, cols=pred):")
print(cm)
print("Label order:", class_names)

