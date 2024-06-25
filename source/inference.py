from hashlib import sha256
import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.ensemble
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
import seaborn as sns
import torch
from tqdm.auto import tqdm
from torchgeo.datasets import QuakeSet

def create_features_vector(images):
    """
    Creates a feature vector from the input image (4, 512, 512).
    This vector is made of:
        - 50 pixel quantiles (*2%) on the 4 channels
        - 50 pixel quantiles (*2%) on the 2 differences between VV and VH channels
    resulting in 300 features.
    """
    r = np.arange(0.0, 1, 0.02)
    images = torch.flatten(images, start_dim=1).numpy().astype(np.float32)
    images_difference_VH = np.clip(images[3] - images[1], -0.05, 0.05)
    images_difference_VV = np.clip(images[2] - images[0], -0.75, 0.75)
    vector = np.concatenate(np.apply_along_axis(lambda x: np.quantile(x, r), 1, images), axis=0)
    vector = np.append(vector, [np.quantile(images_difference_VH, r), np.quantile(images_difference_VV, r)])
    return vector

def get_zero_differences_indices(dataset):
    """
    Returns the indices of the items in the dataset passed as parameter for which at least one of the VV or VH channels
    difference is all-zero.
    """
    zero_differences_indices = []
    for i in tqdm(range(len(dataset))):
        image = dataset[i]["image"].numpy()
        if not np.any(image[3]-image[1]) or not np.any(image[2]-image[0]):
            zero_differences_indices.append(i)
    return zero_differences_indices

def generate_submission_file(dataset, predicted_regression, flops, filename="submission.csv"):
    """
    Generates a submission file based on the prediction for the given parameters.
    """
    predictions = []
    for i in range(len(dataset)):
        metadata = dataset.data[i]
        key = metadata['key']
        predictions += [
            {"key": key, "magnitude": predicted_regression[i], "affected": int(predicted_regression[i]>0), "flops": flops}
        ]
    pd.DataFrame(predictions).to_csv(filename, index=False)

def main():
    test_set = QuakeSet(root="../private_set", split="test")
    
    m = np.load("means.npy")
    s = np.load("std.npy")
    
    test_features = []
    test_labels = []
    test_magnitudes = []
    for i in tqdm(range(len(test_set))):
        image = test_set[i]["image"]
        test_features.append(create_features_vector(image))
        test_labels.append(test_set[i]["label"].item())
        test_magnitudes.append(test_set[i]["magnitude"].item())
    
    test = pd.DataFrame(np.array(test_features, dtype=np.float32))
    test = (test-m)/s
    test["label"] = np.array(test_labels, dtype=np.float32)
    test["magnitude"] = np.array(test_magnitudes, dtype=np.float32)
    test.columns = test.columns.astype(str)
    
    classifier = lightgbm.Booster(model_file='classifier.txt')
    regressor = lightgbm.Booster(model_file='regressor.txt')
    
    test_zero_differences_indices = get_zero_differences_indices(test_set)
    
    test_classifications = classifier.predict(test.drop(["label", "magnitude"], axis=1)).astype(np.int8)
    test_classifications[test_zero_differences_indices] = 0
    
    test_regression_set = test.copy()
    test_regression_set["label"] = test_classifications
    test_regression = regressor.predict(test.drop(["magnitude"], axis=1))
    test_regression[test_zero_differences_indices] = 0
    test_regression[test_regression < 4] = 0
    test_regression[test_regression > 10] = 10
    
    classes = (test_regression>0).astype(np.int8)
    
    flops = 4*512*512+600
    
    generate_submission_file(test_set, test_regression, 4*512*512+600, "submission.csv")

if __name__ == "__main__":
    main()
