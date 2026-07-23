# Breast Cancer Classification using K-Nearest Neighbors (KNN)

## 🎯 Objective
Build a K-Nearest Neighbors (KNN) classification model to predict whether a breast tumor
is Malignant (M) or Benign (B) based on diagnostic measurements from cell-nuclei images.

## 📊 Dataset
**Breast Cancer Wisconsin Diagnostic Dataset**
Kaggle link: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

- 569 records, 33 columns
- Numerical features: 30 diagnostic measurements (mean, standard error, and "worst" values
  of 10 cell-nucleus characteristics — radius, texture, perimeter, area, smoothness, etc.)
- Target variable: `diagnosis` (M = Malignant, B = Benign)
- Dropped columns: `id` (non-predictive identifier) and `Unnamed: 32` (empty artifact
  column from the original Kaggle CSV export)

## 🛠 Libraries Used
- `pandas` – data loading and manipulation
- `numpy` – numerical operations
- `matplotlib` / `seaborn` – data visualization
- `scikit-learn` – train/test split, `StandardScaler`, `LabelEncoder`,
  `KNeighborsClassifier`, evaluation metrics

## 🧭 Methodology
1. **Data Understanding** – Loaded the dataset with Pandas, inspected the first five
   records, identified the 30 numerical features and the `diagnosis` target variable, and
   reviewed dataset info, summary statistics, and class balance (357 Benign / 212
   Malignant).
2. **Data Preprocessing**
   - Checked for missing values (found `Unnamed: 32` fully empty).
   - Removed `id` and `Unnamed: 32` columns.
   - Encoded `diagnosis` with `LabelEncoder` (B=0, M=1).
   - Split the data into 80% training and 20% testing sets (stratified, `random_state=42`).
   - Standardized all features with `StandardScaler`, fitting only on the training set to
     avoid data leakage.
3. **Model Development** – Trained a `KNeighborsClassifier` with **K = 5** on the scaled
   training data and predicted class labels for the test set.
4. **Model Evaluation** – Evaluated the model using Accuracy, Precision, Recall, and
   F1-Score, and visualized results with a Confusion Matrix. Also plotted accuracy across a
   range of K values (1–20) to contextualize the choice of K=5.

## 📈 Results

| Metric | Value |
|---|---|
| Accuracy  | ≈ 0.956 |
| Precision | ≈ 0.974 |
| Recall    | ≈ 0.905 |
| F1-Score  | ≈ 0.938 |

**Key observations:**
- The KNN model (K=5) achieves high accuracy (~96%) on the test set, confirming that the
  30 diagnostic measurements carry strong predictive signal.
- Feature scaling was essential — KNN is a distance-based algorithm, and without
  standardization, larger-scale features (like `area_mean`) would dominate the distance
  calculation and distort predictions.
- Precision is higher than recall for the malignant class, meaning the model is very
  reliable when it predicts malignancy, but still misses a small number of malignant
  cases (false negatives) — a critical consideration in a medical screening context, since
  a missed cancer diagnosis is generally more costly than a false alarm.

## ✅ Conclusion
This project used a K-Nearest Neighbors classifier to predict whether a breast tumor is
malignant or benign based on 30 diagnostic measurements derived from cell-nuclei images.
After removing non-predictive columns, encoding the target variable, standardizing the
features, and splitting the data 80/20, the KNN model (K=5) achieved strong accuracy,
precision, recall, and F1-scores, with the confusion matrix showing very few
misclassifications. **Feature scaling proved essential for KNN**, since the algorithm
classifies a new point based on the distance to its nearest neighbors — without
standardization, features measured on larger numeric scales (like area) would dominate the
distance calculation and distort predictions, regardless of their true diagnostic
importance. A key **limitation of KNN** is that it is a lazy learner with no explicit
training phase: it must compute distances to every training point at prediction time,
making it computationally expensive and slow for large datasets, and its performance is
also sensitive to the choice of K and to irrelevant or noisy features. Overall, KNN proved
to be a simple yet effective approach for this well-separated binary classification
problem.

## 📁 Repository Contents
- `Assignment-4.ipynb` – full notebook with code, outputs, and plots
- `README.md` – this file
