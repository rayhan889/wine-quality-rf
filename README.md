# Supervised Random Forest

Random forest classifier that predicts wine quality class from physicochemical properties. The whole pipeline lives in `main.ipynb`.

## Data

Red and white wine quality datasets in `data/`. After removing duplicates, 5320 rows remain (1359 red, 3961 white).

Quality is grouped into three classes:

| Class | Rule | Share |
|---|---|---|
| low | quality <= 5 | 37.4% |
| medium | quality = 6 | 43.7% |
| high | quality >= 7 | 19.0% |

## Pipeline

1. Cleaning and labeling: merge red and white, drop duplicates, build labels, run sanity checks
2. Preprocessing: 11 physicochemical features plus an `is_red` flag, stratified 80/20 split
3. Feature engineering: `total_acidity` and `free_so2_ratio` were tested and dropped because they did not improve CV macro-F1
4. Training: 5-fold stratified CV with macro-F1, compared against a dummy baseline and a default forest, then `RandomizedSearchCV` (30 iterations)
5. Evaluation: test metrics, confusion matrix, permutation importance
6. Inference: model saved to `rf_wine_quality.joblib` and reused through `predict_wine()`

Each step ends with assertion checkpoints that fail the run on leakage, bad splits, unstable scores or a model that does not reproduce after reload.

## Results

| Model | CV macro-F1 |
|---|---|
| Dummy (most frequent) | 0.203 |
| Default random forest | 0.605 |
| Tuned random forest | 0.614 |

Best params: `n_estimators=200`, `min_samples_leaf=2`, `max_features=sqrt`, `max_depth=None`, `class_weight=balanced`.

Test set: accuracy 0.586, macro-F1 0.587.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| low | 0.638 | 0.705 | 0.670 |
| medium | 0.559 | 0.447 | 0.497 |
| high | 0.534 | 0.668 | 0.593 |

Alcohol is by far the most important feature, followed by volatile acidity and free sulfur dioxide. The medium class is the hardest to separate. Training score is near 0.98, so the forest overfits somewhat.

## Setup

Requires Python 3.14 and uv.

```
uv sync
uv run jupyter lab main.ipynb
```

Run all cells to retrain and regenerate `rf_wine_quality.joblib`.

## Inference

```python
predict_wine({
    "fixed acidity": 7.4, "volatile acidity": 0.7, "citric acid": 0.0,
    "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11,
    "total sulfur dioxide": 34, "density": 0.9978, "pH": 3.51,
    "sulphates": 0.56, "alcohol": 9.4, "wine_type": "red",
})
```

Returns the predicted label and class probabilities.

## Files

| File | Purpose |
|---|---|
| `main.ipynb` | Full pipeline |
| `data/` | Raw red and white wine CSVs |
| `rf_wine_quality.joblib` | Saved model bundle (model, features, labels) |
| `pyproject.toml`, `uv.lock` | Dependencies |
