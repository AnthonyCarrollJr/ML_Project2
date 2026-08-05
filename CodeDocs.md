# Code Doc for main.ipynb

## Use
This is the docs for `main.ipynb`, the Lung Sound Classification (HLS-CMDS) project. This will be used to explain functions, steps used, and other topics we may be questioned on as part of our project. Please add in any notes you see fit.

## Setup
Imports: `pathlib` for file globbing, `pandas` to read the label CSVs, `librosa`/`numpy` for audio loading and feature extraction, `sklearn` for the train/test split and evaluation metrics, `torch`/`skorch` for the CNN and its scikit-learn-compatible training loop.

`ROOT` points at the `HLS-CMDS/` dataset folder (gitignored because of the `.wav` files — see `README.md` for download/layout instructions). Audio is resampled to `SAMPLE_RATE=8000` Hz and every clip is fixed to `SECS=10.0` s so all spectrograms come out the same shape; `N_MELS`/`LENGTH_OF_FFT`/`HOP_LENGTH` control the mel-spectrogram resolution.

## Building the file index
Lung-labeled recordings come from two places: `LS/LS/*.wav` and the `L####.wav` files inside `Mix/Mix/` (the isolated lung component of each mixed recording, labeled by `Mix.csv`'s `Lung Sound ID`/`Lung Sound Type` columns). Combining both sources gives 195 samples across 6 balanced classes instead of just the 50 in `LS.csv` alone.

`LS.csv`'s own `Lung Sound ID` column doesn't reliably match the real filenames (it abbreviates Coarse Crackles as `G` and Fine Crackles as `C`, while the actual `.wav` files use `CC`/`FC`). So for `LS/LS/` the label is decoded directly from the filename's own code via a fixed `CODE2TYPE` table; `Mix.csv`'s numeric IDs don't have this problem and are looked up normally.

`CLASSES` is the sorted set of the 6 lung sound types, and `LABEL2IDX` maps each type name to the integer index used for training.
```python
def build_index():
    recs = [(wav, LABEL2IDX[CODE2TYPE[wav.stem.split("_")[1]]]) for wav in (ROOT/"LS"/"LS").glob("*.wav")]
    recs += [(wav, LABEL2IDX[mix_type[wav.stem]]) for wav in (ROOT/"Mix"/"Mix").glob("L*.wav")]
    return recs
```

## Feature extraction
`to_melspec` turns a `.wav` file into a normalized log-mel spectrogram:
1. Load audio mono at `SAMPLE_RATE` and center-crop or zero-pad it to exactly `SECS` seconds, so every clip yields a fixed-size input.
2. Compute a mel-spectrogram and convert power to decibels (`power_to_db`).
3. Standardize (zero mean, unit variance) so all clips are on a comparable scale.

`to_features` pools each spectrogram into per-mel-band mean and standard deviation — this is what feeds the **MLP**. The **CNN** instead consumes the full 2-D spectrogram from `to_melspec` directly, so it can learn local time-frequency patterns rather than pooled summary statistics.

## Dataset (PyTorch)
`LungSounds` is a thin `torch.utils.data.Dataset` wrapper: given a list of `(path, label)` pairs (from `build_index`), it lazily computes the log-mel spectrogram per item and returns it as a `(1, N_MELS, time)` float tensor plus its integer class label, ready for a `DataLoader`/CNN.

## Removing duplicate recordings
`Mix/Mix/L####.wav` isolates the lung component out of each heart+lung mixture, but the same lung recording gets reused across several different heart-sound pairings — 45 of those isolated components are byte-for-byte identical to a file that also lives in `LS/LS/` (verified directly on the waveforms: `corrcoef` = 1.0, zero max absolute difference). So the 195 `.wav` files reduce to only 94 acoustically unique lung sounds.

Deduplication is done by hashing each file's raw bytes with `md5` and keeping only the first occurrence of each hash:
```python
def audio_group_id(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()
```
The deduplicated `.wav` files and their labels are also copied out to `HLS-CMDS/Unique/` and `HLS-CMDS/unique_lung_sounds.csv`, so group mates can work from the clean 94-clip set directly instead of re-running the hashing step themselves.

## Pipeline
A pipeline chains a sequence of preprocessing steps and a final model into one object that behaves like a single estimator. Every step except the last must be a transformer (has `fit` and `transform`: scalers, PCA, feature selectors). The last step is the estimator (the classifier).
```python
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(max_iter=2000, early_stopping=False, random_state=RANDOM_STATE)),
])
```
This is used because preprocessing statistics are always computed from training data only — including inside every cross-validation fold — so there is no leakage between training and validation data.

## Multi-Layer Perceptron (scikit-learn)
`(features, label)` from `to_features` is split with a stratified `train_test_split`, holding out 20% as the test set. An `MLPClassifier` wrapped in the `Pipeline` above is tuned with `GridSearchCV`, using `StratifiedKFold` as its `cv`. The grid searches over hidden layer size, activation, L2 penalty (`alpha`), and initial learning rate:
```python
param_grid = {
    "mlp__hidden_layer_sizes": [(64,), (128,), (64, 32)],
    "mlp__activation": ["relu", "tanh", "logistic"],
    "mlp__alpha": [1e-4, 1e-3, 1e-2],
    "mlp__learning_rate_init": [1e-3, 1e-2],
}
```

## Stratify
Since some classification tasks can exhibit rare or imbalanced classes, plain random splitting can result in training or validation sets without any occurrence of a particular class, which can lead to errors during training or unreliable metrics. To mitigate this, `train_test_split(..., stratify=y)` and splitters such as `StratifiedKFold` implement stratified sampling to ensure that relative class frequencies are approximately preserved in each split/fold.

## GridSearchCV
`GridSearchCV` performs an exhaustive search over a grid of hyperparameter values, evaluating each combination with cross-validation on the training set only, then refits the best combination on the full training set.

__For every point on the grid, GridSearchCV:__
1. Splits the training data into `cv` folds.
2. For each fold: fits the estimator on the other folds, scores it on the held-out fold.
3. Averages the fold scores → that combination's `mean_test_score`.

__Attributes__
1. `best_params_` — dict of the winning hyperparameters
2. `best_score_` — mean CV score of the winner
3. `best_estimator_` — the refitted pipeline, ready to `.predict()` on the test set
4. `cv_results_` — full results table for every combination

## MLP Evaluation
Refit on the full training split happens automatically inside `GridSearchCV` (`refit=True` by default), so `grid.best_estimator_` is ready to score on the held-out test set. Per-class precision/recall/F1 (`classification_report`) and a labeled confusion matrix (`ConfusionMatrixDisplay.from_predictions`) are reported, then `cross_val_score` re-runs 5-fold stratified CV with the tuned hyperparameters over the entire dataset (`X`, `y`) as a sanity check on the grid-search score.

## Convolutional Neural Network (PyTorch)
`LungCNN` receives each spectrogram as a one-channel image-like tensor `(1, N_MELS, time)`. Three `Conv2d → BatchNorm2d → ReLU` blocks (with `MaxPool2d` after the first two) learn local patterns across frequency and time; `AdaptiveAvgPool2d((1, 1))` collapses the remaining spatial dimensions to a single value per channel regardless of input size, then a `Dropout` + `Linear` head produces the class logits.

## CNN Tuning with GridSearchCV (skorch)
`skorch.NeuralNetClassifier` wraps `LungCNN` in a scikit-learn-compatible estimator so it can be tuned with the same `GridSearchCV`/`StratifiedKFold` pattern used for the MLP, instead of the fixed-hyperparameter manual training loop further down in the notebook (kept commented out for reference rather than removed).

`module__*` params reach into `LungCNN.__init__` (e.g. `module__dropout`); the rest configure the skorch training loop itself (`lr`, `optimizer__weight_decay`, `batch_size`). `EarlyStopping(patience=10)` stops training a candidate once validation loss stops improving, and `ValidSplit(cv=0.2, stratified=True, ...)` carves out skorch's internal validation split used for that early stopping — separate from the outer `GridSearchCV` CV folds.
```python
cnn_net = NeuralNetClassifier(
    LungCNN,
    max_epochs=100,
    optimizer=torch.optim.Adam,
    criterion=nn.CrossEntropyLoss,
    callbacks=[("early_stop", EarlyStopping(patience=10))],
    device=cnn_device,
    train_split=ValidSplit(cv=0.2, stratified=True, random_state=RANDOM_STATE),
)

cnn_param_grid = {
    "lr": [1e-3, 1e-2],
    "optimizer__weight_decay": [1e-4, 1e-3],
    "module__dropout": [0.30, 0.50],
    "batch_size": [8, 16],
}
```
The CNN's own train/test split (`Xc_train`/`Xc_test`) uses full spectrograms from `to_melspec` (not the pooled `to_features` used by the MLP), and is a separate split from the MLP's `X_train`/`X_test`.

## CNN Evaluation (GridSearchCV)
`cnn_grid.best_estimator_` is refit on the full CNN training split, exactly like `grid.best_estimator_` for the MLP; score it on the held-out CNN test set (`Xc_test`/`yc_test`) with `classification_report` and a confusion matrix.

## Manual CNN Training Loop (legacy, commented out)
The commented-out cells (`Train/test split` through the final confusion matrix) are the original fixed-hyperparameter manual training loop, kept for reference rather than deleted now that the skorch/`GridSearchCV` version above does the same job with proper hyperparameter tuning:
- **Train/test split** — stratified split into `train_recs`/`test_recs`, wrapped in `DataLoader`s.
- **Training loop** — `CrossEntropyLoss` (appropriate for multi-class classification; it applies softmax internally) with `Adam`, plus manual early stopping tracked via `best_test_acc`/`epochs_without_improvement`.
- **Evaluation** — per-class precision/recall/F1, loss/accuracy curves over epochs, and a labeled confusion matrix.

## Model Comparison: MLP vs CNN
Both tuned models (`best_mlp` from `GridSearchCV` over the pooled features, `best_cnn` from `GridSearchCV` over the full spectrograms) are scored on their respective held-out test sets. `classification_report(..., output_dict=True)` gives accuracy plus macro-averaged precision/recall/F1 for each model, plotted as a grouped bar chart for a side-by-side comparison.

## Presentation Q&A
Questions an audience/professor is likely to ask, with a pointer to how to answer from the code.

**Dataset & preprocessing**
- Only 94 unique clips after dedup — isn't that too small to trust the results? → Yes, this is the biggest limitation; stratified 5-fold CV and stratified train/test splits are used specifically to get a stable estimate out of a small sample, but variance across folds/runs should be reported, not just one number.
- How did you find and handle the 45 duplicate recordings? → MD5 hash of the raw file bytes (`audio_group_id`), verified independently on the waveforms (`corrcoef` = 1.0, zero max abs difference) before trusting the hash match. See **Removing duplicate recordings**.
- Why fix every clip to `SECS=10.0` seconds instead of using variable length? → Fixed-size input is required for both the pooled MLP features and the CNN's spectrogram tensor shape; center-crop/zero-pad in `to_melspec` handles clips that are shorter/longer than 10s.
- Why 8000 Hz sample rate rather than the original recording rate? → Lung sounds carry diagnostic content well under 4000 Hz (Nyquist at 8000 Hz), so downsampling shrinks compute/feature size without discarding relevant frequency content — worth confirming this was a deliberate choice vs. a dataset default.
- Why standardize (z-score) the spectrogram instead of just using raw dB values? → Puts every clip on a comparable scale before pooling/feeding to the models, independent of overall recording loudness.
- `LS.csv`'s label column doesn't match filenames for Coarse/Fine Crackles — how do you know your `CODE2TYPE` decode is correct? → Be ready to explain the `G`/`C` vs `CC`/`FC` mismatch call-out in **Building the file index** and how it was verified against actual filenames on disk.

**Features & modeling choices**
- Why does the MLP use pooled mean/std features while the CNN uses the full spectrogram? → MLPs take flat feature vectors, so pooling per mel-band (mean, std) is a standard hand-engineered summary; the CNN can consume the full 2-D time-frequency image directly and learn its own local patterns, which is the point of comparing the two.
- What information is lost by pooling to mean/std for the MLP? → Temporal structure/ordering — e.g. a sound that changes character partway through a breath would look the same as one that's uniform throughout, since pooling collapses the time axis.
- Why `StandardScaler` inside a `Pipeline` instead of scaling `X` once up front? → So scaling statistics are fit only on each CV fold's training data — scaling on the full dataset first would leak test-set information into training. See **Pipeline**.
- Why stratify the splits at all given 6 classes? → With a small, class-balanced-by-design dataset, an unlucky random split could easily drop a class from train or test entirely; `stratify=y` / `StratifiedKFold` keeps class proportions consistent across every split. See **Stratify**.
- How were the `GridSearchCV` hyperparameter grids chosen (hidden layer sizes, CNN dropout, etc.)? → Be ready to say whether these were picked from prior experimentation, literature defaults, or compute-budget constraints — this is commonly asked and "trial and error within compute limits" is a fine honest answer.
- Why `n_jobs=1` for the CNN grid search but the MLP grid could use more? → Skorch/PyTorch models don't parallelize well across `GridSearchCV`'s `n_jobs` the way lightweight sklearn estimators do (GPU/CPU contention between folds), so it's run serially.
- What does `EarlyStopping(patience=10)` protect against, and how is it different from the outer `GridSearchCV` CV folds? → It stops an individual candidate's training once its internal validation loss stops improving (via `ValidSplit`), preventing overfitting *within* one hyperparameter combination — separate from the outer 5-fold CV that scores across combinations.
- Why keep the old manual PyTorch training loop commented out instead of deleting it? → Kept for reference/comparison against the tuned skorch/`GridSearchCV` version — shows the progression from fixed hyperparameters to a proper search.

**Evaluation & results**
- The MLP and CNN use different test-set sizes (`test_size=0.2` vs `0.25`) — is the comparison at the end fair? → Worth flagging as a limitation; each model is evaluated on its own held-out split rather than an identical shared split, so the final bar chart compares two different (though both stratified) test sets.
- Why macro-averaged precision/recall/F1 rather than weighted or micro? → Macro-average weights every class equally regardless of support, which matters here since misclassifying a rare/harder class (e.g. Pleural Rub) shouldn't be hidden by good performance on easier, more common classes.
- Which class is hardest to classify and why might that be? → Check the confusion matrices — classes with overlapping frequency signatures (e.g. Coarse vs Fine Crackles) are the most likely confusion pairs; be ready to point at the actual confusion matrix numbers from your last run.
- Does the CNN actually outperform the MLP here, and is the small dataset a factor? → CNNs typically need more data than 94 samples to fully exploit convolutional feature learning; if the CNN doesn't clearly beat the MLP, dataset size is the first thing to point to, not a flaw in the architecture.
- How would you validate this for real clinical use? → Would need a much larger, multi-site dataset, held-out patients (not just held-out clips, to avoid leakage between recordings from the same patient), and likely sensitivity/specificity per class rather than just accuracy.
