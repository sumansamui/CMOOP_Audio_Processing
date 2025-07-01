import tensorflow as tf
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import pandas as pd
import os
import time
from tensorflow.keras import layers
from tensorflow.keras import Model



# -------------------------------------
# GPU Configuration (optional)
# -------------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU configuration successful.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected.")

# -------------------------------------
# Data Loading & Preparation
# -------------------------------------
def load_data(data_path):
    """
    Expects .npy files under data_path:
      - X_train.npy, X_test.npy, X_val.npy
      - y_train.npy, y_test.npy, y_val.npy
    """
    X_train = np.load(f'{data_path}/X_train.npy')
    X_test = np.load(f'{data_path}/X_test.npy')
    X_val = np.load(f'{data_path}/X_val.npy')

    y_train = np.load(f'{data_path}/y_train.npy')
    y_test = np.load(f'{data_path}/y_test.npy')
    y_val = np.load(f'{data_path}/y_val.npy')

    # Add channel dimension for CNN
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    y_train = y_train[..., np.newaxis]
    y_test = y_test[..., np.newaxis]
    y_val = y_val[..., np.newaxis]

    print("Dataset loaded!")
    return X_train, X_test, X_val, y_train, y_test, y_val

def prepare_dataset(data_path):
    """
    Scales features across time–frequency axes via StandardScaler,
    then returns: X_train, y_train, X_val, y_val, X_test, y_test.
    """
    X_train, X_test, X_val, y_train, y_test, y_val = load_data(data_path)

    scaler = StandardScaler()

    # Flatten time–frequency dims for scaling
    n_train, t_steps, n_feats, _ = X_train.shape
    X_train_flat = X_train.reshape(-1, n_feats)
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_train = X_train_flat.reshape(n_train, t_steps, n_feats)[..., np.newaxis]

    n_test, t_steps, n_feats, _ = X_test.shape
    X_test_flat = X_test.reshape(-1, n_feats)
    X_test_flat = scaler.transform(X_test_flat)
    X_test = X_test_flat.reshape(n_test, t_steps, n_feats)[..., np.newaxis]

    n_val, t_steps, n_feats, _ = X_val.shape
    X_val_flat = X_val.reshape(-1, n_feats)
    X_val_flat = scaler.transform(X_val_flat)
    X_val = X_val_flat.reshape(n_val, t_steps, n_feats)[..., np.newaxis]

    return X_train, y_train, X_val, y_val, X_test, y_test

# -------------------------------------
# Paths & Global Variables
# -------------------------------------
DATA_PATH = "/home/22EC1102/soumen/data/KWS_10_log_mel_3000/data_npy_3000"
X_train, y_train, X_val, y_val, X_test, y_test = prepare_dataset(DATA_PATH)
print("Shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

CLASSES = 10
EPOCHS = 300
BATCH_SIZE = 64
PATIENCE = 5

# -------------------------------------
# Hyperparameter Search Space for CNN
# -------------------------------------
FILTER_OPTIONS          = [16, 32, 64]
KERNEL_SIZE_OPTIONS     = [3, 5]
USE_BN_OPTIONS          = [True, False]
RESIDUAL_BLOCK_OPTIONS  = [1, 2, 3]
FC_LAYER_OPTIONS        = [1, 2, 3, 4]
USE_DROPOUT_OPTIONS     = [True, False]

# -------------------------------------
# MOBO Settings
# -------------------------------------
# INITIAL_SAMPLES = 10 #10     # number of initial random hyperparameter sets
MAX_ITERATIONS  = 30 #20     # number of BO iterations
CANDIDATE_BATCH = 500 #500    # how many random candidates to score per iteration

# Constraint thresholds
MIN_ACCURACY   = 0.9    #0.90
MAX_MODEL_SIZE = 2.5    #2.5  # in MB
MAX_FPR        = 0.09   # 0.09 #0.01

# Penalty schedule (linear from λ_start to λ_end)
LAMBDA_START = 1.0
LAMBDA_END   = 50.0
def get_lambda_it(iter_idx):
    frac = iter_idx / float(MAX_ITERATIONS - 1)
    return LAMBDA_START + frac * (LAMBDA_END - LAMBDA_START)

# -------------------------------------
# CNN Builder & Evaluator
# -------------------------------------
def build_model(hparams):
    """
    Build a CNN with:
      - Initial Conv block (filters, kernel_size, use_bn)
      - X Residual blocks (each doubles filters & downsamples)
      - GlobalAveragePooling
      - fc_layers Dense layers + optional Dropout
    hparams keys:
      'filters', 'kernel_size', 'use_bn', 'residual_blocks', 'fc_layers', 'use_dropout'
    """
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    filters       = hparams['filters']
    kernel_size   = hparams['kernel_size']
    use_bn        = hparams['use_bn']
    num_res       = hparams['residual_blocks']
    num_fc        = hparams['fc_layers']
    use_dropout   = hparams['use_dropout']

    inputs = layers.Input(shape=input_shape)

    # Initial Conv Block
    x = layers.Conv2D(filters, kernel_size, padding='same')(inputs)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    # Residual Blocks
    for i in range(num_res):
        skip = layers.Conv2D(filters*2, (1,1), strides=(2,2), padding='same')(x)
        y = layers.Conv2D(filters*2, kernel_size, padding='same')(x)
        if use_bn: y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        y = layers.Conv2D(filters*2, kernel_size, padding='same')(y)
        if use_bn: y = layers.BatchNormalization()(y)
        y = layers.MaxPooling2D((2,2), strides=(2,2), padding='same')(y)
        x = layers.add([y, skip])
        x = layers.ReLU()(x)
        filters *= 2

    x = layers.GlobalAveragePooling2D()(x)

    # Dense Layers
    fc_configs = {
        1: [64],
        2: [128, 64],
        3: [256, 128, 64],
        4: [512, 256, 128, 64]
    }
    if num_fc in fc_configs:
        for idx, units in enumerate(fc_configs[num_fc]):
            x = layers.Dense(units, activation='relu')(x)
            if use_dropout:
                x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(CLASSES, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def compute_model_size_mb(model):
    """
    Returns model size in MB (float32 => 4 bytes per weight).
    """
    params = model.count_params()
    return (params * 4) / (1024 ** 2)

def calculate_fpr(y_true, y_pred, num_classes):
    """
    Macro-averaged False Positive Rate over 'num_classes'.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fpr_vals = []
    for i in range(num_classes):
        FP = np.sum(cm[:, i]) - cm[i, i]
        TN = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        if (FP + TN) > 0:
            fpr_vals.append(FP / (FP + TN))
        else:
            fpr_vals.append(0.0)
    return np.mean(fpr_vals)

def evaluate_individual(hparams):
    """
    Train the CNN briefly and return:
      - accuracy on X_val
      - model size in MB
      - macro FPR on X_val
    """
    tf.keras.backend.clear_session()
    model = build_model(hparams)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=0
    )
    val_acc = history.history['val_accuracy'][-1]
    print(f"Val accuracy: {val_acc:.4f}")

    y_pred_probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_val.flatten()
    fpr = calculate_fpr(y_true, y_pred, CLASSES)
    print(f"Macro FPR: {fpr:.4f}")

    size_mb = compute_model_size_mb(model)
    print(f"Model size: {size_mb:.2f} MB, FPR: {fpr:.4f}")
    return val_acc, size_mb, fpr

# -------------------------------------
# Surrogate (GP) Training & Acquisition
# -------------------------------------
def train_gps(X, Y):
    """
    Trains one separate GaussianProcessRegressor per column of Y.
    Returns list of trained GPR models.
    """
    models = []
    for dim in range(Y.shape[1]):
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        gp.fit(X, Y[:, dim])
        models.append(gp)
    return models

def predict_gps(models, X):
    """
    Returns an (n_samples, n_models) array of predicted means.
    """
    preds = []
    for gp in models:
        mu, _ = gp.predict(X, return_std=True)
        preds.append(mu)
    return np.stack(preds, axis=1)

def penalized_acquisition(x_candidates, obj_gps, cv_gp, lam):
    """
    For each candidate row in x_candidates:
      - Predict mean objectives via obj_gps (shape → (n, 3))
      - Predict mean CV via cv_gp (shape → (n,))
      - Form penalized vector: f_i + lam * CV
      - Return negative sum of penalized values (so we maximize this).
    """
    obj_mu = predict_gps(obj_gps, x_candidates)      # shape (n, 3)
    cv_mu = predict_gps([cv_gp], x_candidates)[:, 0]  # shape (n,)

    penalized = obj_mu + lam * cv_mu.reshape(-1, 1)
    return -np.sum(penalized, axis=1)  # maximize negative sum = minimize sum

# -------------------------------------
# Hyperparameter Encoding / Decoding
# -------------------------------------
def random_hparams():
    """
    Returns one random hyperparameter dictionary.
    """
    return {
        'filters':         random.choice(FILTER_OPTIONS),
        'kernel_size':     random.choice(KERNEL_SIZE_OPTIONS),
        'use_bn':          random.choice(USE_BN_OPTIONS),
        'residual_blocks': random.choice(RESIDUAL_BLOCK_OPTIONS),
        'fc_layers':       random.choice(FC_LAYER_OPTIONS),
        'use_dropout':     random.choice(USE_DROPOUT_OPTIONS)
    }

def hparams_to_vector(hp):
    """
    Encodes hyperparameters into a 6-dimensional numeric vector in [0,1]^6:
      indices / options → normalized continuous.
    Order: filters_idx, kernel_idx, use_bn_idx, res_blocks_idx, fc_layers_idx, use_dropout_idx
    """
    v = np.zeros(6)
    v[0] = FILTER_OPTIONS.index(hp['filters']) / (len(FILTER_OPTIONS) - 1)
    v[1] = KERNEL_SIZE_OPTIONS.index(hp['kernel_size']) / (len(KERNEL_SIZE_OPTIONS) - 1)
    v[2] = USE_BN_OPTIONS.index(hp['use_bn']) / (len(USE_BN_OPTIONS) - 1)
    v[3] = RESIDUAL_BLOCK_OPTIONS.index(hp['residual_blocks']) / (len(RESIDUAL_BLOCK_OPTIONS) - 1)
    v[4] = FC_LAYER_OPTIONS.index(hp['fc_layers']) / (len(FC_LAYER_OPTIONS) - 1)
    v[5] = USE_DROPOUT_OPTIONS.index(hp['use_dropout']) / (len(USE_DROPOUT_OPTIONS) - 1)
    return v

def vector_to_hparams(vec):
    """
    Decodes a 6-dim vector ∈ [0,1]^6 back to a hyperparameter dictionary.
    """
    hp = {}
    idx0 = int(round(vec[0] * (len(FILTER_OPTIONS) - 1)))
    idx1 = int(round(vec[1] * (len(KERNEL_SIZE_OPTIONS) - 1)))
    idx2 = int(round(vec[2] * (len(USE_BN_OPTIONS) - 1)))
    idx3 = int(round(vec[3] * (len(RESIDUAL_BLOCK_OPTIONS) - 1)))
    idx4 = int(round(vec[4] * (len(FC_LAYER_OPTIONS) - 1)))
    idx5 = int(round(vec[5] * (len(USE_DROPOUT_OPTIONS) - 1)))

    hp['filters']         = FILTER_OPTIONS[idx0]
    hp['kernel_size']     = KERNEL_SIZE_OPTIONS[idx1]
    hp['use_bn']          = USE_BN_OPTIONS[idx2]
    hp['residual_blocks'] = RESIDUAL_BLOCK_OPTIONS[idx3]
    hp['fc_layers']       = FC_LAYER_OPTIONS[idx4]
    hp['use_dropout']     = USE_DROPOUT_OPTIONS[idx5]
    return hp

# -------------------------------------
# Main MOBO Loop
# -------------------------------------
def run_mobo():
    """
    1.  Loads initial hyperparameter sets and their results from an Excel file.
    2.  Computes objectives=[-acc, size, fpr] and the constraint violation (CV) for the initial data.
    3.  Fits Gaussian Process (GP) models for each objective and for the CV.
    4.  For a maximum number of ITERATIONS:
        a. Compute the current penalty parameter λ.
        b. Generate a batch of random candidate vectors.
        c. Evaluate the acquisition function on candidates and select the best one (x_next).
        d. Evaluate the selected x_next, get its performance, and append it to the dataset.
        e. Re-fit the GP models with the newly augmented data in the next loop.
    5.  Saves per-iteration data and returns the final calculated Pareto front.
    """
    dim = 6  # hyperparameter vector dim
    # 1. Initial random samples
    # --------------------------------------------------------------------------
    # MODIFIED SECTION: Load initial samples from the provided Excel file
    # --------------------------------------------------------------------------
    # Define the file path in a variable for clarity and easy modification
    initial_data_filepath = "/home/22EC1102/soumen/NIT-GPU_2/code/c_moo/2_stage/Final.xlsx"
    print(f"Loading initial samples from '{initial_data_filepath}'...")
    try:
        df_initial = pd.read_excel(initial_data_filepath)   #   pd.read_csv()
    except FileNotFoundError:
        print(f"Error: The file '{initial_data_filepath}' was not found.")
        return [], [], 0.0

    INITIAL_SAMPLES = len(df_initial)
    print(f"Found {INITIAL_SAMPLES} initial samples.")

    # 1. Prepare Y_objs and Y_cv from the file's objective columns
    acc_init = df_initial['Accuracy'].values
    size_init = df_initial['Size_MB'].values
    fpr_init = df_initial['FPR'].values

    # Objectives to be minimized: [-accuracy, size, fpr]
    Y_objs = np.vstack([-acc_init, size_init, fpr_init]).T
    print(f"Initial objectives shape: {Y_objs.shape}")

    # Calculate Constraint Violation (CV) for each initial point
    cv_init = (np.maximum(0.0, MIN_ACCURACY - acc_init) +
               np.maximum(0.0, size_init - MAX_MODEL_SIZE) +
               np.maximum(0.0, fpr_init - MAX_FPR))
    print(f"Initial CV shape: {cv_init.shape}")
    Y_cv = cv_init.reshape(-1, 1)
    print(f"Initial CV values: {Y_cv.flatten()}")

    # 2. Prepare X_vec and all_hparams from the file's hyperparameter columns
    # Assumes the remaining columns are the hyperparameters
    hp_cols = [col for col in df_initial.columns if col not in ['Accuracy', 'Size_MB', 'FPR']]
    df_hps = df_initial[hp_cols]
    
    all_hparams = [row.to_dict() for _, row in df_hps.iterrows()]
    print(f"Initial hyperparameters shape: {len(all_hparams)} samples, {len(all_hparams[0])} hparams")
    
    # Convert hyperparameter dictionaries to normalized vectors for the GPs
    X_vec = np.array([hparams_to_vector(hp) for hp in all_hparams])
    print(f"Initial hyperparameter vectors shape: {X_vec.shape}")
    # --------------------------------------------------------------------------
    # END OF MODIFIED SECTION
    # --------------------------------------------------------------------------
    gen_dfs = []
    # all_hparams = [vector_to_hparams(X_vec[i]) for i in range(INITIAL_SAMPLES)]

    for it in range(MAX_ITERATIONS):
        print(f"\n----- MOBO Iteration {it+1}/{MAX_ITERATIONS} -----")
        lam = get_lambda_it(it)
        print(f"Current penalty λ = {lam:.3f}")
        start_initial_eval = time.perf_counter()
        # 3. Fit GPs
        gp_objs = train_gps(X_vec, Y_objs)
        gp_cv   = train_gps(X_vec, Y_cv)[0]

        # 4a. Generate random candidate vectors in [0,1]^6
        candidates = np.random.rand(CANDIDATE_BATCH, dim)

        # 4b. Compute acquisition for each candidate
        acq_vals = penalized_acquisition(candidates, gp_objs, gp_cv, lam)

        # 4c. Pick best candidate
        best_idx = np.argmax(acq_vals)
        x_next = candidates[best_idx]
        hp_next = vector_to_hparams(x_next)

        # 4d. Evaluate the chosen hyperparameters
        print(f"Sampling new hparams: {hp_next}")
        acc_n, size_n, fpr_n = evaluate_individual(hp_next)
        cv_n = max(0.0, MIN_ACCURACY - acc_n) + max(0.0, size_n - MAX_MODEL_SIZE) + max(0.0, fpr_n - MAX_FPR)

        # Append new data
        X_vec = np.vstack([X_vec, x_next.reshape(1, -1)])
        Y_objs = np.vstack([Y_objs, [-acc_n, size_n, fpr_n]])
        Y_cv   = np.vstack([Y_cv, [[cv_n]]])
        all_hparams.append(hp_next)

        # 4e. (Re-fitting GPs happens at top of loop next iteration)

        # Save per-iteration data
        records = []
        for j in range(X_vec.shape[0]):
            hp_j = all_hparams[j]
            f1, f2, f3 = Y_objs[j]
            cvj = Y_cv[j, 0]
            rec = {
                'Iteration': j if j < INITIAL_SAMPLES else f"init+{j-INITIAL_SAMPLES+1}",
                'Accuracy': -f1,
                'Size_MB': f2,
                'FPR': f3,
                'CV': cvj,
                **hp_j
            }
            records.append(rec)

        df_iter = pd.DataFrame(records)
        gen_dfs.append(df_iter)

        initial_eval_time = time.perf_counter() - start_initial_eval
        print(f"One iteration time {initial_eval_time:.4f} seconds")

    # 5. After all iterations, extract feasible Pareto front
    feasible_idxs = [i for i in range(Y_cv.shape[0]) if Y_cv[i, 0] <= 1e-8]
    if not feasible_idxs:
        print("No feasible solutions found.")
        return [], gen_dfs, initial_eval_time

    feasible_objs = Y_objs[feasible_idxs]
    # Pareto filter: non-dominated among feasible
    pareto_mask = np.ones(len(feasible_idxs), dtype=bool)
    for i in range(len(feasible_idxs)):
        for j in range(len(feasible_idxs)):
            if i == j: continue
            if all(feasible_objs[j] <= feasible_objs[i]) and any(feasible_objs[j] < feasible_objs[i]):
                pareto_mask[i] = False
                break
    pareto_idxs = np.array(feasible_idxs)[pareto_mask]
    pareto_solutions = [(all_hparams[i], Y_objs[i], Y_cv[i,0]) for i in pareto_idxs]
    return pareto_solutions, gen_dfs , initial_eval_time

# -------------------------------------
# Execute MOBO & Save Results
# -------------------------------------
pareto_solutions, gen_dfs, initial_eval_time = run_mobo()
print(f"One iteration time {initial_eval_time:.4f} seconds")

# Save per-iteration dataframes into one Excel (multiple sheets)
excel_path = "/home/22EC1102/soumen/NIT-GPU_2/code/c_moo/2_stage/mobo_iterations.xlsx"   # data
# Combine all generation DataFrames into one
all_data = pd.concat(gen_dfs, ignore_index=True)
all_data.to_excel(excel_path, index=False) #, sheet_name="All_Generations"
# with pd.ExcelWriter(excel_path) as writer:
#     for idx, df in enumerate(gen_dfs):
#         df.to_excel(writer, sheet_name=f"Iter_{idx}", index=False)
print(f"✔️ Iteration data saved to '{excel_path}'")

# Print final Pareto set
print("\nFinal Pareto-Optimal Feasible Solutions:")
for hp, obj, cv in pareto_solutions:
    acc = -obj[0]
    size = obj[1]
    fpr = obj[2]
    print(f"Acc={acc:.3f}, Size={size:.3f}MB, FPR={fpr:.4f}, CV={cv:.4f}, HP={hp}")

# Save final Pareto set to CSV
pareto_records = []
for hp, obj, cv in pareto_solutions:
    acc = -obj[0]
    size = obj[1]
    fpr = obj[2]
    rec = {
        'Accuracy': acc,
        'Size_MB':  size,
        'FPR':      fpr,
        'CV':       cv,
        **hp
    }
    pareto_records.append(rec)
df_pareto = pd.DataFrame(pareto_records)
csv_path = "/home/22EC1102/soumen/NIT-GPU_2/code/c_moo/2_stage/mobo_pareto.csv"  # data
df_pareto.to_csv(csv_path, index=False)
print(f"✔️ Final Pareto set saved to '{csv_path}'")
