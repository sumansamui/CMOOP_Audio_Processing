import tensorflow as tf
import numpy as np
import random
import pandas as pd
import time
import os
from copy import deepcopy

# Scikit-learn imports for modeling and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, WhiteKernel
from sklearn.metrics import confusion_matrix

# Keras imports for building the CNN
from tensorflow.keras import layers, models, Model
from tensorflow.keras.callbacks import EarlyStopping

# --- GPU Configuration ---
# Ensure TensorFlow can see and use the GPU if available.
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU configuration successful.")
    else:
        print("No GPU detected. Running on CPU.")
except RuntimeError as e:
    print(e)

# --- Data Loading and Preparation ---
def load_data(data_path):
    print("Loading dataset...")
    X_train = np.load(f'{data_path}/X_train.npy')
    X_test = np.load(f'{data_path}/X_test.npy')
    X_validation = np.load(f'{data_path}/X_val.npy')
    y_train = np.load(f'{data_path}/y_train.npy')
    y_test = np.load(f'{data_path}/y_test.npy')
    y_validation = np.load(f'{data_path}/y_val.npy')
    y_train = y_train[..., np.newaxis]
    y_test = y_test[..., np.newaxis]
    y_validation = y_validation[..., np.newaxis]
    print("Dataset loaded!")
    return X_train, X_test, X_validation, y_train, y_test, y_validation

def prepare_dataset(data_path):
    X_train, X_test, X_validation, y_train, y_test, y_validation = load_data(data_path)
    scaler = StandardScaler()
    splits = [(X_train, 'X_train'), (X_test, 'X_test'), (X_validation, 'X_validation')]
    for i, (X_split, name) in enumerate(splits):
        num_instances, num_time_steps, num_features = X_split.shape
        X_reshaped = X_split.reshape(-1, num_features)
        if name == 'X_train':
            scaler.fit(X_reshaped)
        X_scaled = scaler.transform(X_reshaped)
        splits[i] = (X_scaled.reshape(num_instances, num_time_steps, num_features), name)
    X_train, _ = splits[0]
    X_test, _ = splits[1]
    X_validation, _ = splits[2]
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    return X_train, y_train, X_validation, y_validation, X_test, y_test

# --- Global Constants and Hyperparameters ---
DATA_PATH = "/home/ec.gpu/Desktop/Soumen/Dataset/kws/data_10_wav/data_npy_3000"
CLASS_NAMES = ['off', 'left', 'down', 'up', 'go', 'on', 'stop', 'unknown', 'right', 'yes']
CLASSES = 10
EPOCHS = 300
BATCH_SIZE = 64
PATIENCE = 5

# --- NSGA-II Search Space and Parameters ---
FILTER_OPTIONS = [16, 32, 64]
KERNEL_SIZE_OPTIONS = [3, 5]
USE_BN_OPTIONS = [True, False]
RESIDUAL_BLOCK_OPTIONS = [1, 2, 3]
FC_LAYER_OPTIONS = [1, 2, 3, 4]
USE_DROPOUT_OPTIONS = [True, False]
HPARAM_SPACE = {'filters': FILTER_OPTIONS, 'kernel_size': KERNEL_SIZE_OPTIONS, 'use_bn': USE_BN_OPTIONS, 
                'residual_blocks': RESIDUAL_BLOCK_OPTIONS, 'fc_layers': FC_LAYER_OPTIONS, 'use_dropout': USE_DROPOUT_OPTIONS}

POP_SIZE = 15
MAX_GEN = 30
INFILL_PERCENT = 0.334
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.2
EPSILON = 1e-6

MIN_ACCURACY = 0.90
MAX_MODEL_SIZE = 2.5
MAX_FPR = 0.09

LAMBDA_INITIAL = 1.0
LAMBDA_FINAL = 50.0

def get_lambda(gen):
    frac = gen / float(MAX_GEN - 1) if MAX_GEN > 1 else 1.0
    return LAMBDA_INITIAL + frac * (LAMBDA_FINAL - LAMBDA_INITIAL)

X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_dataset(DATA_PATH)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")


# --- CNN Model Builder & Evaluation ---
def build_model(hparams):
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    filters, kernel_size, use_bn, num_res_blocks, num_fc_layers, use_dropout = hparams.values()
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(inputs)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)
    for _ in range(num_res_blocks):
        res_filters = filters * 2
        skip = layers.Conv2D(res_filters, (1, 1), strides=(2, 2), padding='same')(x)
        y = layers.Conv2D(res_filters, kernel_size, padding='same', activation='relu')(x)
        if use_bn: y = layers.BatchNormalization()(y)
        y = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(y)
        x = layers.add([y, skip])
        x = layers.ReLU()(x)
        filters = res_filters
    x = layers.GlobalAveragePooling2D()(x)
    fc_layer_configs = {1: [64], 2: [128, 64], 3: [256, 128, 64], 4: [512, 256, 128, 64]}
    if num_fc_layers in fc_layer_configs:
        for units in fc_layer_configs[num_fc_layers]:
            x = layers.Dense(units, activation='relu')(x)
            if use_dropout: x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(CLASSES, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

def compute_model_size_mb(model):
    return model.count_params() * 4 / (1024**2)

def calculate_fpr(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fpr_vals = [ (np.sum(cm[:, i]) - cm[i, i]) / (np.sum(cm) - np.sum(cm[i, :])) for i in range(num_classes) if (np.sum(cm) - np.sum(cm[i, :])) > 0]
    return np.mean(fpr_vals) if fpr_vals else 0.0

def evaluate_individual(hparams):
    tf.keras.backend.clear_session()
    model = build_model(hparams)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), 
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping], verbose=0)
    _, accuracy = model.evaluate(X_validation, y_validation, verbose=0)
    y_pred = np.argmax(model.predict(X_validation, verbose=0), axis=1)
    y_true = y_validation.flatten() 
    fpr = calculate_fpr(y_true, y_pred, CLASSES)
    size_mb = compute_model_size_mb(model)
    print(f"  -> True Eval: Acc={accuracy:.4f}, Size={size_mb:.2f}MB, FPR={fpr:.4f}")
    return accuracy, size_mb, fpr

def compute_objectives_and_constraints(population_hparams):
    results = []
    for hparams in population_hparams:
        acc, size_mb, fpr = evaluate_individual(hparams)
        f1, f2, f3 = -acc, size_mb, fpr
        g1, g2, g3 = max(0.0, MIN_ACCURACY - acc), max(0.0, size_mb - MAX_MODEL_SIZE), max(0.0, fpr - MAX_FPR)
        CV = g1 + g2 + g3
        results.append({'hparams': hparams, 'objs': [f1, f2, f3], 'CV': CV})
    return results

# --- Surrogate Model Manager ---
class SurrogateManager:
    def __init__(self):
        self.is_fitted = False
        self.categorical_features = ['use_bn', 'use_dropout']
        self.numerical_features = ['filters', 'kernel_size', 'residual_blocks', 'fc_layers']
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', self.numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features)
            ], remainder='passthrough'
        )
        kernel = C(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1)
        self.models = {k: GaussianProcessRegressor(kernel=deepcopy(kernel), n_restarts_optimizer=10) for k in ['neg_acc', 'size', 'fpr', 'cv']}
        self.scalers = {k: StandardScaler() for k in self.models}
        self.training_data = pd.DataFrame()

    def _prepare_X_y(self, hparams_list, results_list):
        X_df = pd.DataFrame(hparams_list)
        # --- FIX: Start ---
        # Enforce a consistent boolean dtype for categorical features.
        # This prevents pandas from creating an 'object' dtype column due to
        # mixed types (e.g., True/False and 1/0), which causes the TypeError
        # in the OneHotEncoder.
        for col in self.categorical_features:
            if col in X_df.columns:
                X_df[col] = X_df[col].astype(bool)
        # --- FIX: End ---
        y_dict = {
            'neg_acc': np.array([r['objs'][0] for r in results_list]).reshape(-1, 1),
            'size': np.array([r['objs'][1] for r in results_list]).reshape(-1, 1),
            'fpr': np.array([r['objs'][2] for r in results_list]).reshape(-1, 1),
            'cv': np.array([r['CV'] for r in results_list]).reshape(-1, 1)
        }
        return X_df, y_dict

    def update(self, hparams_list, results_list):
        print(f"Updating surrogate models with {len(hparams_list)} new data points...")
        new_X_df, new_y_dict = self._prepare_X_y(hparams_list, results_list)
        new_data_df = new_X_df.copy()
        for k, v in new_y_dict.items(): new_data_df[f'y_{k}'] = v
        self.training_data = pd.concat([self.training_data, new_data_df]).drop_duplicates(
            subset=self.numerical_features + self.categorical_features, keep='last'
        ).reset_index(drop=True)
        X_train_df = self.training_data[self.numerical_features + self.categorical_features]
        X_transformed = self.preprocessor.fit_transform(X_train_df)
        for key in self.models.keys():
            y_train = self.training_data[f'y_{key}'].values.reshape(-1, 1)
            y_scaled = self.scalers[key].fit_transform(y_train)
            self.models[key].fit(X_transformed, y_scaled)
        self.is_fitted = True
        print("Surrogate models updated successfully.")

    def predict(self, hparams_list, return_std=False):
        if not self.is_fitted: raise RuntimeError("Surrogate models must be fitted.")
        X_df = pd.DataFrame(hparams_list)
        # --- FIX: Start ---
        # Also apply the dtype enforcement here for prediction consistency.
        for col in self.categorical_features:
            if col in X_df.columns:
                X_df[col] = X_df[col].astype(bool)
        # --- FIX: End ---
        X_transformed = self.preprocessor.transform(X_df)
        predictions = {}
        stds = {}
        for key, model in self.models.items():
            pred_scaled, std_scaled = model.predict(X_transformed, return_std=True)
            pred_scaled, std_scaled = pred_scaled.reshape(-1, 1), std_scaled.reshape(-1, 1)
            predictions[key] = self.scalers[key].inverse_transform(pred_scaled).flatten()
            stds[key] = std_scaled.flatten() * np.sqrt(self.scalers[key].var_) if self.scalers[key].var_ > 0 else np.zeros_like(std_scaled.flatten())
        return (predictions, stds) if return_std else predictions
    
    def predict_and_structure(self, hparams_list):
        preds, _ = self.predict(hparams_list, return_std=True)
        results = []
        for i, hparams in enumerate(hparams_list):
            results.append({
                'hparams': hparams,
                'objs': [preds['neg_acc'][i], preds['size'][i], preds['fpr'][i]],
                'CV': max(0, preds['cv'][i])
            })
        return results

# --- NSGA-II Core Helpers ---
# def initialize_population(pop_size):
#     return [ {k: random.choice(v) for k, v in HPARAM_SPACE.items()} for _ in range(pop_size) ]
###########################
initial_file = "/home/ec.gpu/Desktop/Soumen/nitd_0.4/nitd-0.4/kws/kws/c_moo/KWS/suroget_nsga/abil_study/Final.xlsx"
def initialize_population():
    df = pd.read_excel(initial_file)
    pop_data = []
    for _, r in df.iterrows():
        hp = {k:int(r[k]) if isinstance(r[k], (int, float)) else bool(r[k])
              for k in ['filters','kernel_size','use_bn','residual_blocks','fc_layers','use_dropout']}
        objs = [-r['Accuracy'], r['Size_MB'], r['FPR']]
        # compute CV from constraints
        g1 = max(0, MIN_ACCURACY - (-objs[0]))
        g2 = max(0, objs[1] - MAX_MODEL_SIZE)
        g3 = max(0, objs[2] - MAX_FPR)
        pop_data.append({'hparams':hp, 'objs':objs, 'CV':g1+g2+g3})
    print(f"Loaded {len(pop_data)} initial individuals from Excel.")
    return pop_data
#############################
def dominates(a, b, lam):
    Pa = [f + lam * a['CV'] for f in a['objs']]
    Pb = [f + lam * b['CV'] for f in b['objs']]
    better_all = all(pa <= pb for pa, pb in zip(Pa, Pb))
    strictly_better = any(pa < pb for pa, pb in zip(Pa, Pb))
    return better_all and strictly_better

def fast_non_dominated_sort(results, lam):
    fronts, S, n = [[]], [[] for _ in results], [0] * len(results)
    for p_idx, p_res in enumerate(results):
        for q_idx, q_res in enumerate(results):
            if p_idx == q_idx: continue
            if dominates(p_res, q_res, lam): S[p_idx].append(q_idx)
            elif dominates(q_res, p_res, lam): n[p_idx] += 1
        if n[p_idx] == 0: fronts[0].append(p_idx)
    i = 0
    while fronts[i]:
        next_front = []
        for p_idx in fronts[i]:
            for q_idx in S[p_idx]:
                n[q_idx] -= 1
                if n[q_idx] == 0: next_front.append(q_idx)
        i += 1
        fronts.append(next_front)
    return [f for f in fronts if f]

def crowding_distance(front, results):
    if not front: return {}
    distance = {idx: 0.0 for idx in front}
    num_objs = len(results[0]['objs'])
    for m in range(num_objs):
        front_sorted = sorted(front, key=lambda idx: results[idx]['objs'][m])
        distance[front_sorted[0]] = distance[front_sorted[-1]] = float('inf')
        f_min, f_max = results[front_sorted[0]]['objs'][m], results[front_sorted[-1]]['objs'][m]
        if f_max - f_min > EPSILON:
            for i in range(1, len(front_sorted) - 1):
                distance[front_sorted[i]] += (results[front_sorted[i+1]]['objs'][m] - results[front_sorted[i-1]]['objs'][m]) / (f_max - f_min)
    return distance

def tournament_selection(results, lam, k=2):
    idxs = random.sample(range(len(results)), k)
    best = idxs[0]
    for idx in idxs[1:]:
        if dominates(results[idx], results[best], lam):
            best = idx
    return best

def crossover(p1, p2):
    c1, c2 = deepcopy(p1), deepcopy(p2)
    for key in c1:
        if random.random() < 0.5:
            c1[key], c2[key] = c2[key], c1[key]
    return c1, c2

def mutate(hparams):
    ind = deepcopy(hparams)
    for key, options in HPARAM_SPACE.items():
        if random.random() < MUTATION_PROB:
            if isinstance(options[0], bool):
                ind[key] = not ind[key]
            else:
                ind[key] = random.choice(options)
    return ind
def select_infill_points(predicted_offspring_data, num_to_select):
    """
    Selects the best individuals from offspring to be evaluated by the true function.
    Selection criteria:
    1. Prefer predicted feasible solutions over infeasible ones.
    2. Rank feasible solutions by an equally-weighted sum of their normalized objectives.
    3. Rank infeasible solutions by their predicted constraint violation (CV).
    """
    feasible = []
    infeasible = []
    for i, res in enumerate(predicted_offspring_data):
        # Use a small tolerance for predicted feasibility
        if res['CV'] < EPSILON:
            feasible.append((i, res))
        else:
            infeasible.append((i, res))

    # --- Rank feasible solutions ---
    if feasible:
        # Normalize objectives to give them equal weight
        objectives = np.array([res['objs'] for _, res in feasible])
        min_objs = objectives.min(axis=0)
        max_objs = objectives.max(axis=0)
        range_objs = max_objs - min_objs
        range_objs[range_objs < EPSILON] = 1.0 # Avoid division by zero
        
        normalized_objs = (objectives - min_objs) / range_objs
        scores = normalized_objs.sum(axis=1)
        
        # Sort by the summed normalized score (lower is better)
        feasible_sorted_indices = [idx for idx, _ in sorted(zip([f[0] for f in feasible], scores), key=lambda pair: pair[1])]
    else:
        feasible_sorted_indices = []

    # --- Rank infeasible solutions by CV (lower is better) ---
    if infeasible:
        infeasible_sorted_indices = [idx for idx, _ in sorted(infeasible, key=lambda item: item[1]['CV'])]
    else:
        infeasible_sorted_indices = []

    # Combine lists and select the top N
    combined_indices = feasible_sorted_indices + infeasible_sorted_indices
    selected_indices = combined_indices[:num_to_select]
    
    selected_hparams = [predicted_offspring_data[i]['hparams'] for i in selected_indices]
    
    return selected_indices, selected_hparams
# --- NEW: Local Search Functionality ---
def perturb_hparams(hparams):
    """Slightly changes a single hyperparameter."""
    hparams_new = deepcopy(hparams)
    param_to_perturb = random.choice(list(HPARAM_SPACE.keys()))
    
    if isinstance(HPARAM_SPACE[param_to_perturb][0], bool):
        hparams_new[param_to_perturb] = not hparams_new[param_to_perturb]
    else:
        current_value = hparams_new[param_to_perturb]
        possible_values = [v for v in HPARAM_SPACE[param_to_perturb] if v != current_value]
        if possible_values:
            hparams_new[param_to_perturb] = random.choice(possible_values)
    return hparams_new

def lcb_dominates(sol_a, sol_b):
    """Dominance check based on Lower Confidence Bound (LCB)."""
    return all(a <= b for a, b in zip(sol_a['lcb_objs'], sol_b['lcb_objs'])) and \
           any(a < b for a, b in zip(sol_a['lcb_objs'], sol_b['lcb_objs']))

def perform_local_search(offspring_data, surrogate_manager, k_lcb=1.0):
    """
    Performs a local search on the best predicted offspring to refine them.
    
    Args:
        offspring_data (list of dicts): The offspring population with predicted objectives.
        surrogate_manager (SurrogateManager): The trained surrogate manager.
        k_lcb (float): The exploration factor for the LCB calculation.

    Returns:
        list of dicts: The potentially improved list of offspring hyperparameters.
    """
    print("Performing local search on predicted offspring...")
    # 1. Calculate LCB for all offspring
    for sol in offspring_data:
        means = np.array(sol['objs'])
        stds = np.array(sol['stds'])
        lcb_objs = means - k_lcb * stds
        sol['lcb_objs'] = lcb_objs.tolist()

    # 2. Find the non-dominated front based on LCB
    # This is a simplified sort: find the first front of an LCB-based dominance check
    elite_indices = []
    for i in range(len(offspring_data)):
        is_dominated_by_any = False
        for j in range(len(offspring_data)):
            if i == j: continue
            if lcb_dominates(offspring_data[j], offspring_data[i]):
                is_dominated_by_any = True
                break
        if not is_dominated_by_any:
            elite_indices.append(i)
    
    # 3. Perform 5 iterations of local search on the elite individuals
    for _ in range(5):
        for idx in elite_indices:
            original_sol = offspring_data[idx]
            
            # Perturb the hyperparameters to create a neighbor
            perturbed_hparams = perturb_hparams(original_sol['hparams'])
            
            # Predict the neighbor's performance
            pred_perturbed, std_perturbed = surrogate_manager.predict([perturbed_hparams], return_std=True)
            
            # Calculate LCB for the new solution
            lcb_perturbed = {k: pred_perturbed[k][0] - k_lcb * std_perturbed[k][0] for k in pred_perturbed}
            lcb_objs_perturbed = [lcb_perturbed['neg_acc'], lcb_perturbed['size'], lcb_perturbed['fpr']]

            # Create a temporary structure for comparison
            perturbed_sol_data = {'lcb_objs': lcb_objs_perturbed}

            # If the new perturbed solution is better (dominates the original), replace it
            if lcb_dominates(perturbed_sol_data, original_sol):
                # Update the original solution in the offspring_data list
                offspring_data[idx]['hparams'] = perturbed_hparams
                offspring_data[idx]['lcb_objs'] = lcb_objs_perturbed
                # Update mean and stds as well for consistency
                means = [pred_perturbed['neg_acc'][0], pred_perturbed['size'][0], pred_perturbed['fpr'][0]]
                stds_vals = [std_perturbed['neg_acc'][0], std_perturbed['size'][0], std_perturbed['fpr'][0]]
                offspring_data[idx]['objs'] = means
                offspring_data[idx]['stds'] = stds_vals

    # Return the hparams of the (potentially improved) full offspring population
    return [sol['hparams'] for sol in offspring_data]

# --- Main Surrogate-Assisted NSGA-II Loop ---
def nsga2(pop_size, max_gen, infill_percent):
    # 1. Initialize population & perform true evaluation
    print("--- Initializing Population ---")
    pop_data = initialize_population()
    print(f"Initial population size: {len(pop_data)}")
    
    surrogate_manager = SurrogateManager()
    surrogate_manager.update([d['hparams'] for d in pop_data], pop_data)
    
    all_gen_dfs = []
    total_start_time = time.perf_counter()

    for gen in range(max_gen):
        gen_start_time = time.perf_counter()
        lam = get_lambda(gen)
        print(f"\n--- Generation {gen}/{max_gen-1}, Lambda={lam:.2f} ---")

        tournament_start = time.perf_counter()
        parents_indices = [tournament_selection(pop_data, lam) for _ in range(pop_size)]
        tournament_end = time.perf_counter() - tournament_start
        print(f"Tournament {gen} finished in {tournament_end:.2f} seconds.")

        parent_hparams = [pop_data[i]['hparams'] for i in parents_indices]
        
        offspring_start = time.perf_counter()
        offspring_hparams = []
        while len(offspring_hparams) < pop_size:
            p1, p2 = random.sample(parent_hparams, 2)
            c1, c2 = crossover(p1, p2) if random.random() < CROSSOVER_PROB else (deepcopy(p1), deepcopy(p2))
            offspring_hparams.extend([mutate(c1), mutate(c2)])
        offspring_hparams = offspring_hparams[:pop_size]
        offspring_end = time.perf_counter() - offspring_start
        print(f"Offspring generation {gen} created in {offspring_end:.2f} seconds.")

        print(f"Evaluating {len(offspring_hparams)} offspring with surrogate models...")
        preds, stds = surrogate_manager.predict(offspring_hparams, return_std=True)
        
        off_data_predicted = []
        for i, hparams in enumerate(offspring_hparams):
            off_data_predicted.append({
                'hparams': hparams,
                'objs': [preds['neg_acc'][i], preds['size'][i], preds['fpr'][i]],
                'stds': [stds['neg_acc'][i], stds['size'][i], stds['fpr'][i]], # Storing stds
                'CV': max(0, preds['cv'][i])
            })
        
        # --- NEW LOCAL SEARCH STEP ---
        # This step refines the offspring based on surrogate predictions before true evaluation
        improved_offspring_hparams = perform_local_search(off_data_predicted, surrogate_manager)
        
        # After local search, we have a potentially better set of offspring hparams
        # We need to get their final predicted values for the infill selection process
        off_data_final_predicted = surrogate_manager.predict_and_structure(improved_offspring_hparams)
        
        # --- INFILL SELECTION ---
        num_infill = max(1, int(pop_size * infill_percent))
        print(f"Selecting {num_infill} infill points for true evaluation...")
        infill_indices, infill_hparams = select_infill_points(off_data_final_predicted, num_infill)
        
        infill_data_true = compute_objectives_and_constraints(infill_hparams)
        surrogate_manager.update(infill_hparams, infill_data_true)
        
        off_data = list(off_data_final_predicted)
        for i, true_res in enumerate(infill_data_true):
            original_index = infill_indices[i]
            off_data[original_index] = true_res

        # --- ENVIRONMENTAL SELECTION ---
        combined = pop_data + off_data
        combined_fronts = fast_non_dominated_sort(combined, lam)
        
        new_pop_data = []
        for front in combined_fronts:
            if len(new_pop_data) + len(front) <= pop_size:
                new_pop_data.extend([combined[i] for i in front])
            else:
                remaining = pop_size - len(new_pop_data)
                front_dist = crowding_distance(front, combined)
                sorted_front = sorted(front, key=lambda idx: front_dist.get(idx, 0), reverse=True)
                new_pop_data.extend([combined[i] for i in sorted_front[:remaining]])
                break
        
        pop_data = new_pop_data
        
        # --- Logging ---
        gen_time = time.perf_counter() - gen_start_time
        print(f"Generation {gen} finished in {gen_time:.2f} seconds.")
        print(f"Tournament {gen} finished in {tournament_end:.2f} seconds.")
        print(f"Offspring generation {gen} created in {offspring_end:.2f} seconds.")
        gen_records = []
        for ind in pop_data:
            rec = {'Generation': gen, 'Accuracy': -ind['objs'][0], 'Size_MB': ind['objs'][1], 
                   'FPR': ind['objs'][2], 'CV': ind['CV'], **ind['hparams']}
            gen_records.append(rec)
        all_gen_dfs.append(pd.DataFrame(gen_records))
        feasibles = [ind for ind in pop_data if ind['CV'] == 0]
        frac_feas = len(feasibles) / len(pop_data) if pop_data else 0
        avg_cv = np.mean([ind['CV'] for ind in pop_data]) if pop_data else 0
        print(f"  -> Status: Fraction Feasible={frac_feas:.2f}, Avg CV={avg_cv:.4f}")

    total_time = time.perf_counter() - total_start_time
    print(f"\n--- NSGA-II Finished in {total_time/60:.2f} minutes ---")
    print(f"Generation {gen} finished in {gen_time:.2f} seconds.")
    print(f"Tournament {gen} finished in {tournament_end:.2f} seconds.")
    print(f"Offspring generation {gen} created in {offspring_end:.2f} seconds.")
    
    final_feasibles = [ind for ind in pop_data if ind['CV'] == 0]
    if not final_feasibles:
        print("WARNING: No feasible solutions found in the final population.")
        return [], all_gen_dfs
    
    final_fronts = fast_non_dominated_sort(final_feasibles, LAMBDA_FINAL)
    if not final_fronts:
        print("WARNING: Could not determine a Pareto front from feasible solutions.")
        return [], all_gen_dfs
        
    pareto_indices = final_fronts[0]
    pareto_set = [final_feasibles[i] for i in pareto_indices]
    
    return pareto_set, all_gen_dfs

# --- Run Experiment and Save Results ---
if __name__ == "__main__":
    pareto_solutions, all_gen_dfs = nsga2(POP_SIZE, MAX_GEN, INFILL_PERCENT)
    output_dir = "2_stage_init_sa_nsga_local_results"
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, "all_generations_surrogate_ls.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        for i, df in enumerate(all_gen_dfs):
            df.to_excel(writer, sheet_name=f"Gen_{i}", index=False)
    print(f"\n✔️ All generation data saved to '{excel_path}'")

    print("\n--- Final Pareto-Optimal Feasible Solutions ---")
    if pareto_solutions:
        pareto_records = []
        for sol in pareto_solutions:
            acc, size, fpr = -sol['objs'][0], sol['objs'][1], sol['objs'][2]
            print(f"  - Acc={acc:.4f}, Size={size:.3f}MB, FPR={fpr:.4f}, HParams={sol['hparams']}")
            record = {'Accuracy': acc, 'Size_MB': size, 'FPR': fpr, **sol['hparams']}
            pareto_records.append(record)
        
        pareto_df = pd.DataFrame(pareto_records)
        csv_path = os.path.join(output_dir, "final_pareto_surrogate_ls.csv")
        pareto_df.to_csv(csv_path, index=False)
        print(f"✔️ Final Pareto front saved to '{csv_path}'")
    else:
        print("No Pareto-optimal solutions were found.")


