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
    """
    Loads pre-split training, validation, and test dataset from .npy files.
    :param data_path (str): Path to the directory containing .npy files.
    :return: Tuple of X_train, X_test, X_validation, y_train, y_test, y_validation
    """
    print("Loading dataset...")
    X_train = np.load(f'{data_path}/X_train.npy')
    X_test = np.load(f'{data_path}/X_test.npy')
    X_validation = np.load(f'{data_path}/X_val.npy')

    y_train = np.load(f'{data_path}/y_train.npy')
    y_test = np.load(f'{data_path}/y_test.npy')
    y_validation = np.load(f'{data_path}/y_val.npy')

    # Add a dimension for the integer labels to be compatible with sparse loss
    y_train = y_train[..., np.newaxis]
    y_test = y_test[..., np.newaxis]
    y_validation = y_validation[..., np.newaxis]
    
    print("Dataset loaded!")
    return X_train, X_test, X_validation, y_train, y_test, y_validation

def prepare_dataset(data_path):
    """
    Loads data, scales features, and reshapes it for the CNN.
    :param data_path (str): Path to the data directory.
    :return: Tuple of processed train, validation, and test sets.
    """
    X_train, X_test, X_validation, y_train, y_test, y_validation = load_data(data_path)
    
    # --- Scaling the features (MFCCs) ---
    scaler = StandardScaler()
    
    # Reshape, scale, and reshape back for each dataset split
    # Create a list of tuples to handle variable names correctly
    splits = [(X_train, 'X_train'), (X_test, 'X_test'), (X_validation, 'X_validation')]
    for i, (X_split, name) in enumerate(splits):
        num_instances, num_time_steps, num_features = X_split.shape
        # Reshape to (num_samples * timesteps, features) for scaling
        X_reshaped = X_split.reshape(-1, num_features)
        # Fit scaler only on training data
        if name == 'X_train':
            scaler.fit(X_reshaped)
        X_scaled = scaler.transform(X_reshaped)
        # Reshape back to (num_samples, timesteps, features) and update the original array
        splits[i] = (X_scaled.reshape(num_instances, num_time_steps, num_features), name)

    X_train, _ = splits[0]
    X_test, _ = splits[1]
    X_validation, _ = splits[2]

    # Add a channel dimension for the CNN (required for Conv2D)
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
# Hyperparameter search space for the CNN architecture
FILTER_OPTIONS = [16, 32, 64]
KERNEL_SIZE_OPTIONS = [3, 5]
USE_BN_OPTIONS = [True, False]
RESIDUAL_BLOCK_OPTIONS = [1, 2, 3]
FC_LAYER_OPTIONS = [1, 2, 3, 4]
USE_DROPOUT_OPTIONS = [True, False]

# NSGA-II parameters
POP_SIZE = 15
MAX_GEN =  30
INFILL_PERCENT = 0.334  # 0.2     # Percentage of offspring to evaluate with the true function each generation
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.2
EPSILON = 1e-6  # Small value for crowding distance ties

# Constraint thresholds
MIN_ACCURACY = 0.90
MAX_MODEL_SIZE = 2.5  # in MB
MAX_FPR = 0.09 # 0.01

# Adaptive penalty schedule
LAMBDA_INITIAL = 1.0
LAMBDA_FINAL = 50.0

def get_lambda(gen):
    frac = gen / float(MAX_GEN - 1) if MAX_GEN > 1 else 1.0
    return LAMBDA_INITIAL + frac * (LAMBDA_FINAL - LAMBDA_INITIAL)


# --- Load and Prepare Data ---
X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_dataset(DATA_PATH)
print("Data shapes:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_validation: {X_validation.shape}, y_validation: {y_validation.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")


# --- CNN Model Builder ---

def build_model(hparams):
    """
    Builds a custom residual CNN based on a dictionary of hyperparameters.
    """
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    
    filters = hparams['filters']
    kernel_size = hparams['kernel_size']
    use_bn = hparams['use_bn']
    num_res_blocks = hparams['residual_blocks']
    num_fc_layers = hparams['fc_layers']
    use_dropout = hparams['use_dropout']
    
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
    model = Model(inputs=inputs, outputs=outputs)
    return model

# --- Objective and Constraint Calculation ---

def compute_model_size_mb(model):
    """Calculates the model's size in megabytes."""
    return model.count_params() * 4 / (1024**2)

def calculate_fpr(y_true, y_pred, num_classes):
    """Calculates the macro-averaged False Positive Rate."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fpr_vals = []
    for i in range(num_classes):
        FP = np.sum(cm[:, i]) - cm[i, i]
        TN = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fpr_vals.append(FP / (FP + TN) if (FP + TN) > 0 else 0.0)
    return np.mean(fpr_vals)

def evaluate_individual(hparams):
    """
    The "true" expensive evaluation function. Builds, trains, and evaluates a model.
    :param hparams (dict): A dictionary of hyperparameters.
    :return: A tuple of (accuracy, size_mb, fpr).
    """
    tf.keras.backend.clear_session()
    model = build_model(hparams)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), 
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping], verbose=0)
    
    loss, accuracy = model.evaluate(X_validation, y_validation, verbose=0)
    
    y_pred = np.argmax(model.predict(X_validation, verbose=0), axis=1)
    # y_validation is (samples, 1), flatten it to a 1D array for metrics
    y_true = y_validation.flatten() 
    
    fpr = calculate_fpr(y_true, y_pred, CLASSES)
    size_mb = compute_model_size_mb(model)
    
    print(f"  -> True Eval: Acc={accuracy:.4f}, Size={size_mb:.2f}MB, FPR={fpr:.4f}")
    return accuracy, size_mb, fpr

def compute_objectives_and_constraints(population_hparams):
    """
    Wrapper to evaluate a list of hyperparameter sets and structure the results.
    :param population_hparams (list of dicts): The individuals to evaluate.
    :return: A list of dictionaries, each containing hparams, objectives, and CV.
    """
    results = []
    for hparams in population_hparams:
        acc, size_mb, fpr = evaluate_individual(hparams)
        
        # Objectives (we minimize all of them)
        f1 = -acc      # Minimize negative accuracy
        f2 = size_mb   # Minimize size
        f3 = fpr       # Minimize FPR
        
        # Constraint violations
        g1 = max(0.0, MIN_ACCURACY - acc)
        g2 = max(0.0, size_mb - MAX_MODEL_SIZE)
        g3 = max(0.0, fpr - MAX_FPR)
        CV = g1 + g2 + g3
        
        results.append({'hparams': hparams, 'objs': [f1, f2, f3], 'CV': CV})
    return results


# --- Surrogate Model Manager ---

class SurrogateManager:
    """
    Manages the Gaussian Process surrogate models for each objective and the CV.
    Handles data transformation, model training, and prediction.
    """
    def __init__(self):
        self.is_fitted = False
        # Define categorical and numerical features from the hparam dict
        self.categorical_features = ['use_bn', 'use_dropout']
        self.numerical_features = ['filters', 'kernel_size', 'residual_blocks', 'fc_layers']
        
        # Create a preprocessor to one-hot encode categorical features
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', self.numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features)
            ], remainder='passthrough'
        )
        
        # Define a standard kernel for the GP models
        kernel = C(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1)

        # Initialize four GP models, one for each target
        self.models = {
            'neg_acc': GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10),
            'size': GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10),
            'fpr': GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10),
            'cv': GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        }
        
        # Scalers for the target variables (objectives) to improve GP performance
        self.scalers = {
            'neg_acc': StandardScaler(),
            'size': StandardScaler(),
            'fpr': StandardScaler(),
            'cv': StandardScaler()
        }
        self.training_data = pd.DataFrame()


    def _prepare_X_y(self, hparams_list, results_list):
        """Prepares X (from hparams) and y (from results) for GP training."""
        X_df = pd.DataFrame(hparams_list)
        
        y_neg_acc = np.array([res['objs'][0] for res in results_list]).reshape(-1, 1)
        y_size = np.array([res['objs'][1] for res in results_list]).reshape(-1, 1)
        y_fpr = np.array([res['objs'][2] for res in results_list]).reshape(-1, 1)
        y_cv = np.array([res['CV'] for res in results_list]).reshape(-1, 1)
        
        y_dict = {'neg_acc': y_neg_acc, 'size': y_size, 'fpr': y_fpr, 'cv': y_cv}
        
        return X_df, y_dict

    def update(self, hparams_list, results_list):
        """Trains or re-trains the GP models on the given data."""
        print(f"Updating surrogate models with {len(hparams_list)} new data points...")
        
        new_X_df, new_y_dict = self._prepare_X_y(hparams_list, results_list)
        
        # Combine hparams with results for the training data dataframe
        new_data_df = new_X_df.copy()
        new_data_df['y_neg_acc'] = new_y_dict['neg_acc']
        new_data_df['y_size'] = new_y_dict['size']
        new_data_df['y_fpr'] = new_y_dict['fpr']
        new_data_df['y_cv'] = new_y_dict['cv']

        # Append new data and remove duplicates, keeping the most recent evaluation
        self.training_data = pd.concat([self.training_data, new_data_df]).drop_duplicates(
            subset=self.numerical_features + self.categorical_features, keep='last'
        )
        
        X_train_df = self.training_data[self.numerical_features + self.categorical_features]
        
        # Transform hyperparameters into a numerical matrix
        X_transformed = self.preprocessor.fit_transform(X_train_df)
        
        for key in self.models.keys():
            y_train = self.training_data[f'y_{key}'].values.reshape(-1, 1)
            y_scaled = self.scalers[key].fit_transform(y_train)
            self.models[key].fit(X_transformed, y_scaled)
        
        self.is_fitted = True
        print("Surrogate models updated successfully.")

    def predict(self, hparams_list):
        """Predicts objectives and CV for new hyperparameters using the GPs."""
        if not self.is_fitted:
            raise RuntimeError("Surrogate models must be fitted before prediction.")
            
        X_df = pd.DataFrame(hparams_list)
        X_transformed = self.preprocessor.transform(X_df)
        
        predictions = {}
        for key, model in self.models.items():
            pred_scaled = model.predict(X_transformed).reshape(-1, 1)
            predictions[key] = self.scalers[key].inverse_transform(pred_scaled).flatten()
            
        # Structure the predictions into the same format as true evaluations
        predicted_results = []
        for i, hparams in enumerate(hparams_list):
            predicted_results.append({
                'hparams': hparams,
                'objs': [predictions['neg_acc'][i], predictions['size'][i], predictions['fpr'][i]],
                'CV': max(0, predictions['cv'][i]) # Ensure predicted CV is non-negative
            })
        return predicted_results

# --- NSGA-II Core Algorithm Helpers ---
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


def dominates(a, b, lam):
    """Penalized dominance check."""
    # Add penalty term to objectives
    Pa = [f + lam * a['CV'] for f in a['objs']]
    Pb = [f + lam * b['CV'] for f in b['objs']]
    
    better_all = all(pa <= pb for pa, pb in zip(Pa, Pb))
    strictly_better = any(pa < pb for pa, pb in zip(Pa, Pb))
    return better_all and strictly_better

def fast_non_dominated_sort(results, lam):
    """Sorts the population into Pareto fronts using penalized dominance."""
    fronts = [[]]
    S = [[] for _ in results]
    n = [0] * len(results)
    
    for p_idx, p_res in enumerate(results):
        for q_idx, q_res in enumerate(results):
            if p_idx == q_idx: continue
            if dominates(p_res, q_res, lam):
                S[p_idx].append(q_idx)
            elif dominates(q_res, p_res, lam):
                n[p_idx] += 1
        if n[p_idx] == 0:
            fronts[0].append(p_idx)
    
    i = 0
    while fronts[i]:
        next_front = []
        for p_idx in fronts[i]:
            for q_idx in S[p_idx]:
                n[q_idx] -= 1
                if n[q_idx] == 0:
                    next_front.append(q_idx)
        i += 1
        fronts.append(next_front)
        
    return [f for f in fronts if f] # Return only non-empty fronts

def crowding_distance(front, results):
    """Calculates the crowding distance for each individual in a front."""
    if not front:
        return {}
    distance = {idx: 0.0 for idx in front}
    num_objs = len(results[0]['objs'])

    for m in range(num_objs):
        front_sorted = sorted(front, key=lambda idx: results[idx]['objs'][m])
        
        distance[front_sorted[0]] = float('inf')
        distance[front_sorted[-1]] = float('inf')
        
        f_min = results[front_sorted[0]]['objs'][m]
        f_max = results[front_sorted[-1]]['objs'][m]
        
        if f_max - f_min > EPSILON:
            for i in range(1, len(front_sorted) - 1):
                prev_obj = results[front_sorted[i-1]]['objs'][m]
                next_obj = results[front_sorted[i+1]]['objs'][m]
                distance[front_sorted[i]] += (next_obj - prev_obj) / (f_max - f_min)
    return distance

def tournament_selection(results, lam, k=2):
    """Selects an individual using a binary tournament."""
    idxs = random.sample(range(len(results)), k)
    best = idxs[0]
    for idx in idxs[1:]:
        if dominates(results[idx], results[best], lam):
            best = idx
    return best

def crossover(p1_hparams, p2_hparams):
    """Performs crossover on hyperparameter dictionaries."""
    c1, c2 = deepcopy(p1_hparams), deepcopy(p2_hparams)
    for key in c1:
        if random.random() < 0.5:
            c1[key], c2[key] = c2[key], c1[key]
    return c1, c2

def mutate(hparams):
    """Mutates a hyperparameter set."""
    ind = deepcopy(hparams)
    if random.random() < MUTATION_PROB: ind['filters'] = random.choice(FILTER_OPTIONS)
    if random.random() < MUTATION_PROB: ind['kernel_size'] = random.choice(KERNEL_SIZE_OPTIONS)
    if random.random() < MUTATION_PROB: ind['use_bn'] = not ind['use_bn']
    if random.random() < MUTATION_PROB: ind['residual_blocks'] = random.choice(RESIDUAL_BLOCK_OPTIONS)
    if random.random() < MUTATION_PROB: ind['fc_layers'] = random.choice(FC_LAYER_OPTIONS)
    if random.random() < MUTATION_PROB: ind['use_dropout'] = not ind['use_dropout']
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

# --- Main Surrogate-Assisted NSGA-II Loop ---

def nsga2(pop_size, max_gen, infill_percent):
    """
    Performs the main NSGA-II optimization loop with surrogate assistance.
    """
    # 1. Initialize population & perform true evaluation
    print("--- Initializing Population ---")
    pop_data = initialize_population()
    print(f"Initial population size: {len(pop_data)}")
    
    # 2. Initialize and train surrogate models on the initial population
    surrogate_manager = SurrogateManager()
    initial_hparams = [d['hparams'] for d in pop_data]
    surrogate_manager.update(initial_hparams, pop_data)
    
    gen_dfs = [] # To store data from each generation
    total_start_time = time.perf_counter()

    for gen in range(max_gen):
        gen_start_time = time.perf_counter()
        lam = get_lambda(gen)
        print(f"\n--- Generation {gen}/{max_gen-1}, Lambda={lam:.2f} ---")

        # 3. Sort current population and select parents
        fronts = fast_non_dominated_sort(pop_data, lam)
        parents_indices = [tournament_selection(pop_data, lam) for _ in range(pop_size)]
        
        # 4. Generate offspring via crossover and mutation
        offspring_hparams = []
        parent_hparams = [pop_data[i]['hparams'] for i in parents_indices]
        # Ensure we generate pop_size offspring
        while len(offspring_hparams) < pop_size:
            p1, p2 = random.sample(parent_hparams, 2)
            if random.random() < CROSSOVER_PROB:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = deepcopy(p1), deepcopy(p2)
            offspring_hparams.extend([mutate(c1), mutate(c2)])
        offspring_hparams = offspring_hparams[:pop_size]

        # 5. Evaluate offspring using surrogates
        print(f"Evaluating {len(offspring_hparams)} offspring with surrogate models...")
        off_data_predicted = surrogate_manager.predict(offspring_hparams)
        
        # 6. Select infill points for true evaluation and update surrogates
        num_infill = max(1, int(pop_size * infill_percent))
        print(f"Selecting {num_infill} infill points for true evaluation...")
        infill_indices, infill_hparams = select_infill_points(off_data_predicted, num_infill)
        
        # Perform true evaluation only on the infill points
        infill_data_true = compute_objectives_and_constraints(infill_hparams)
        
        # Update the surrogate models with this new, high-quality information
        surrogate_manager.update(infill_hparams, infill_data_true)
        
        # 7. Create the full offspring dataset for selection
        # Use true values for infill points, and predicted values for the rest
        off_data = list(off_data_predicted) # Create a mutable copy
        for i, true_res in enumerate(infill_data_true):
            # The infill_indices tell us where the truly evaluated individuals are in the predicted list
            original_index = infill_indices[i]
            off_data[original_index] = true_res

        # 8. Combine parents and offspring and select the next generation
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
        
        # --- Logging and Saving ---
        gen_time = time.perf_counter() - gen_start_time
        print(f"Generation {gen} finished in {gen_time:.2f} seconds.")

        # Store generation data in a DataFrame
        gen_records = []
        for ind in pop_data:
            rec = {'Generation': gen, 'Accuracy': -ind['objs'][0], 'Size_MB': ind['objs'][1], 
                   'FPR': ind['objs'][2], 'CV': ind['CV'], **ind['hparams']}
            gen_records.append(rec)
        gen_df = pd.DataFrame(gen_records)
        gen_dfs.append(gen_df)

        # Print generation status
        feasibles = [ind for ind in pop_data if ind['CV'] == 0]
        frac_feas = len(feasibles) / len(pop_data) if pop_data else 0
        avg_cv = np.mean([ind['CV'] for ind in pop_data]) if pop_data else 0
        print(f"  -> Status: Fraction Feasible={frac_feas:.2f}, Avg CV={avg_cv:.4f}")

    # --- Final Results ---
    total_time = time.perf_counter() - total_start_time
    print(f"\n--- NSGA-II Finished in {total_time/60:.2f} minutes ---")
    
    final_feasibles = [ind for ind in pop_data if ind['CV'] == 0]
    if not final_feasibles:
        print("WARNING: No feasible solutions found in the final population.")
        return [], gen_dfs

    final_fronts = fast_non_dominated_sort(final_feasibles, LAMBDA_FINAL)
    if not final_fronts:
        print("WARNING: Could not determine a Pareto front from feasible solutions.")
        return [], gen_dfs
        
    pareto_indices = final_fronts[0]
    pareto_set = [final_feasibles[i] for i in pareto_indices]
    
    return pareto_set, gen_dfs

# --- Run Experiment and Save Results ---

if __name__ == "__main__":
    pareto_solutions, all_gen_dfs = nsga2(POP_SIZE, MAX_GEN, INFILL_PERCENT)

    # Save all generation data to a single Excel file
    output_dir = "2_stage_init_sa_nsga_results"
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, "all_generations_surrogate.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        for i, df in enumerate(all_gen_dfs):
            df.to_excel(writer, sheet_name=f"Gen_{i}", index=False)
    print(f"\n✔️ All generation data saved to '{excel_path}'")

    # Print and save the final Pareto-optimal solutions
    print("\n--- Final Pareto-Optimal Feasible Solutions ---")
    if pareto_solutions:
        pareto_records = []
        for sol in pareto_solutions:
            acc, size, fpr = -sol['objs'][0], sol['objs'][1], sol['objs'][2]
            print(f"  - Acc={acc:.4f}, Size={size:.3f}MB, FPR={fpr:.4f}, HParams={sol['hparams']}")
            record = {'Accuracy': acc, 'Size_MB': size, 'FPR': fpr, **sol['hparams']}
            pareto_records.append(record)
        
        pareto_df = pd.DataFrame(pareto_records)
        csv_path = os.path.join(output_dir, "final_pareto_surrogate.csv")
        pareto_df.to_csv(csv_path, index=False)
        print(f"✔️ Final Pareto front saved to '{csv_path}'")
    else:
        print("No Pareto-optimal solutions were found.")
