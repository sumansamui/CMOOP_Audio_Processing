import tensorflow as tf
import numpy as np
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import csv
from keras import optimizers
import keras
from functools import partial
# from keras.backend import sigmoid
from math import exp
from keras.utils import get_custom_objects
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import joblib
import pickle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from itertools import cycle
import pandas as pd
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set only the first GPU as visible
        tf.config.set_visible_devices(gpus[0], 'GPU')
        # Allow memory growth to allocate memory dynamically on the GPU
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU configuration successful.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected.")

def load_data(data_path):
    """Loads training dataset from .npy files.
    :param data_path (str): Path to directory containing .npy files
    :return X_train, X_test, X_validation, y_train, y_test, y_validation (ndarray): Data splits
    """
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
    """Creates train, validation and test sets with scaling.
    :param data_path (str): Path to directory containing data
    :return X_train, y_train, X_validation, y_validation, X_test, y_test (ndarray): Processed data splits
    """
    X_train, X_test, X_validation, y_train, y_test, y_validation = load_data(data_path)

    # Initialize and fit scaler ONLY on the training data
    scaler = StandardScaler()
    num_instances_train, num_time_steps_train, num_features_train = X_train.shape
    X_train_reshaped = X_train.reshape(-1, num_features_train)
    scaler.fit(X_train_reshaped)

    # Save the scaler
    # joblib.dump(scaler, './scaler/scaler.pkl')

    # Transform training data
    X_train = scaler.transform(X_train_reshaped).reshape(num_instances_train, num_time_steps_train, num_features_train)

    # Transform validation data
    num_instances_val, num_time_steps_val, num_features_val = X_validation.shape
    X_validation_reshaped = X_validation.reshape(-1, num_features_val)
    X_validation = scaler.transform(X_validation_reshaped).reshape(num_instances_val, num_time_steps_val, num_features_val)

    # Transform test data
    num_instances_test, num_time_steps_test, num_features_test = X_test.shape
    X_test_reshaped = X_test.reshape(-1, num_features_test)
    X_test = scaler.transform(X_test_reshaped).reshape(num_instances_test, num_time_steps_test, num_features_test)

    # Add channel dimension for CNN
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, y_train, X_validation, y_validation, X_test, y_test

# --- Configuration ---
DATA_PATH = "/home/ec.gpu/Desktop/Soumen/Dataset/kws/data_10_wav/data_npy_3000"
CLASS_NAMES = ['off', 'left', 'down', 'up', 'go', 'on', 'stop', 'unknown', 'right', 'yes']
CLASSES = 10
EPOCHS = 300
BATCH_SIZE = 64
PATIENCE = 5
LEARNING_RATE = 0.0001

# --- Load Data ---
X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_dataset(DATA_PATH)
print("Data Shapes:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_validation: {X_validation.shape}, y_validation: {y_validation.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")


# =============================================================================
# Hyperparameter Search Space
# =============================================================================
FILTER_OPTIONS = [16, 32, 64]
KERNEL_SIZE_OPTIONS = [3, 5]
USE_BN_OPTIONS = [True, False]
RESIDUAL_BLOCK_OPTIONS = [1, 2, 3]
FC_LAYER_OPTIONS = [1, 2, 3, 4]
USE_DROPOUT_OPTIONS = [True, False]

# =============================================================================
# NSGA-II Parameters
# =============================================================================
POP_SIZE = 15
MAX_GEN = 30
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.2
EPSILON = 1e-6

# =============================================================================
# MODIFIED: Constraint thresholds (Accuracy constraint removed)
# =============================================================================
MAX_MODEL_SIZE = 2.5  # in MB
MAX_FPR = 0.09        # Maximum allowed False Positive Rate

# =============================================================================
# Penalty schedule
# =============================================================================
LAMBDA_INITIAL = 1.0
LAMBDA_FINAL = 50.0

def get_lambda(gen):
    frac = gen / float(MAX_GEN - 1)
    return LAMBDA_INITIAL + frac * (LAMBDA_FINAL - LAMBDA_INITIAL)

# =============================================================================
# Model Building and Evaluation
# =============================================================================
from tensorflow.keras import layers, Model
def build_model(hparams):
    """
    Builds a custom CNN model based on the provided hyperparameters.
    """
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    filters = hparams['filters']
    kernel_size = hparams['kernel_size']
    use_bn = hparams['use_bn']
    num_res_blocks = hparams['residual_blocks']
    num_fc_layers = hparams['fc_layers']
    use_dropout = hparams['use_dropout']

    inputs = layers.Input(shape=input_shape, name="input_layer")

    # Initial Conv Block
    x = layers.Conv2D(filters, kernel_size, padding='same')(inputs)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # Residual Blocks
    for block_idx in range(num_res_blocks):
        skip = layers.Conv2D(filters * 2, (1, 1), strides=(2, 2), padding='same')(x)
        y = layers.Conv2D(filters * 2, kernel_size, padding='same')(x)
        if use_bn:
            y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        y = layers.Conv2D(filters * 2, kernel_size, padding='same')(y)
        if use_bn:
            y = layers.BatchNormalization()(y)
        y = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(y)
        x = layers.add([y, skip])
        x = layers.ReLU(name=f"res_relu_{block_idx}")(x)
        filters *= 2

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Dense (FC) Layers
    fc_layer_configs = {1: [64], 2: [128, 64], 3: [256, 128, 64], 4: [512, 256, 128, 64]}
    if num_fc_layers in fc_layer_configs:
        for i, units in enumerate(fc_layer_configs[num_fc_layers]):
            x = layers.Dense(units, activation='relu', name=f"fc{i+1}")(x)
            if use_dropout:
                x = layers.Dropout(0.3, name=f"dropout{i+1}")(x)

    # Output Layer
    outputs = layers.Dense(CLASSES, activation='softmax', name="output_layer")(x)
    model = Model(inputs=inputs, outputs=outputs, name="CustomCNN")
    return model

def compute_model_size_mb(model):
    """Calculate model size in MB."""
    return model.count_params() * 4 / (1024**2)

def calculate_fpr(y_true, y_pred, num_classes):
    """Calculate macro-averaged False Positive Rate."""
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
    Build, train, and evaluate a model to get accuracy, size, and FPR.
    """
    tf.keras.backend.clear_session()
    model = build_model(hparams)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping], verbose=0)
    
    loss, accuracy = model.evaluate(X_validation, y_validation, verbose=0)
    print(f"Validation accuracy >>: {accuracy:.4f}")
    
    y_pred_probs = model.predict(X_validation)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    
    fpr = calculate_fpr(y_validation.flatten(), y_pred_labels, CLASSES)
    print(f"Macro-averaged FPR: {fpr:.4f}")

    size_mb = compute_model_size_mb(model)
    print(f"Model size: {size_mb:.2f} MB")

    return accuracy, size_mb, fpr


# =============================================================================
# NSGA-II Core Functions
# =============================================================================
def initialize_population(pop_size):
    """Create the initial population with random hyperparameters."""
    population = []
    for _ in range(pop_size):
        hparams = {
            'filters': random.choice(FILTER_OPTIONS),
            'kernel_size': random.choice(KERNEL_SIZE_OPTIONS),
            'use_bn': random.choice(USE_BN_OPTIONS),
            'residual_blocks': random.choice(RESIDUAL_BLOCK_OPTIONS),
            'fc_layers': random.choice(FC_LAYER_OPTIONS),
            'use_dropout': random.choice(USE_DROPOUT_OPTIONS)
        }
        print(f"Initialized hyperparameters: {hparams}")
        population.append(hparams)
    return population

def compute_objectives_and_constraints(population):
    """
    MODIFIED: Evaluate each individual for 2 objectives (Size, FPR) and 2 constraints.
    The accuracy is calculated and stored but not used as an objective or constraint.
    """
    results = []
    for ind in population:
        acc, size_mb, fpr = evaluate_individual(ind)
        
        # MODIFIED: Objectives are now [Size, FPR]
        f1 = size_mb # Minimize model size
        f2 = fpr    # Minimize False Positive Rate
        
        # The accuracy is now just a tracked metric, not an objective
        tracked_acc = acc
        
        # MODIFIED: Constraint violations (g1 for accuracy is removed)
        g2 = max(0.0, size_mb - MAX_MODEL_SIZE)
        g3 = max(0.0, fpr - MAX_FPR)
        CV = g2 + g3
        
        results.append({
            'acc_metric': tracked_acc, # Store accuracy separately
            'hparams': ind,
            'objs': [f1, f2],         # List of 2 objectives
            'CV': CV
        })
    return results

def dominates(a, b, lam):
    """Dominance check with adaptive penalty."""
    fa, ga = a['objs'], a['CV']
    fb, gb = b['objs'], b['CV']
    Pa = [fa[i] + lam * ga for i in range(len(fa))]
    Pb = [fb[i] + lam * gb for i in range(len(fb))]
    
    better_all = all(Pa[i] <= Pb[i] for i in range(len(Pa)))
    strictly_better = any(Pa[i] < Pb[i] for i in range(len(Pa)))
    
    return better_all and strictly_better

def fast_non_dominated_sort(results, lam):
    """Performs fast non-dominated sort with adaptive penalty."""
    fronts = [[]]
    S = [[] for _ in range(len(results))]
    n = [0] * len(results)
    rank = [0] * len(results)

    for p_idx, p_res in enumerate(results):
        for q_idx, q_res in enumerate(results):
            if p_idx == q_idx: continue
            if dominates(p_res, q_res, lam):
                S[p_idx].append(q_idx)
            elif dominates(q_res, p_res, lam):
                n[p_idx] += 1
        if n[p_idx] == 0:
            rank[p_idx] = 0
            fronts[0].append(p_idx)
    
    i = 0
    while fronts[i]:
        next_front = []
        for p_idx in fronts[i]:
            for q_idx in S[p_idx]:
                n[q_idx] -= 1
                if n[q_idx] == 0:
                    rank[q_idx] = i + 1
                    next_front.append(q_idx)
        i += 1
        if next_front:
            fronts.append(next_front)
        else:
            break
            
    return fronts

def crowding_distance(front, results):
    """Compute crowding distance for a given front."""
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
                prev_obj = results[front_sorted[i - 1]]['objs'][m]
                next_obj = results[front_sorted[i + 1]]['objs'][m]
                distance[front_sorted[i]] += (next_obj - prev_obj) / (f_max - f_min)
    return distance

def tournament_selection(results, lam, k=2):
    """Binary tournament selection using penalized dominance."""
    idxs = random.sample(range(len(results)), k)
    best = idxs[0]
    for idx in idxs[1:]:
        if dominates(results[idx], results[best], lam):
            best = idx
    return best

def crossover(parent1, parent2):
    """Single-point style crossover on hyperparameter dicts."""
    child1 = deepcopy(parent1)
    child2 = deepcopy(parent2)
    keys_to_swap = list(parent1.keys())
    
    for key in keys_to_swap:
        if random.random() < 0.5:
             child1[key], child2[key] = child2[key], child1[key]

    return child1, child2

def mutate(individual):
    """Mutate each hyperparameter with probability MUTATION_PROB."""
    ind = deepcopy(individual)
    if random.random() < MUTATION_PROB:
        ind['filters'] = random.choice(FILTER_OPTIONS)
    if random.random() < MUTATION_PROB:
        ind['kernel_size'] = random.choice(KERNEL_SIZE_OPTIONS)
    if random.random() < MUTATION_PROB:
        ind['use_bn'] = not ind['use_bn']
    if random.random() < MUTATION_PROB:
        ind['residual_blocks'] = random.choice(RESIDUAL_BLOCK_OPTIONS)
    if random.random() < MUTATION_PROB:
        ind['fc_layers'] = random.choice(FC_LAYER_OPTIONS)
    if random.random() < MUTATION_PROB:
        ind['use_dropout'] = not ind['use_dropout']
    
    if ind != individual:
        print(f"Mutated hyperparameters: {ind}")

    return ind

# =============================================================================
# NSGA‐II Main Loop
# =============================================================================
def nsga2(pop_size, max_gen):
    # 1. Initialize and evaluate population
    population_hparams = initialize_population(pop_size)
    pop_data = compute_objectives_and_constraints(population_hparams)
    gen_dfs = []

    for gen in range(max_gen):
        lam = get_lambda(gen)
        print(f"\n--- Generation {gen}, Lambda = {lam:.3f} ---")
        start_gen_time = time.perf_counter()

        # 2. Generate and evaluate offspring
        parents_indices = [tournament_selection(pop_data, lam, k=2) for _ in range(pop_size)]
        
        offspring_hparams = []
        for i in range(0, pop_size, 2):
            if i + 1 < len(parents_indices):
                parent1 = pop_data[parents_indices[i]]['hparams']
                parent2 = pop_data[parents_indices[i+1]]['hparams']
                if random.random() < CROSSOVER_PROB:
                    child1, child2 = crossover(parent1, parent2)
                else:
                    child1, child2 = deepcopy(parent1), deepcopy(parent2)
                offspring_hparams.extend([mutate(child1), mutate(child2)])
        
        # Evaluate offspring
        off_data = compute_objectives_and_constraints(offspring_hparams)

        # 3. Combine parents and offspring
        combined_data = pop_data + off_data
        
        # 4. Sort combined population
        combined_fronts = fast_non_dominated_sort(combined_data, lam)

        # 5. Build the next generation
        new_pop_data = []
        for front in combined_fronts:
            if len(new_pop_data) + len(front) <= pop_size:
                new_pop_data.extend([combined_data[i] for i in front])
            else:
                remaining_space = pop_size - len(new_pop_data)
                crowd_dist = crowding_distance(front, combined_data)
                sorted_front = sorted(front, key=lambda idx: crowd_dist[idx], reverse=True)
                new_pop_data.extend([combined_data[i] for i in sorted_front[:remaining_space]])
                break
        
        pop_data = new_pop_data
        gen_time = time.perf_counter() - start_gen_time
        print(f"Generation {gen} took {gen_time:.2f} seconds")

        # --- Logging and Saving ---
        gen_records = []
        for ind in pop_data:
            rec = {
                'Generation': gen,
                'Accuracy': ind['acc_metric'], # MODIFIED: Get Accuracy from its new key
                'Size_MB': ind['objs'][0],
                'FPR': ind['objs'][1],
                'CV': ind['CV'],
                **ind['hparams']
            }
            gen_records.append(rec)
        
        gen_df = pd.DataFrame(gen_records)
        gen_dfs.append(gen_df)

        # --- Print Status ---
        feasibles = [ind for ind in pop_data if ind['CV'] == 0]
        frac_feas = len(feasibles) / len(pop_data)
        avg_cv = np.mean([ind['CV'] for ind in pop_data])
        print(f"  Fraction Feasible: {frac_feas:.2f}, Avg CV: {avg_cv:.4f}")
        
        if feasibles:
            best_size = min(ind['objs'][0] for ind in feasibles)
            best_fpr = min(ind['objs'][1] for ind in feasibles)
            print(f"  Best feasible metrics in current pop -> Size: {best_size:.3f} MB, FPR: {best_fpr:.4f}")
        else:
            print("  No feasible solutions this generation.")

    # Final extraction of Pareto front from strictly feasible solutions
    final_feasibles = [ind for ind in pop_data if ind['CV'] == 0]
    if not final_feasibles:
        print("\nWARNING: No feasible solutions found in the final population.")
        return [], gen_dfs

    # Use a large penalty to ensure only CV=0 solutions are in the first front
    final_fronts = fast_non_dominated_sort(final_feasibles, lam=LAMBDA_FINAL * 100)
    pareto_indices = final_fronts[0]
    pareto_set = [final_feasibles[i] for i in pareto_indices]
    
    return pareto_set, gen_dfs

# =============================================================================
# Run Experiment and Save Results
# =============================================================================
# NOTE: The file paths for saving results have been updated.
# Please ensure the directory exists or change the path as needed.
output_dir = "/home/ec.gpu/Desktop/Soumen/nitd_0.4/nitd-0.4/kws/kws/c_moo/2_stage/size_fpr"
os.makedirs(output_dir, exist_ok=True)

pareto_solutions, all_gen_dfs = nsga2(POP_SIZE, MAX_GEN)

# --- Save all generation data to an Excel file ---
excel_path = os.path.join(output_dir, "all_generations_size_fpr.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    for i, df in enumerate(all_gen_dfs):
        df.to_excel(writer, sheet_name=f"Gen_{i}", index=False)
print(f"\n✔️ All generation data saved to '{excel_path}'")

# --- Process and save the final Pareto front ---
if pareto_solutions:
    print("\nFinal Pareto-Optimal Feasible Solutions (Size vs. FPR):")
    pareto_records = []
    for sol in pareto_solutions:
        size = sol['objs'][0]
        fpr = sol['objs'][1]
        acc = sol['acc_metric'] # MODIFIED: Get Accuracy from its new key
        
        print(f"  Size={size:.3f} MB, FPR={fpr:.4f}, Accuracy={acc:.4f}, Hyperparams={sol['hparams']}")
        
        record = {
            'Accuracy': acc,
            'Size_MB': size,
            'FPR': fpr,
            **sol['hparams']
        }
        pareto_records.append(record)

    # Save final Pareto front to a CSV file
    pareto_df = pd.DataFrame(pareto_records)
    csv_path = os.path.join(output_dir, "final_pareto_size_fpr.csv")
    pareto_df.to_csv(csv_path, index=False)
    print(f"✔️ Final Pareto solutions saved to '{csv_path}'")
else:
    print("\nNo Pareto-optimal feasible solutions were found.")