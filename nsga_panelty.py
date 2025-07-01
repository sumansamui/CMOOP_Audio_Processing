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

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

gpus = tf.config.list_physical_devices('GPU')
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
    """Loads training dataset from json file.
    :param data_path (str): Path to json file containing data
    :return X (ndarray): Inputs
    :return y (ndarray): Targets
    """
				
    X_train = np.load(f'{data_path}/X_train.npy')
    X_test = np.load(f'{data_path}/X_test.npy')
    X_validation = np.load(f'{data_path}/X_val.npy')
 

    y_train = np.load(f'{data_path}/y_train.npy')
    y_test = np.load(f'{data_path}/y_test.npy')
    y_validation = np.load(f'{data_path}/y_val.npy')
    # print(y_train.shape)

    y_train = y_train[..., np.newaxis]
    y_test = y_test[..., np.newaxis]
    y_validation = y_validation[..., np.newaxis]
    # print(y_train.shape)

    #y = y -1
    print("Dataset loaded!")
    
    
    return X_train, X_test, X_validation, y_train, y_test, y_validation

def prepare_dataset(data_path):
    """Creates train, validation and test sets.
    :param data_path (str): Path to json file containing data
    :param test_size (flaot): Percentage of dataset used for testing
    :param validation_size (float): Percentage of train set used for cross-validation
    :return X_train (ndarray): Inputs for the train set
    :return y_train (ndarray): Targets for the train set
    :return X_validation (ndarray): Inputs for the validation set
    :return y_validation (ndarray): Targets for the validation set
    :return X_test (ndarray): Inputs for the test set
    :return X_test (ndarray): Targets for the test set
    """

    # load dataset
    X_train, X_test, X_validation, y_train, y_test, y_validation = load_data(data_path)
    #print(X.shape)
    
    ################################## Scaleing the data #############################
    scaler = StandardScaler()
    num_instances, num_time_steps, num_features = X_train.shape
    #print(num_instances)
    #print(num_time_steps)
    #print(num_features)

    X_train = X_train.reshape(-1, num_features)
    #print(X_train.shape)
    X_train = scaler.fit_transform(X_train)
    
     #reshapeing
    X_train = X_train.reshape(num_instances, num_time_steps, num_features) 
    #print(X_train.shape)

    num_instances, num_time_steps, num_features = X_test.shape
    #print(num_instances)
    #print(num_time_steps)
    #print(num_features)

    X_test = X_test.reshape(-1, num_features)
    #print(X.shape)
    X_test = scaler.fit_transform(X_test)
    
     #reshapeing
    X_test = X_test.reshape(num_instances, num_time_steps, num_features) 
    #print(X_test.shape)

    num_instances, num_time_steps, num_features = X_validation.shape
    #print(num_instances)
    #print(num_time_steps)
    #print(num_features)

    X_validation = X_validation.reshape(-1, num_features)
    #print(X.shape)
    X_validation = scaler.fit_transform(X_validation)
    
     #reshapeing
    X_validation = X_validation.reshape(num_instances, num_time_steps, num_features) 
    #print(X_validation.shape)    

    #######################################
    #######################################
    # # Save the scaler to a file
    # joblib.dump(scaler, './scaler/scaler.pkl')
    #######################################
    
    # print(X_train.shape)
    # add an axis to nd array
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]

    return X_train, y_train, X_validation, y_validation, X_test, y_test

DATA_PATH =  "/home/22EC1102/soumen/data/KWS_10_log_mel_3000/data_npy_3000"  #"/home/22EC1102/soumen/data/kws_10_log_mel"
class_names = ['off', 'left', 'down', 'up', 'go', 'on', 'stop', 'unknown', 'right', 'yes']  #, 'silence' , 'no'
EPOCHS = 300
BATCH_SIZE = 64
PATIENCE = 5
LEARNING_RATE = 0.0001
SKIP = 1
CLASS = 10

# generate train, validation and test sets
X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_dataset(DATA_PATH)
print(X_train.shape)
print(y_train.shape)
print(X_validation.shape)
print(y_validation.shape)
print(X_test.shape)
print(y_test.shape)


CLASSES = 10
EPOCHS = 300
BATCH_SIZE = 64  #64
PATIENCE = 5  # 5

# =============================================================================
# New Hyperparameter Search Space (for the “Initial Conv + Residual Blocks” model)
# =============================================================================

# 1) Base number of filters for the initial conv block:
FILTER_OPTIONS       = [16, 32, 64]        # (e.g. try 16, 32 or 64)
# 2) Kernel size for all Conv2D layers:
KERNEL_SIZE_OPTIONS  = [3, 5]             # (3×3 or 5×5)
# 3) Whether to use BatchNormalization after each conv:
USE_BN_OPTIONS       = [True, False]
# 4) How many residual blocks to stack (each doubles filters + downsamples):
RESIDUAL_BLOCK_OPTIONS = [1, 2, 3]        # (1, 2, or 3 blocks)
# 5) How many fully‐connected layers after global pooling (1–4):
FC_LAYER_OPTIONS     = [1, 2, 3, 4]       # (choose exactly this many FC layers)
# 6) Whether to apply Dropout(0.2) after each dense layer:
USE_DROPOUT_OPTIONS  = [True, False]
#####################################################
# NSGA-II parameters
POP_SIZE = 15 #10
MAX_GEN = 30  #5
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.2
EPSILON = 1e-6  # Small value to break ties in crowding distance    # 1e-9

# Constraint thresholds
MIN_ACCURACY = 0.9  #0.90
MAX_MODEL_SIZE = 2.5 # 2.5  # in MB
MAX_FPR = 0.1 #0.09 #0.01


# =============================================================================
# Penalty schedule
# =============================================================================
LAMBDA_INITIAL = 1.0
LAMBDA_FINAL   = 50.0

def get_lambda(gen):
    frac = gen / float(MAX_GEN - 1)
    return LAMBDA_INITIAL + frac * (LAMBDA_FINAL - LAMBDA_INITIAL)



from tensorflow.keras import layers, Model

def build_model(hparams):
    """
    Builds a CNN with:
      - Initial Conv Block
      - X Residual Blocks
      - Global Average Pooling
      - Dense FC layers (1–4)
    
    hparams must contain:
      - 'filters': base number of filters for the initial block (int)
      - 'kernel_size': size of the Conv2D kernels, e.g. 3 or 5 (int)
      - 'use_bn': whether to apply BatchNormalization (bool)
      - 'residual_blocks': how many residual blocks to stack (int ≥ 1)
      - 'fc_layers': number of dense layers after pooling (choose 1,2,3,4)
      - 'use_dropout': whether to add Dropout(0.2) after each dense (bool)
    """
    input_shape = (X_train.shape[1], X_train.shape[2], 1)    # e.g. (height, width, channels)
    
    filters       = hparams['filters']
    kernel_size   = hparams['kernel_size']
    use_bn        = hparams['use_bn']
    num_res_blocks = hparams['residual_blocks']
    num_fc_layers  = hparams['fc_layers']
    use_dropout   = hparams['use_dropout']
    
    inputs = layers.Input(shape=input_shape, name="input_layer")

    # -------------------------------------------------------------------------
    # Initial Conv Block
    # -------------------------------------------------------------------------
    x = layers.Conv2D(filters, kernel_size, padding='same')(inputs)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.ReLU(name="relu1_1")(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = layers.ReLU(name="relu1_2")(x)
    
    x = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)

    # -------------------------------------------------------------------------
    # Residual Blocks
    # Each residual block:
    #   - Projects x via 1×1 conv with stride=2 to match new filter dimensions
    #   - Two Conv2D → (BN?) → ReLU → Conv2D → (BN?)
    #   - Downsample / MaxPool by factor 2
    #   - Add the projected skip connection, then ReLU
    # Filters double on each new block.
    # -------------------------------------------------------------------------
    for block_idx in range(num_res_blocks):
        # 1×1 projection for the skip connection (downsample + increase filters)
        skip = layers.Conv2D(filters * 2, (1,1),
                             strides=(2,2),
                             padding='same')(x)
        
        # First conv in the block
        y = layers.Conv2D(filters * 2, kernel_size, padding='same')(x)
        if use_bn:
            y = layers.BatchNormalization()(y)
        y = layers.ReLU()(y)
        
        # Second conv in the block
        y = layers.Conv2D(filters * 2, kernel_size, padding='same')(y)
        if use_bn:
            y = layers.BatchNormalization()(y)
        
        # Downsample / MaxPool
        y = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(y)
        
        # Add skip connection
        x = layers.add([y, skip])
        x = layers.ReLU(name=f"res_relu_{block_idx}")(x)
        
        # Double filters for next block
        filters *= 2

    # -------------------------------------------------------------------------
    # Global Average Pooling
    # -------------------------------------------------------------------------
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # -------------------------------------------------------------------------
    # Dense (FC) Layers

    fc_layer_configs = {
        1: [64],
        2: [128, 64],
        3: [256, 128, 64],
        4: [512, 256, 128, 64]
    }

    if num_fc_layers in fc_layer_configs:
        for i, units in enumerate(fc_layer_configs[num_fc_layers]):
            x = layers.Dense(units, activation='relu',
                             name=f"fc{i+1}")(x)
            if use_dropout:
                x = layers.Dropout(0.3, name=f"dropout{i+1}")(x)

    # -------------------------------------------------------------------------
    # Final Softmax Output
    # -------------------------------------------------------------------------
    outputs = layers.Dense(CLASSES,
                           activation='softmax',
                           name="output_layer")(x)

    model = Model(inputs=inputs, outputs=outputs, name="CustomCNN")
    # model.summary()
    return model


def compute_model_size_mb(model):
    """
    Calculate model size in megabytes (MB) by summing parameter count * 4 bytes.
    """
    param_count = model.count_params()
    size_bytes = param_count * 4  # 4 bytes per float32 parameter
    size_mb = size_bytes / (1024**2)
    return size_mb

import numpy as np
from sklearn.metrics import confusion_matrix


####################
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
    Build/train the model, then compute:
      - Accuracy on validation set (multiclass)
      - False Positive Rate (macro‐averaged over 10 classes)
      - Model size in MB
    """
    tf.keras.backend.clear_session()
    model = build_model(hparams)  # same build_model as before
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train for a small number of epochs for demonstration
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE)
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping], verbose=0)
    accuracy = history.history['val_accuracy'][-1]
    print(f"Validation accuracy >>: {accuracy:.4f}")
    y_pred = np.argmax(model.predict(X_validation), axis=1)
    y_true = np.argmax(y_validation, axis=1)  
    fpr = calculate_fpr(y_true, y_pred, CLASSES)   
    print(f"Macro-averaged FPR: {fpr:.4f}")

    # 3) Model size
    size_mb = compute_model_size_mb(model)
    print(f"Model size: {size_mb:.2f} MB")

    return accuracy, size_mb, fpr


# -------------------------------------------------------------
# NSGA-II helper functions
# -------------------------------------------------------------

def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        hparams = {
																		'filters':          random.choice(FILTER_OPTIONS),
																		'kernel_size':      random.choice(KERNEL_SIZE_OPTIONS),
																		'use_bn':           random.choice(USE_BN_OPTIONS),
																		'residual_blocks':  random.choice(RESIDUAL_BLOCK_OPTIONS),
																		'fc_layers':        random.choice(FC_LAYER_OPTIONS),
																		'use_dropout':      random.choice(USE_DROPOUT_OPTIONS)
														}
        print(f"Initialized hyperparameters: {hparams}")
        population.append(hparams)
    return population


def compute_objectives_and_constraints(population):
    """
    For each individual in population, compute (f1, f2, f3) and constraint violations.
    We store:
      - 'objs': [f1, f2, f3]
      - 'CV': total constraint violation = sum(max(0, threshold - actual))
    """
    results = []
    for ind in population:
        acc, size_mb, fpr = evaluate_individual(ind)
        # Objectives: we minimize [-Accuracy, ModelSize, FPR]
        f1 = -acc
        f2 = size_mb
        f3 = fpr
        # Constraint violations:
        g1 = max(0.0, MIN_ACCURACY - acc)
        g2 = max(0.0, size_mb - MAX_MODEL_SIZE)
        g3 = max(0.0, fpr - MAX_FPR)
        CV = g1 + g2 + g3
        results.append({
            'hparams': ind,
            'objs': [f1, f2, f3],
            'CV': CV
        })
    return results


# =============================================================================
# Dominance with penalty
# =============================================================================
def dominates(a, b, lam):
    fa, ga = a['objs'], a['CV']
    fb, gb = b['objs'], b['CV']
    Pa = [fa[i] + lam * ga for i in range(len(fa))]
    Pb = [fb[i] + lam * gb for i in range(len(fb))]
    better_all = True
    strictly_better = False
    for i in range(len(Pa)):
        if Pa[i] > Pb[i]:   # Condition 1: Pa is worse than Pb in objective i
            better_all = False
            break
        if Pa[i] < Pb[i]:   # Condition 2: Pa is strictly better than Pb in objective i
            strictly_better = True
    return better_all and strictly_better


# =============================================================================
# (3) Fast non-dominated sort that takes lam
# =============================================================================
def fast_non_dominated_sort(results, lam):
    fronts = []
    S = [[] for _ in range(len(results))]
    n = [0] * len(results)
    rank = [0] * len(results)

    for p in range(len(results)):
        S[p] = []
        n[p] = 0
        for q in range(len(results)):
            if dominates(results[p], results[q], lam):
                S[p].append(q)
            elif dominates(results[q], results[p], lam):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            if len(fronts) == 0:
                fronts.append([])
            fronts[0].append(p)

    i = 0
    while True:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        if len(next_front) == 0:
            break
        fronts.append(next_front)
        i += 1

    return fronts


def crowding_distance(front, results):
    """
    Compute crowding distance for a given front. Returns a dict mapping index->distance.
    """
    distance = {idx: 0.0 for idx in front}
    num_objs = len(results[0]['objs'])

    for m in range(num_objs):
        front_sorted = sorted(front, key=lambda idx: results[idx]['objs'][m])
        # Assign infinite distance to boundary points
        distance[front_sorted[0]] = float('inf')
        distance[front_sorted[-1]] = float('inf')
        f_min = results[front_sorted[0]]['objs'][m]
        f_max = results[front_sorted[-1]]['objs'][m]
        if f_max - f_min < EPSILON:
            continue
        for i in range(1, len(front_sorted)-1):
            prev_obj = results[front_sorted[i-1]]['objs'][m]
            next_obj = results[front_sorted[i+1]]['objs'][m]
            distance[front_sorted[i]] += (next_obj - prev_obj) / (f_max - f_min)
    return distance



def tournament_selection(results, lam, k=2):
    """
    Binary tournament selection using penalized dominance lam.
    Pick k random indices, return the one that dominates the other (or random if none).
    """
    idxs = random.sample(range(len(results)), k)
    best = idxs[0]
    for idx in idxs[1:]:
        if dominates(results[idx], results[best], lam):
            best = idx
    return best


def crossover(parent1, parent2):
    """
    Single‐point–style crossover on the new hyperparameter dicts. We randomly choose
    which of the six fields to swap between parent1 and parent2 to produce child1/child2.
    """
    child1 = deepcopy(parent1)
    child2 = deepcopy(parent2)

    # 1) Swap 'filters' with 50% chance
    if random.random() < 0.5:
        child1['filters'], child2['filters'] = parent2['filters'], parent1['filters']

    # 2) Swap 'kernel_size'
    if random.random() < 0.5:
        child1['kernel_size'], child2['kernel_size'] = parent2['kernel_size'], parent1['kernel_size']

    # 3) Swap 'use_bn'
    if random.random() < 0.5:
        child1['use_bn'], child2['use_bn'] = parent2['use_bn'], parent1['use_bn']

    # 4) Swap 'residual_blocks'
    if random.random() < 0.5:
        child1['residual_blocks'], child2['residual_blocks'] = (
            parent2['residual_blocks'], parent1['residual_blocks']
        )

    # 5) Swap 'fc_layers'
    if random.random() < 0.5:
        child1['fc_layers'], child2['fc_layers'] = parent2['fc_layers'], parent1['fc_layers']

    # 6) Swap 'use_dropout'
    if random.random() < 0.5:
        child1['use_dropout'], child2['use_dropout'] = (
            parent2['use_dropout'], parent1['use_dropout']
        )

    return child1, child2

def mutate(individual):
    """
    Mutate each of the six hyperparameters with probability MUTATION_PROB.
    """
    ind = deepcopy(individual)
    if random.random() < MUTATION_PROB:
        ind['filters'] = random.choice([16, 32, 64])

    if random.random() < MUTATION_PROB:
        ind['kernel_size'] = random.choice([3, 5])

    if random.random() < MUTATION_PROB:
        ind['use_bn'] = not ind['use_bn']

    if random.random() < MUTATION_PROB:
        ind['residual_blocks'] = random.choice([1, 2, 3])

    if random.random() < MUTATION_PROB:
        ind['fc_layers'] = random.choice([1, 2, 3, 4])

    if random.random() < MUTATION_PROB:
        ind['use_dropout'] = not ind['use_dropout']
    print(f"Mutated hyperparameters: {ind}")

    return ind



# =============================================================================
# NSGA‐II Main Loop (with Adaptive Penalty)
# =============================================================================
def nsga2(pop_size, max_gen):
    # 1. Initialize population & evaluate
    population = initialize_population(pop_size)
    pop_data = compute_objectives_and_constraints(population)
    # We will keep track of “per-generation” data here:
    # a list of DataFrames, one per generation.
    gen_dfs = []

    for gen in range(max_gen):
        lam = get_lambda(gen)  # current penalty coefficient
        print(f"Generation {gen}, lambda = {lam:.3f}")

        # 2. Fast non‐dominated sort under penalty lam
        fronts = fast_non_dominated_sort(pop_data, lam)

        # 3. Compute crowding distances for each front
        crowd_dist = {}
        for f in fronts:
            dist = crowding_distance(f, pop_data)
            crowd_dist.update(dist)

        # 4. Build exactly pop_size “tournament winners” (indices)
        parents = [
            tournament_selection(pop_data, lam, k=2)
            for _ in range(pop_size)
        ]
        # parents is guaranteed length = pop_size

        # 5. Generate offspring by pairing parents safely via zip
        offspring_hparams = []
        # Pair parents in adjacent pairs (0&1, 2&3, …). If pop_size is odd, the last parent is skipped.
        for idx1, idx2 in zip(parents[0::2], parents[1::2]):
            parent1 = pop_data[idx1]['hparams']
            parent2 = pop_data[idx2]['hparams']

            if random.random() < CROSSOVER_PROB:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = deepcopy(parent1), deepcopy(parent2)

            child1 = mutate(child1)
            child2 = mutate(child2)
            offspring_hparams.append(child1)
            offspring_hparams.append(child2)

        # If pop_size is odd, zip(...) leaves out the last parent. We can simply clone & mutate one:
        if pop_size % 2 == 1:
            last_idx = parents[-1]
            last_parent = pop_data[last_idx]['hparams']
            lonely_child = mutate(deepcopy(last_parent))
            offspring_hparams.append(lonely_child)

        # Ensure exactly pop_size offspring
        offspring_hparams = offspring_hparams[:pop_size]
        # print(f"  Generated {len(offspring_hparams)} offspring.")
        # print(f"  Offspring hyperparameters: {offspring_hparams}")
        # print(f"  Parents selected: {[pop_data[i]['hparams'] for i in parents]}")
									

        # 6. Evaluate offspring
        off_data = compute_objectives_and_constraints(offspring_hparams)

        # 7. Combine parents + offspring
        combined = pop_data + off_data

        # 8. Sort combined population under same lam
        combined_fronts = fast_non_dominated_sort(combined, lam)

        # 9. Build next generation by filling from combined_fronts
        new_pop_data = []
        for front in combined_fronts:
            if len(new_pop_data) + len(front) <= pop_size:
                new_pop_data.extend([combined[i] for i in front])
            else:
                remaining = pop_size - len(new_pop_data)
                front_dist = crowding_distance(front, combined)
                sorted_front = sorted(
                    front, key=lambda idx: front_dist[idx], reverse=True
                )
                new_pop_data.extend([combined[i] for i in sorted_front[:remaining]])
                break

        pop_data = new_pop_data

        # ────────────────────────────────
        # SAVE “PER‐GENERATION” DATA HERE
        # ────────────────────────────────

        # Build a DataFrame of exactly this generation’s population
        # Include: Generation, Accuracy, Size_MB, FPR, CV, and all hyperparams
        gen_records = []
        for ind in pop_data:
            acc   = -ind['objs'][0]
            size  = ind['objs'][1]
            fpr   = ind['objs'][2]
            cv    = ind['CV']
            hpar  = ind['hparams']

            rec = {
                'Generation': gen,
                'Accuracy':   acc,
                'Size_MB':    size,
                'FPR':        fpr,
                'CV':         cv,
                # Unpack each hyperparameter into its own column
                **hpar
            }
            gen_records.append(rec)

        gen_df = pd.DataFrame(gen_records)

        # Also store in gen_dfs (if you want to inspect later in Python)
        gen_dfs.append(gen_df)

        # 10. Print status: fraction feasible, average CV, best feasible metrics
        feasibles = [ind for ind in pop_data if ind['CV'] == 0]
        frac_feas = len(feasibles) / float(len(pop_data))
        avg_cv = np.mean([ind['CV'] for ind in pop_data])
        print(f"  Fraction Feasible: {frac_feas:.2f}, Avg CV: {avg_cv:.4f}")
        if feasibles:
            best_acc  = max(-ind['objs'][0] for ind in feasibles)
            best_size = min(ind['objs'][1] for ind in feasibles)
            best_fpr  = min(ind['objs'][2] for ind in feasibles)
            print(f"  Best feasible → Acc: {best_acc:.3f}, Size: {best_size:.3f} MB, FPR: {best_fpr:.4f}")
        else:
            print("  No feasible solutions this generation.")

        #################### after 5 iterations, save the DataFrame to Excel ###############
        if (gen + 1) % 5 == 0 or gen == max_gen - 1:     # 5 iterations
            final_feasibles_5 = [ind for ind in pop_data if ind['CV'] == 0]
            if not final_feasibles_5:
                print("WARNING: No feasible solutions found in final population.")

            final_fronts_5 = fast_non_dominated_sort(final_feasibles_5, LAMBDA_FINAL)
            pareto_indices_5 = final_fronts_5[0]
            pareto_set_5 = [final_feasibles_5[i] for i in pareto_indices_5]

            records_5 = []
            for sol in pareto_set_5:
                acc  = -sol['objs'][0]
                size = sol['objs'][1]
                fpr  = sol['objs'][2]
                rec_5 = {
                    'Accuracy': acc,
                    'Size_MB':  size,
                    'FPR':      fpr,
                    **sol['hparams']
                }
                records_5.append(rec_5)

            df_pareto_5 = pd.DataFrame(records_5)
            excel_path_5 = f"/home/22EC1102/soumen/NIT-GPU_2/code/c_moo/nsga_II_constrained_panelty/5_iter/mobo_iteration_{gen+1}.csv"  # data
            df_pareto_5.to_csv(excel_path_5, index=False)
            print(f"✔️ Pareto set after 5 iterations saved to '{excel_path_5}'")


    # 11. Final extraction of Pareto front among strictly feasible solutions
    final_feasibles = [ind for ind in pop_data if ind['CV'] == 0]
    if not final_feasibles:
        print("WARNING: No feasible solutions found in final population.")
        return [], gen_dfs

    # Use a large penalty to ensure only CV=0 remain in front 0
    final_fronts = fast_non_dominated_sort(final_feasibles, LAMBDA_FINAL)
    pareto_indices = final_fronts[0]
    pareto_set = [final_feasibles[i] for i in pareto_indices]
    return pareto_set , gen_dfs



# -------------------------------------------------------------
# Run NSGA-II
# -------------------------------------------------------------
pareto_solutions, gen_dfs = nsga2(POP_SIZE, MAX_GEN)
 
with pd.ExcelWriter("/home/22EC1102/soumen/NIT-GPU_2/code/c_moo/nsga_II_constrained_panelty/5_iter/all_generations.xlsx") as writer:   #/data
    for i, df in enumerate(gen_dfs):
        df.to_excel(writer, sheet_name=f"Gen_{i}", index=False)
print("✔️ All generation data saved to 'all_generations.xlsx'")

# Print final Pareto solutions
print("\nFinal Pareto-Optimal Feasible Solutions:")
for sol in pareto_solutions:
    acc = -sol['objs'][0]
    size = sol['objs'][1]
    fpr = sol['objs'][2]
    print(f"Accuracy={acc:.3f}, Size={size:.3f} MB, FPR={fpr:.4f}, Hyperparams={sol['hparams']}")



import pandas as pd

# 1) Flatten Pareto solutions into a list of dicts
records = []
for sol in pareto_solutions:
    acc  = -sol['objs'][0]
    size = sol['objs'][1]
    fpr  = sol['objs'][2]
    hparams = sol['hparams']
    # Build a single record with all fields
    record = {
        'Accuracy': acc,
        'Size_MB': size,
        'FPR': fpr,
        # Unpack each hyperparameter into its own column:
        **hparams
    }
    records.append(record)

# 2) Create a DataFrame
df = pd.DataFrame(records)
pd.DataFrame(records).to_csv("/home/22EC1102/soumen/NIT-GPU_2/code/c_moo/nsga_II_constrained_panelty/5_iter/final_pareto.csv",index=False)  # data