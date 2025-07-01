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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, WhiteKernel
from sklearn.metrics import confusion_matrix

# Keras imports for building the CNN
from tensorflow.keras import layers, models, Model
from tensorflow.keras.callbacks import EarlyStopping

# pyDOE for Latin Hypercube Sampling
# You may need to install it: pip install pyDOE
from pyDOE import lhs

# --- GPU Configuration ---
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


# --- Data Loading & Preparation ---
def load_data(data_path):
    """Loads pre-split training, validation, and test datasets from .npy files."""
    print("Loading dataset...")
    X_train = np.load(f'{data_path}/X_train.npy')
    X_test = np.load(f'{data_path}/X_test.npy')
    X_validation = np.load(f'{data_path}/X_val.npy')
    y_train = np.load(f'{data_path}/y_train.npy')
    y_test = np.load(f'{data_path}/y_test.npy')
    y_validation = np.load(f'{data_path}/y_val.npy')
    # Add a channel dimension for CNN compatibility
    X_train, X_test, X_validation = X_train[..., np.newaxis], X_test[..., np.newaxis], X_validation[..., np.newaxis]
    y_train, y_test, y_validation = y_train[..., np.newaxis], y_test[..., np.newaxis], y_validation[..., np.newaxis]
    print("Dataset loaded!")
    return X_train, X_test, X_validation, y_train, y_test, y_validation

def prepare_dataset(data_path):
    """Loads and scales the features of the dataset."""
    X_train, X_test, X_validation, y_train, y_test, y_validation = load_data(data_path)
    scaler = StandardScaler()
    # Fit on training data and transform all splits
    n_train, t_steps, n_feats, _ = X_train.shape
    scaler.fit(X_train.reshape(-1, n_feats))
    for X_split in [X_train, X_test, X_validation]:
        n, t, f, _ = X_split.shape
        X_flat = X_split.reshape(-1, f)
        X_split[:] = scaler.transform(X_flat).reshape(n, t, f, 1)
    return X_train, y_train, X_validation, y_validation, X_test, y_test

# --- Global Constants and Hyperparameters ---
DATA_PATH = "/home/ec.gpu/Desktop/Soumen/Dataset/kws/data_10_wav/data_npy_3000"
X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_dataset(DATA_PATH)
print(f"Data shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")

CLASSES = 10
EPOCHS = 300
BATCH_SIZE = 64
PATIENCE = 5

# --- NSGA-II Search Space and Parameters ---
HPARAM_SPACE = {
    'filters': [16, 32, 64], 'kernel_size': [3, 5], 'use_bn': [True, False],
    'residual_blocks': [1, 2, 3], 'fc_layers': [1, 2, 3, 4], 'use_dropout': [True, False]
}
POP_SIZE = 15
MAX_GEN = 30
INFILL_PERCENT = 0.334
CROSSOVER_PROB = 0.9
MUTATION_PROB = 0.2
EPSILON = 1e-6

# --- Constraint and Penalty Settings ---
MIN_ACCURACY = 0.90
MAX_MODEL_SIZE = 2.5
MAX_FPR = 0.09
LAMBDA_INITIAL = 1.0
LAMBDA_FINAL = 50.0

def get_lambda(gen):
    """Calculates the adaptive penalty parameter lambda for a given generation."""
    frac = gen / float(MAX_GEN - 1) if MAX_GEN > 1 else 1.0
    return LAMBDA_INITIAL + frac * (LAMBDA_FINAL - LAMBDA_INITIAL)

# --- CNN Model Builder & Evaluator ---
def build_model(hparams):
    """Builds a custom residual CNN based on a dictionary of hyperparameters."""
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    hp_values = [hparams[k] for k in sorted(HPARAM_SPACE.keys())]
    filters, num_fc, ks, num_res, use_bn, use_drop = hp_values

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters, ks, padding='same', activation='relu')(inputs)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)

    for _ in range(num_res):
        res_filters = filters * 2
        skip = layers.Conv2D(res_filters, 1, strides=2, padding='same')(x)
        y = layers.Conv2D(res_filters, ks, padding='same', activation='relu')(x)
        if use_bn: y = layers.BatchNormalization()(y)
        y = layers.MaxPooling2D((2, 2), padding='same')(y)
        x = layers.add([y, skip])
        x = layers.ReLU()(x)
        filters = res_filters
        
    x = layers.GlobalAveragePooling2D()(x)
    fc_configs = {1: [64], 2: [128, 64], 3: [256, 128, 64], 4: [512, 256, 128, 64]}
    if num_fc in fc_configs:
        for units in fc_configs[num_fc]:
            x = layers.Dense(units, activation='relu')(x)
            if use_drop: x = layers.Dropout(0.3)(x)
            
    outputs = layers.Dense(CLASSES, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

def compute_model_size_mb(model):
    """Calculates the model's size in megabytes."""
    return (model.count_params() * 4) / (1024 ** 2)

def calculate_fpr(y_true, y_pred, num_classes):
    """Calculates the macro-averaged False Positive Rate."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fp_sum = cm.sum(axis=0) - np.diag(cm)
    tn_sum = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))
    fpr_vals = [fp / (fp + tn) if (fp + tn) > 0 else 0.0 for fp, tn in zip(fp_sum, tn_sum)]
    return np.mean(fpr_vals)

def evaluate_individual(hparams):
    """The 'true' expensive evaluation function. Builds, trains, and evaluates a model."""
    tf.keras.backend.clear_session()
    model = build_model(hparams)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), 
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=0)
    
    _, accuracy = model.evaluate(X_validation, y_validation, verbose=0)
    y_pred = np.argmax(model.predict(X_validation, verbose=0), axis=1)
    fpr = calculate_fpr(y_validation.flatten(), y_pred, CLASSES)
    size_mb = compute_model_size_mb(model)
    
    print(f"  -> True Eval: Acc={accuracy:.4f}, Size={size_mb:.2f}MB, FPR={fpr:.4f}")
    return accuracy, size_mb, fpr

def compute_objectives_and_constraints(population_hparams):
    """Wrapper to evaluate a list of hyperparameter sets and structure the results."""
    results = []
    for hparams in population_hparams:
        acc, size_mb, fpr = evaluate_individual(hparams)
        f1, f2, f3 = -acc, size_mb, fpr
        g1 = max(0.0, MIN_ACCURACY - acc)
        g2 = max(0.0, size_mb - MAX_MODEL_SIZE)
        g3 = max(0.0, fpr - MAX_FPR)
        results.append({'hparams': hparams, 'objs': [f1, f2, f3], 'CV': g1 + g2 + g3})
    return results

# --- Surrogate Model Manager ---
class SurrogateManager:
    """Manages the Gaussian Process surrogate models for each objective and the CV."""
    def __init__(self):
        self.is_fitted = False
        self.categorical_features = ['use_bn', 'use_dropout']
        self.numerical_features = ['filters', 'kernel_size', 'residual_blocks', 'fc_layers']
        self.preprocessor = ColumnTransformer(
            transformers=[('num', 'passthrough', self.numerical_features),
                          ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features)],
            remainder='passthrough'
        )
        kernel = C(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1)
        self.models = {k: GaussianProcessRegressor(kernel=deepcopy(kernel), n_restarts_optimizer=10) for k in ['neg_acc', 'size', 'fpr', 'cv']}
        self.scalers = {k: StandardScaler() for k in self.models}
        self.training_data = pd.DataFrame()

    def _prepare_X_y(self, hparams_list, results_list):
        X_df = pd.DataFrame(hparams_list)
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
        X_transformed = self.preprocessor.transform(X_df)
        predictions, stds = {}, {}
        for key, model in self.models.items():
            pred_scaled, std_scaled = model.predict(X_transformed, return_std=True)
            pred_scaled, std_scaled = pred_scaled.reshape(-1, 1), std_scaled.reshape(-1, 1)
            predictions[key] = self.scalers[key].inverse_transform(pred_scaled).flatten()
            stds[key] = std_scaled.flatten() * np.sqrt(self.scalers[key].var_) if self.scalers[key].var_ > 0 else np.zeros_like(std_scaled.flatten())
        return (predictions, stds) if return_std else predictions
    
    def predict_and_structure(self, hparams_list):
        preds, _ = self.predict(hparams_list, return_std=True)
        return [{'hparams': h, 'objs': [preds[k][i] for k in ['neg_acc', 'size', 'fpr']], 'CV': max(0, preds['cv'][i])}
                for i, h in enumerate(hparams_list)]

# --- NSGA-II Core Helpers ---
def latin_hypercube_initialization(pop_size, hparam_space):
    """Generates a well-distributed initial population using Latin Hypercube Sampling."""
    print("Initializing population with Latin Hypercube Sampling...")
    keys = sorted(hparam_space.keys())
    dims = len(keys)
    unit_lhs = lhs(dims, samples=pop_size, criterion='maximin')
    population = []
    for row in unit_lhs:
        h_sample = {}
        for i, key in enumerate(keys):
            options = hparam_space[key]
            choice_index = int(np.floor(row[i] * len(options)))
            h_sample[key] = options[min(choice_index, len(options)-1)]
        population.append(h_sample)
    print(f"Initialized {pop_size} individuals using LHS.")
    return population

def dominates(a, b, lam):
    """Penalized dominance check."""
    Pa = [f + lam * a['CV'] for f in a['objs']]
    Pb = [f + lam * b['CV'] for f in b['objs']]
    better_all = all(pa <= pb for pa, pb in zip(Pa, Pb))
    strictly_better = any(pa < pb for pa, pb in zip(Pa, Pb))
    return better_all and strictly_better

def fast_non_dominated_sort(results, lam):
    """Sorts the population into Pareto fronts using penalized dominance."""
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
    """Calculates the crowding distance for each individual in a front."""
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
    for key, options in HPARAM_SPACE.items():
        if random.random() < MUTATION_PROB:
            ind[key] = random.choice(options) if not isinstance(options[0], bool) else not ind[key]
    return ind
    
# --- Local Search and Infill Selection ---
def perturb_hparams(hparams):
    """Slightly changes a single hyperparameter to explore a neighbor."""
    hparams_new = deepcopy(hparams)
    param_to_perturb = random.choice(list(HPARAM_SPACE.keys()))
    current_value = hparams_new[param_to_perturb]
    options = HPARAM_SPACE[param_to_perturb]
    if isinstance(options[0], bool):
        hparams_new[param_to_perturb] = not current_value
    else:
        possible_values = [v for v in options if v != current_value]
        if possible_values: hparams_new[param_to_perturb] = random.choice(possible_values)
    return hparams_new

def lcb_dominates(sol_a, sol_b):
    """Dominance check based on the Lower Confidence Bound (LCB) of objectives."""
    return all(a <= b for a, b in zip(sol_a['lcb_objs'], sol_b['lcb_objs'])) and \
           any(a < b for a, b in zip(sol_a['lcb_objs'], sol_b['lcb_objs']))

def perform_local_search(offspring_data, surrogate_manager, k_lcb=1.0):
    """Performs a local search on promising offspring to refine them using surrogate predictions."""
    print("Performing surrogate-based local search...")
    for sol in offspring_data:
        means, stds = np.array(sol['objs']), np.array(sol['stds'])
        sol['lcb_objs'] = (means - k_lcb * stds).tolist()

    elite_indices = [i for i, sol in enumerate(offspring_data) if not any(lcb_dominates(other, sol) for other in offspring_data if other != sol)]
    
    for _ in range(5): # 5 local search iterations
        for idx in elite_indices:
            original_sol = offspring_data[idx]
            perturbed_hparams = perturb_hparams(original_sol['hparams'])
            
            pred_p, std_p = surrogate_manager.predict([perturbed_hparams], return_std=True)
            lcb_p = {k: pred_p[k][0] - k_lcb * std_p[k][0] for k in pred_p}
            
            perturbed_sol_data = {'lcb_objs': [lcb_p[k] for k in ['neg_acc', 'size', 'fpr']]}
            
            if lcb_dominates(perturbed_sol_data, original_sol):
                offspring_data[idx]['hparams'] = perturbed_hparams
                offspring_data[idx]['lcb_objs'] = perturbed_sol_data['lcb_objs']
                offspring_data[idx]['objs'] = [pred_p[k][0] for k in ['neg_acc', 'size', 'fpr']]
                offspring_data[idx]['stds'] = [std_p[k][0] for k in ['neg_acc', 'size', 'fpr']]
                
    return [sol['hparams'] for sol in offspring_data]

def select_infill_points(predicted_offspring_data, num_to_select):
    """Selects the best individuals from offspring for true evaluation."""
    feasible = [ (i, res) for i, res in enumerate(predicted_offspring_data) if res['CV'] < EPSILON ]
    infeasible = [ (i, res) for i, res in enumerate(predicted_offspring_data) if res['CV'] >= EPSILON ]

    if feasible:
        objectives = np.array([res['objs'] for _, res in feasible])
        min_o, max_o = objectives.min(axis=0), objectives.max(axis=0)
        range_o = np.maximum(max_o - min_o, EPSILON)
        scores = ((objectives - min_o) / range_o).sum(axis=1)
        feasible_sorted = [idx for idx, _ in sorted(zip([f[0] for f in feasible], scores), key=lambda p: p[1])]
    else:
        feasible_sorted = []
        
    infeasible_sorted = [idx for idx, _ in sorted(infeasible, key=lambda item: item[1]['CV'])]
    
    combined_indices = feasible_sorted + infeasible_sorted
    selected_indices = combined_indices[:num_to_select]
    
    return selected_indices, [predicted_offspring_data[i]['hparams'] for i in selected_indices]
    
# --- Main Surrogate-Assisted Memetic NSGA-II Loop ---
def nsga2(pop_size, max_gen, infill_percent):
    """Main optimization loop combining NSGA-II with LHS, Surrogates, and Local Search."""
    population_hparams = latin_hypercube_initialization(pop_size, HPARAM_SPACE)
    pop_data = compute_objectives_and_constraints(population_hparams)
    
    surrogate_manager = SurrogateManager()
    surrogate_manager.update([d['hparams'] for d in pop_data], pop_data)
    
    all_gen_dfs = []
    total_start_time = time.perf_counter()

    for gen in range(max_gen):
        gen_start_time = time.perf_counter()
        lam = get_lambda(gen)
        print(f"\n--- Generation {gen+1}/{max_gen}, Lambda={lam:.2f} ---")

        parents_indices = [tournament_selection(pop_data, lam) for _ in range(pop_size)]
        parent_hparams = [pop_data[i]['hparams'] for i in parents_indices]
        
        offspring_hparams = []
        while len(offspring_hparams) < pop_size:
            p1, p2 = random.sample(parent_hparams, 2)
            c1, c2 = crossover(p1, p2) if random.random() < CROSSOVER_PROB else (deepcopy(p1), deepcopy(p2))
            offspring_hparams.extend([mutate(c1), mutate(c2)])
        offspring_hparams = offspring_hparams[:pop_size]

        print(f"Evaluating {len(offspring_hparams)} offspring with surrogates...")
        preds, stds = surrogate_manager.predict(offspring_hparams, return_std=True)
        off_data_predicted = [{'hparams': hp, 'objs': [preds[k][i] for k in ['neg_acc', 'size', 'fpr']],
                               'stds': [stds[k][i] for k in ['neg_acc', 'size', 'fpr']], 'CV': max(0, preds['cv'][i])}
                              for i, hp in enumerate(offspring_hparams)]
        
        improved_offspring_hparams = perform_local_search(off_data_predicted, surrogate_manager)
        off_data_final_predicted = surrogate_manager.predict_and_structure(improved_offspring_hparams)

        num_infill = max(1, int(pop_size * infill_percent))
        print(f"Selecting {num_infill} infill points for true evaluation...")
        infill_indices, infill_hparams = select_infill_points(off_data_final_predicted, num_infill)
        
        infill_data_true = compute_objectives_and_constraints(infill_hparams)
        surrogate_manager.update(infill_hparams, infill_data_true)
        
        off_data = list(off_data_final_predicted)
        for i, true_res in enumerate(infill_data_true):
            off_data[infill_indices[i]] = true_res

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
        
        gen_time = time.perf_counter() - gen_start_time
        print(f"Generation {gen+1} finished in {gen_time:.2f} seconds.")
        
        # Logging
        gen_records = [{'Generation': gen + 1, 'Accuracy': -ind['objs'][0], 'Size_MB': ind['objs'][1], 
                        'FPR': ind['objs'][2], 'CV': ind['CV'], **ind['hparams']} for ind in pop_data]
        all_gen_dfs.append(pd.DataFrame(gen_records))
        feasibles = [ind for ind in pop_data if ind['CV'] == 0]
        frac_feas = len(feasibles) / len(pop_data) if pop_data else 0
        avg_cv = np.mean([ind['CV'] for ind in pop_data]) if pop_data else 0
        print(f"  -> Status: Fraction Feasible={frac_feas:.2f}, Avg CV={avg_cv:.4f}")

    total_time = time.perf_counter() - total_start_time
    print(f"\n--- NSGA-II Finished in {total_time/60:.2f} minutes ---")
    
    final_feasibles = [ind for ind in pop_data if ind['CV'] == 0]
    if not final_feasibles: return [], all_gen_dfs
    
    final_fronts = fast_non_dominated_sort(final_feasibles, LAMBDA_FINAL)
    if not final_fronts: return [], all_gen_dfs
        
    return [final_feasibles[i] for i in final_fronts[0]], all_gen_dfs

# --- Run Experiment ---
if __name__ == "__main__":
    pareto_solutions, all_gen_dfs = nsga2(POP_SIZE, MAX_GEN, INFILL_PERCENT)
    output_dir = "init_sa_nsga_local_results"
    os.makedirs(output_dir, exist_ok=True)
    
    excel_path = os.path.join(output_dir, "all_generations_memetic.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        for i, df in enumerate(all_gen_dfs):
            df.to_excel(writer, sheet_name=f"Gen_{i+1}", index=False)
    print(f"\n✔️ All generation data saved to '{excel_path}'")

    print("\n--- Final Pareto-Optimal Feasible Solutions ---")
    if pareto_solutions:
        pareto_records = [{'Accuracy': -s['objs'][0], 'Size_MB': s['objs'][1], 'FPR': s['objs'][2], **s['hparams']} for s in pareto_solutions]
        for r in pareto_records:
            print(f"  - Acc={r['Accuracy']:.4f}, Size={r['Size_MB']:.3f}MB, FPR={r['FPR']:.4f}, HParams={r['hparams']}")
        
        pd.DataFrame(pareto_records).to_csv(os.path.join(output_dir, "final_pareto_memetic.csv"), index=False)
        print(f"✔️ Final Pareto front saved.")
    else:
        print("No Pareto-optimal solutions were found.")
