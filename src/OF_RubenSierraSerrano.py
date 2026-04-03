import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# Use a non-interactive backend to avoid requiring a display/GUI
matplotlib.use("Agg")


def load_shards(folder_path):
    """
    Use:
        Reads .pt data files and creates a sliding window of signal samples. 
        It shifts data to include a historical sample (n-1), creating an 
        8-sample snapshot for each signal event.
    Args:
        folder_path (str): Path to the directory containing PyTorch shard files.
    Returns:
        tuple: (x_data, y_data)
            - x_data (np.ndarray): 2D array (Samples, 8) of signal windows.
            - y_data (np.ndarray): 1D array of target energy values.
    """
    files = sorted(glob.glob(os.path.join(folder_path, '*.pt')))
    x_final_list, y_final_list = [], []

    for f in files:
        # Load shard data onto CPU
        data = torch.load(f, map_location='cpu', weights_only=False)
        x_np = data['X'].numpy()[:, 1, :]  # Shape (M, 7) - Low Gain Channel
        y_np = data['y'].numpy()[:, 1]     # Shape (M,) - Low Gain Target

        # Vectorized window construction:
        # We need sample [n-4...n+3]. The shard provides [n-3...n+3].
        # 'prefix' takes the first sample of the previous row to act as n-4.
        prefix = x_np[:-1, 0:1]
        current_windows = x_np[1:, :]
        
        # Horizontal stack to create the 8-sample window
        windows_8 = np.hstack((prefix, current_windows))

        x_final_list.append(windows_8)
        y_final_list.append(y_np[1:])

    # combine all shards into single arrays
    return np.vstack(x_final_list), np.concatenate(y_final_list)


def eda(x_train, y_train):
    """
    Use:
        Exploratory Data Analysis. Isolates high-energy pulses to find the 
        average pulse shape (g) and low-energy pulses to calculate the 
        noise covariance matrix (R).
    Args:
        x_train (np.ndarray): Input signal windows.
        y_train (np.ndarray): True energy targets.
    Returns:
        tuple: (g_shape, g_prime, noise_cov)
            - g_shape (np.ndarray): Normalized average pulse profile.
            - g_prime (np.ndarray): Derivative of the pulse profile.
            - noise_cov (np.ndarray): Statistical noise covariance matrix.
    """
    # Use top 2% energy events to define the ideal pulse shape
    high_energy_mask = y_train > np.percentile(y_train, 98)
    x_high = x_train[high_energy_mask]

    # Baseline subtraction and normalization
    g_shape = np.mean(x_high - x_high[:, 0:1], axis=0)
    g_shape /= np.max(g_shape)
    g_prime = np.gradient(g_shape)

    # Visualization of Signal Pulse
    plt.figure(figsize=(8, 6))
    sample_indices = np.arange(-4, 4)
    plt.plot(sample_indices, g_shape, marker='o', label='Pulse Shape (g)',
             color='blue', linewidth=2)
    plt.plot(sample_indices, g_prime, marker='s', linestyle='--',
             label="Derivative (g')", color='red', linewidth=2)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title("Normalized Average Pulse Shape and Derivative")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.savefig("Images/eda_pulse_shape.png")
    plt.close()

    # Use bottom 10% energy events to characterize electronic noise
    low_energy_mask = y_train < np.percentile(y_train, 10)
    x_low = x_train[low_energy_mask]
    noise_cov = np.cov(x_low.T)

    # Visualization of Noise Correlation
    plt.figure(figsize=(8, 6))
    plt.imshow(noise_cov, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Covariance [ADC counts^2]')
    plt.title("Noise Covariance Matrix (R)\n(Electronic Noise + Pileup)")
    plt.xlabel("Sample Index i")
    plt.ylabel("Sample Index j")
    plt.savefig("Images/noise_covariance_matrix.png")
    plt.close()

    return g_shape, g_prime, noise_cov


def calculate_of3_weights(g, g_prime, noise_r):
    """
    Use:
        Derives Optimal Filter weights by solving the constrained 
        minimization problem: Minimize variance subject to 
        reconstructing amplitude, timing drift, and pedestal.
    Args:
        g (np.ndarray): Reference pulse shape.
        g_prime (np.ndarray): Pulse shape derivative.
        noise_r (np.ndarray): Noise covariance matrix.
    Returns:
        tuple: (w, v)
            - w (np.ndarray): Energy reconstruction weights.
            - v (np.ndarray): Timing reconstruction weights.
    """
    # C matrix contains constraints for [Amplitude, Time, Pedestal]
    r_inv = np.linalg.inv(noise_r)
    c_mat = np.vstack((g, g_prime, np.ones_like(g))).T
    m_mat = np.linalg.inv(c_mat.T @ r_inv @ c_mat)

    # Extract weights for Energy (w) and Time (v)
    w_weights = r_inv @ c_mat @ m_mat @ np.array([1.0, 0.0, 0.0])
    v_weights = r_inv @ c_mat @ m_mat @ np.array([0.0, -1.0, 0.0])

    print("Condition Number of R:", np.linalg.cond(noise_r))
    print("Constraint g: ", w_weights @ g)      # Should be approx 1.0
    print("Constraint g': ", w_weights @ g_prime) # Should be approx 0.0
    print("Constraint 1: ", w_weights @ np.ones_like(g)) # Should be approx 0.0

    return w_weights, v_weights


def invert_standardization(y_scaled, mean, std):
    """
    Use:
        Converts normalized/standardized energy targets back into 
        physical units (ADC counts).
    Args:
        y_scaled (np.ndarray): Normalized data.
        mean (float): Mean used for original scaling.
        std (float): Std dev used for original scaling.
    Returns:
        np.ndarray: Physical scale data.
    """
    return (y_scaled * std) + mean


def evaluate_metrics(y_true, y_pred, t_pred, dataset_name):
    """
    Use:
        Calculates residuals and creates histograms for Energy and Time 
        reconstruction accuracy.
    Args:
        y_true, y_pred (np.ndarray): True vs Predicted physical energy.
        t_pred (np.ndarray): Predicted signal time.
        dataset_name (str): Label for saved plots.
    """
    # Filter out pure noise for metric calculation
    mask = y_true > 0.1
    y_t, y_p, t_p = y_true[mask], y_pred[mask], t_pred[mask]

    # Relative Error
    residuals = (y_p - y_t) / y_t
    mean_res, rms_res = np.mean(residuals), np.std(residuals)

    print(f"\n--- {dataset_name} Energy Metrics ---")
    print(f"Mean Relative Residual: {mean_res:.5f}")
    print(f"RMS Relative Residual:  {rms_res:.5f}")
    print(f"Mean Reconstructed Time: {np.mean(t_p):.5f} BC")
    print(f"RMS Reconstructed Time:  {np.std(t_p):.5f} BC")

    # Plotting Energy Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=100, range=(-2, 2), color='darkblue', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f"{dataset_name}: Energy Relative Residual")
    plt.xlabel(r"$\frac{E_{reco} - E_{true}}{E_{true}}$")
    plt.yscale('log')
    plt.savefig(f"Images/{dataset_name}_energy_residual_hist.png")

    # Plotting Energy Correlation (2D)
    plt.figure(figsize=(10, 6))
    plt.hist2d(y_t, residuals, bins=[100, 100],
               range=[[0, np.percentile(y_t, 99)], [-2, 2]], cmap='plasma')
    plt.colorbar(label='Density')
    plt.title(f"{dataset_name}: Energy Residual vs Target")
    plt.xlabel("True Energy [ADC]")
    plt.ylabel("Relative Residual")
    plt.savefig(f"Images/{dataset_name}_energy_residual_2d.png")

    # Plotting Time Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(t_p, bins=100, range=(-2, 2), color='forestgreen', alpha=0.7)
    plt.axvline(0, color='black', linestyle='--')
    plt.title(f"{dataset_name}: Reconstructed Signal Time")
    plt.xlabel("Time [Bunch Crossings, 1 BC = 25ns]")
    plt.savefig(f"Images/{dataset_name}_time_distribution.png")

    plt.close('all')


def study_tikhonov_grid_search(g, noise_r, x_val, y_val_true):
    """
    Use:
        Finds the optimal Tikhonov regularization parameter (lambda) by 
        testing a range of values and measuring the resulting RMS error 
        on a validation set.
    Args:
        g, noise_r: Signal shape and unregularized noise matrix.
        x_val, y_val_true: Data used for tuning.
    Returns:
        float: The lambda value that minimizes residual RMS.
    """
    base_thermal_variance = np.mean(np.diag(noise_r))
    # Define Test range
    lambdas_tested = np.logspace(-8, 0, num=30) * base_thermal_variance

    conds, valid_rms = [], []
    mask_physical = y_val_true > 0.1
    y_target_evaluate = y_val_true[mask_physical]

    for lam in lambdas_tested:
        # Apply Tikhonov regularization: R_reg = R + lambda*I
        r_reg = noise_r + lam * np.eye(noise_r.shape[0])
        conds.append(np.linalg.cond(r_reg))

        r_inv = np.linalg.inv(r_reg)
        w_curr = (r_inv @ g) / (g.T @ r_inv @ g + 1e-12)

        preds_raw = x_val @ w_curr

        # Calibration for the current lambda
        cal_m = y_val_true > np.percentile(y_val_true, 90)
        if np.any(preds_raw[cal_m] > 1.0):
            k_curr = np.mean(y_val_true[cal_m] / preds_raw[cal_m])
        else:
            k_curr = 1.0

        y_p = preds_raw[mask_physical] * k_curr
        relative_residuals = (y_p - y_target_evaluate) / y_target_evaluate
        valid_rms.append(np.std(relative_residuals))

    best_idx = np.argmin(valid_rms)
    best_lam = lambdas_tested[best_idx]

    # Save Study Plot (RMS vs Lambda)
    plt.figure(figsize=(8, 6))
    plt.semilogx(lambdas_tested, valid_rms, marker='o', color='darkblue')
    plt.axvline(best_lam, color='red', linestyle='--',
                label=f'Best RMS at $\\lambda$: {best_lam:.1e}')
    plt.title("Regularization Grid Search: Validation RMS")
    plt.xlabel("Regularization Penalty $\\lambda$ (log scale)")
    plt.ylabel("Relative Residual RMS")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig("Images/study_lambda_rms.png")
    plt.close()

    # Save Study Plot (Condition Number vs Lambda)
    plt.figure(figsize=(8, 6))
    plt.loglog(lambdas_tested, conds, marker='s', color='forestgreen')
    plt.title("Matrix Stabilization: Effect of $\\lambda$")
    plt.xlabel("Regularization Penalty $\\lambda$ (log scale)")
    plt.ylabel("Condition Number (log scale)")
    plt.grid(True, which="both")
    plt.savefig("Images/study_lambda_cond.png")
    plt.close()

    print(f"Optimal Lambda: {best_lam:.2e} | Condition R_reg: {conds[best_idx]:.2f}")
    return best_lam


if __name__ == "__main__":
    # Create directory for plots if it doesn't exist
    os.makedirs("Images", exist_ok=True)

    # Load Data
    X_TRAIN, Y_TRAIN = load_shards('data/train/')
    X_VAL, Y_VAL = load_shards('data/val/')
    X_TEST, Y_TEST = load_shards('data/test/')

    # Load Stats for Target Re-scaling
    stats = np.load('data/y_stats.npz')
    y_mean = stats['mean'].flatten()[1]
    y_std = stats['std'].flatten()[1]

    print("--- STANDARD OPTIMAL FILTER ---")

    # Model Design
    g_vec, gp_vec, r_mat = eda(X_TRAIN, Y_TRAIN)
    w_final, v_final = calculate_of3_weights(g_vec, gp_vec, r_mat)

    # Calibration (Determine the k-factor to map weights to physical units)
    y_tr_true = invert_standardization(Y_TRAIN, y_mean, y_std)
    y_tr_reco_raw = X_TRAIN @ w_final
    cal_mask = y_tr_true > np.percentile(y_tr_true, 90)
    k_const = np.mean(y_tr_true[cal_mask] / y_tr_reco_raw[cal_mask])

    print(f"Calibration Constant (k): {k_const:.4f}")

    # Evaluate Standard Filter
    eval_list = [("Validation", X_VAL, Y_VAL), ("Test", X_TEST, Y_TEST)]
    for name, x_data, y_raw in eval_list:
        y_true_vals = invert_standardization(y_raw, y_mean, y_std)
        e_raw = x_data @ w_final
        e_reco = e_raw * k_const
        tau_reco = (x_data @ v_final) / (e_raw + 1e-9)

        evaluate_metrics(y_true_vals, e_reco, tau_reco, name)

    print("\n--- OPTIMAL FILTER WITH TIKHONOV REGULARIZATION ---")

    # Grid Search for Stabilization
    Y_VAL_TRUE = invert_standardization(Y_VAL, y_mean, y_std)
    lambda_opt = study_tikhonov_grid_search(g_vec, r_mat, X_VAL, Y_VAL_TRUE)

    # Apply Regularization to the Noise Matrix
    r_mat_reg = r_mat + lambda_opt * np.eye(r_mat.shape[0])

    w_reg, v_reg = calculate_of3_weights(g_vec, gp_vec, r_mat_reg)

    # Final Regularized Evaluation
    y_true_test = invert_standardization(Y_TEST, y_mean, y_std)
    e_raw_reg = X_TEST @ w_reg

    # Recalculate k specifically for regularized weights
    cal_mask_eval = y_true_test > np.percentile(y_true_test, 90)
    k_eval = np.mean(y_true_test[cal_mask_eval] / e_raw_reg[cal_mask_eval])
    print(f"Regularized Calibration Constant (k): {k_eval:.4f}")

    e_reco_reg = e_raw_reg * k_eval
    tau_reco_reg = (X_TEST @ v_reg) / (e_raw_reg + 1e-9)

    evaluate_metrics(y_true_test, e_reco_reg, tau_reco_reg, "Test_Regularized")