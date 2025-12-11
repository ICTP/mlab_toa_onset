import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
from sklearn.metrics import auc
from joblib import Parallel, delayed
from typing import Optional, List, Tuple
from .onset import DLIM


class PulseStatistics:
    """General-purpose statistical engine for time-of-arrival evaluation.

    This module works in two modes:
    
    - simulation = True:
        Ground-truth t0 is known. All supervised statistics
        (error, bias, ROC, TPR/FPR, aggregated metrics)
        are enabled.
    
    - simulation = False:
        No ground-truth exists (real data).
        Only unsupervised statistics are available:
        - method comparison
        - histograms
        - distribution analysis
        - detection validity

    All onset detection methods must be passed as callables.
    """

    def __init__(self, n_jobs: int = -2, simulation: bool = True):
        """Initialize the statistics engine.

        Args:
            n_jobs (int): Parallel job count for joblib.
            simulation (bool): 
                True  => ground truth exists; supervised stats enabled.
                False => no ground truth; supervised stats disabled.
        """
        self.n_jobs = n_jobs
        self.simulation = simulation

    # ----------------------------------------------------------------------
    # SUPERVISED ANALYSIS (only if simulation=True)
    # ----------------------------------------------------------------------

    def _require_simulation(self, func_name: str):
        """Throw a clear exception if supervised analysis is used without ground truth."""
        if not self.simulation:
            raise RuntimeError(
                f"{func_name} requires simulation=True because it depends on ground truth t0."
            )

    # ----------------------------------------------------------------------
    # SINGLE SUPERVISED EVALUATION
    # ----------------------------------------------------------------------

    def evaluate_single(self, pulse_row, detect_func, true_t0=None, **kwargs):
        """Evaluate a single pulse with any detection function.

        Args:
            pulse_row (dict or pandas.Series): Must contain key "pulse".
            detect_func (callable): t0 estimator function: detect_func(pulse, **kwargs).
            true_t0 (float): Ground-truth time-of-arrival.

        Returns:
            dict: error metrics + t0 estimation.
        """
        self._require_simulation("evaluate_single")

        if not isinstance(pulse_row, (dict, pd.Series)):
            raise TypeError(
                f"pulse_row must be dict or pd.Series with 'pulse' field. "
                f"Received type: {type(pulse_row).__name__}"
            )
        
        # Check for 'pulse' field
        if "pulse" not in pulse_row:
            available_keys = list(pulse_row.keys()) if hasattr(pulse_row, 'keys') else []
            raise ValueError(
                "Dataset row must contain a 'pulse' field. Expected structure:\n"
                "  {\n"
                "    't0': float,\n"
                "    'amplitude': float,\n"
                "    'tau': float,\n"
                "    'snr': float,\n"
                "    'pulse': np.ndarray\n"
                "  }\n"
                f"Received keys: {available_keys}"
            )
        
        pulse = pulse_row["pulse"]
        if true_t0 is None:
            true_t0 = pulse_row["t0"]
        
        t0_est = detect_func(pulse, **kwargs)
        
        valid = np.isfinite(t0_est) and t0_est > 0
        if not valid:
            return {
                "t0_est": np.nan,
                "absolute_error": np.nan,
                "signed_error": np.nan,
                "normalized_error": np.nan,
                "valid": False,
            }

        abs_err = np.abs(t0_est - true_t0)
        signed_err = t0_est - true_t0
        norm_err = signed_err

        return {
            "t0_est": t0_est,
            "absolute_error": abs_err,
            "signed_error": signed_err,
            "normalized_error": norm_err,
            "valid": True,
        }

    # ----------------------------------------------------------------------
    # SUPERVISED DATASET EVALUATION
    # ----------------------------------------------------------------------

    def evaluate_dataset(self, dataset, detect_func, true_t0=None, **kwargs):
        """Evaluate a full dataset using joblib in parallel.

        Returns:
            pandas.DataFrame: dataset + supervised statistics.
        """
        self._require_simulation("evaluate_dataset")

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.evaluate_single)(row, detect_func, true_t0, **kwargs)
            for _, row in dataset.iterrows()
        )

        stats_df = pd.DataFrame(results)
        return pd.concat([dataset.reset_index(drop=True), stats_df], axis=1)

    # ----------------------------------------------------------------------
    # SUPERVISED AGGREGATION
    # ----------------------------------------------------------------------

    def aggregated_metrics(self, evaluated_df, groupby_fields):
        """Aggregate supervised metrics by parameter groups."""
        self._require_simulation("aggregated_metrics")

        valid_mask = evaluated_df["valid"] == True
        df_valid = evaluated_df[valid_mask]

        grouped = df_valid.groupby(groupby_fields)

        agg = grouped["absolute_error"].agg(["mean", "std"]).reset_index()
        agg.rename(columns={"mean": "mean", "std": "std"}, inplace=True)

        TP = grouped.size().reset_index(name="TP")
        total = evaluated_df.groupby(groupby_fields).size().reset_index(name="total")

        merged = agg.merge(TP, on=groupby_fields).merge(total, on=groupby_fields)
        merged["FP"] = merged["total"] - merged["TP"]

        return merged

    # ----------------------------------------------------------------------
    # SUPERVISED GRID EVALUATION
    # ----------------------------------------------------------------------

    def evaluate_param_grid(self, dataset, detect_func, true_t0=None, param_grid=None, **kwargs):
        """Evaluate a t0 detector over a grid of parameters."""
        self._require_simulation("evaluate_param_grid")

        if true_t0 is None:
            raise ValueError("true_t0 must be provided when simulation=True.")

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        def recursive_iter(params, idx):
            if idx == len(keys):
                yield dict(zip(keys, params))
                return
            for v in values[idx]:
                yield from recursive_iter(params + [v], idx + 1)

        all_results = []

        for combo in recursive_iter([], 0):
            df = dataset.copy()
            for k, v in combo.items():
                df[k] = v

            result_df = self.evaluate_dataset(
                df,
                detect_func,
                true_t0,
                **{k: v for k, v in combo.items() if True},
            )

            all_results.append(result_df)

        return pd.concat(all_results, ignore_index=True)
    
    def compute_tpr_fpr(self, results, true_t0, threshold_list):
        self._require_simulation("compute_tpr_fpr")

        tpr_list, fpr_list = [], []

        for thr in threshold_list:
            TP = np.sum(
                (results["valid"] == True) &
                (np.abs(results["t0_est"] - true_t0) <= thr)
            )

            FP = np.sum(
                (results["valid"] == True) &
                (np.abs(results["t0_est"] - true_t0) > thr)
            )

            P = np.sum(results["valid"] == True)
            N = np.sum(results["valid"] == False)

            tpr = TP / P if P > 0 else 0
            fpr = FP / (FP + N) if (FP + N) > 0 else 0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        return np.array(fpr_list), np.array(tpr_list)

    def compute_roc_auc(self, results, true_t0, thresholds):
        self._require_simulation("compute_roc_auc")

        fpr, tpr = self.compute_tpr_fpr(results, true_t0, thresholds)
        roc_auc = auc(fpr, tpr)

        return {
            "fpr": fpr,
            "tpr": tpr,
            "auc": roc_auc,
            "thresholds": thresholds,
        }

    # ----------------------------------------------------------------------
    # ROC PLOTTING (SUPERVISED ONLY)
    # ----------------------------------------------------------------------

    @staticmethod
    def plot_roc_curve(roc_dict, method_name="method"):
        plt.figure(figsize=(8, 6))
        plt.plot(roc_dict["fpr"], roc_dict["tpr"],
                 label=f"{method_name} (AUC={roc_dict['auc']:.3f})", linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {method_name}")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_multi_roc(roc_dicts):
        plt.figure(figsize=(8, 6))
        for name, roc in roc_dicts.items():
            plt.plot(roc["fpr"], roc["tpr"], label=f"{name} (AUC={roc['auc']:.3f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Comparison of Methods")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()


def plot_overlaid_histograms(df, column, hue, bins=200, xlim=None, title=None):
    """Plot overlaid histograms of arrival times or errors."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column, hue=hue, bins=bins, alpha=0.4)
    if xlim:
        plt.xlim(*xlim)
    plt.grid()
    plt.title(title if title else f"Histogram of {column}")
    plt.tight_layout()
    plt.show()


def collect_single_result(pulse, detect_func, true_t0=None, simulation=True):
    """Collect a t0 result in supervised or unsupervised mode."""
    x0, t0 = detect_func(pulse)

    if not np.isfinite(t0) or t0 <= 0:
        return {
            "t0_est": np.nan,
            "signed_error": np.nan,
            "absolute_error": np.nan,
            "valid": False,
        }

    result = {
        "t0_est": t0,
        "valid": True,
        "signed_error": np.nan,
        "absolute_error": np.nan,
    }

    if simulation and true_t0 is not None:
        signed = t0 - true_t0
        result["signed_error"] = signed
        result["absolute_error"] = abs(signed)

    return result


def scan_dataset(pulses, detect_func, true_t0=None, simulation=True, threshold=None):
    """Unsupervised or supervised scanning of pulses."""
    results = []

    for i, pulse in enumerate(pulses):
        if threshold is not None and pulse.max() <= threshold:
            continue
        results.append(
            collect_single_result(pulse, detect_func, true_t0 if type(true_t0)==int else true_t0[i], simulation)
        )

    return pd.DataFrame(results)


def compare_methods(pulses, methods_dict, true_t0=None, simulation=True, threshold=None):
    """Compare multiple detection methods on the same dataset."""
    all_results = []

    for name, method in methods_dict.items():
        df = scan_dataset(
            pulses,
            method,
            true_t0=true_t0,
            simulation=simulation,
            threshold=threshold,
        )
        df["method"] = name
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)

def _validate_array_length(
    reference_array: np.ndarray,
    target_array: Optional[np.ndarray],
    array_name: str
) -> np.ndarray:
    """
    Validate that target array matches reference array length or initialize empty array.
    
    Args:
        reference_array: Array to compare length against
        target_array: Array to validate (or None)
        array_name: Name of target array for error messages
        
    Returns:
        Validated array or empty list if target_array is None
        
    Raises:
        ValueError: If arrays have mismatched lengths
    """
    if target_array is None:
        return np.array([])
    
    if len(reference_array) != len(target_array):
        raise ValueError(
            f"{array_name} length ({len(target_array)}) must match "
            f"pulses_array length ({len(reference_array)})"
        )
    
    return target_array


def _compute_offset_from_indices(
    pulses_array: np.ndarray,
    offset_indices: Tuple[int, int]
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute offset, SNR, and log SNR from pulse data at specified indices.
    
    Args:
        pulses_array: Array of pulse signals
        offset_indices: Tuple of (start_index, end_index) for offset calculation
        
    Returns:
        Tuple of (offset_array, snr_array, log_snr_array)
    """
    offset_list = []
    snr_list = []
    log_snr_list = []
    
    for pulse in pulses_array:
        offset_region = pulse[offset_indices[0]:offset_indices[1]]
        offset_mean = np.mean(offset_region)
        offset_std = np.std(offset_region)
        
        offset_list.append(offset_mean)
        snr_list.append(offset_std)
        log_snr_list.append(20 * np.log10(offset_mean) if offset_mean > 0 else -np.inf)
    
    return offset_list, snr_list, log_snr_list


def _compute_amplitudes(
    pulses_array: np.ndarray,
    offset_array: np.ndarray
) -> np.ndarray:
    """
    Compute pulse amplitudes as max value minus offset.
    
    Args:
        pulses_array: Array of pulse signals
        offset_array: Array of offset values for each pulse
        
    Returns:
        Array of computed amplitudes
    """
    return np.array([
        np.max(pulse) - offset
        for pulse, offset in zip(pulses_array, offset_array)
    ])


def label_and_create_dataset(
    pulses_array: np.ndarray,
    t0_array: Optional[np.ndarray] = None,
    amplitude_array: Optional[np.ndarray] = None,
    snr_array: Optional[np.ndarray] = None,
    log_snr_array: Optional[np.ndarray] = None,
    tau_array: Optional[np.ndarray] = None,
    offset_array: Optional[np.ndarray] = None,
    round_decimals: int = 4
) -> pd.DataFrame:
    """
    Create a labeled dataset from pulse arrays and associated parameters.
    
    Args:
        pulses_array: Array of pulse signals (required)
        t0_array: Time zero values (scalar or array matching pulses_array length)
        amplitude_array: Pulse amplitude values
        snr_array: Signal-to-noise ratio values (linear scale)
        log_snr_array: Signal-to-noise ratio values (dB scale)
        tau_array: Time constant values (scalar or array)
        offset_array: Offset values OR tuple of (start_idx, end_idx) for auto-calculation
        round_decimals: Number of decimal places for rounding
        
    Returns:
        DataFrame with labeled pulse data including all parameters
        
    Raises:
        ValueError: If array lengths don't match or inputs are invalid
        
    Examples:
        >>> pulses = np.array([np.random.randn(100) for _ in range(10)])
        >>> df = label_and_establish_dataset(
        ...     pulses_array=pulses,
        ...     t0_array=np.array([0.0]),
        ...     tau_array=np.array([1.0])
        ... )
    """
    if pulses_array is None or len(pulses_array) == 0:
        raise ValueError("pulses_array cannot be None or empty")
    
    n_pulses = len(pulses_array)
    
    # Validate and prepare arrays
    snr_array = _validate_array_length(pulses_array, snr_array, "snr_array")
    log_snr_array = _validate_array_length(pulses_array, log_snr_array, "log_snr_array")
    
    # Handle scalar or array inputs for t0 and tau
    if t0_array is None or len(t0_array) == 0:
        t0_array = np.zeros(n_pulses)
    elif len(t0_array) == 1:
        t0_array = np.full(n_pulses, t0_array[0])
    else:
        t0_array = _validate_array_length(pulses_array, t0_array, "t0_array")
    
    if tau_array is None or len(tau_array) == 0:
        tau_array = np.ones(n_pulses)
    elif len(tau_array) == 1:
        tau_array = np.full(n_pulses, tau_array[0])
    else:
        tau_array = _validate_array_length(pulses_array, tau_array, "tau_array")
    
    # Handle offset calculation or validation
    if offset_array is not None and len(offset_array) == 2:
        # Interpret as indices for offset calculation
        offset_list, snr_calc, log_snr_calc = _compute_offset_from_indices(
            pulses_array, tuple(offset_array)
        )
        offset_array = np.array(offset_list)
        
        # Use calculated values if not provided
        if len(snr_array) == 0:
            snr_array = np.array(snr_calc)
        if len(log_snr_array) == 0:
            log_snr_array = np.array(log_snr_calc)
    else:
        offset_array = _validate_array_length(pulses_array, offset_array, "offset_array")
        if len(offset_array) == 0:
            offset_array = np.zeros(n_pulses)
    
    # Handle amplitude calculation or validation
    amplitude_array = _validate_array_length(pulses_array, amplitude_array, "amplitude_array")
    if len(amplitude_array) == 0:
        amplitude_array = _compute_amplitudes(pulses_array, offset_array)
    
    # Ensure SNR arrays are populated
    if len(snr_array) == 0:
        snr_array = np.ones(n_pulses)
    if len(log_snr_array) == 0:
        log_snr_array = np.zeros(n_pulses)
    
    # Build dataset records
    records = [
        {
            "t0": round(t0_array[i], round_decimals),
            "amplitude": round(amplitude_array[i], round_decimals),
            "tau": round(tau_array[i], round_decimals),
            "snr_dB": round(log_snr_array[i], round_decimals),
            "snr": round(snr_array[i], round_decimals),
            "offset": round(offset_array[i], round_decimals),
            "pulse": pulses_array[i]
        }
        for i in range(n_pulses)
    ]
    
    return pd.DataFrame(records)

def plot_relacion_acc_pre(data, ylim_acc, ylim_pre, anotation=False, save_plot =False):
    sns.set_context("paper", font_scale=1.3)
    xaxis='w'
    # Crear figura con subgráficos
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Gráfico de Accuracy
    sns.lineplot(ax=axes[0], x=xaxis, y="mean", data=data, marker='o')
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(ylim_acc)
    axes[0].grid(False)
    axes[0].tick_params(direction='in', top=True, right=True, bottom=True, left=True)
    if anotation == True:
        #print(data["cfir_maer"][-1:])
        last_point_std = data.iloc[-1]
        axes[0].annotate(f'{last_point_std["mean"]:.2f}', 
             xy=(last_point_std['w'], 3), 
             xytext=(last_point_std['w'], 3 - 0.55),
             arrowprops=dict(facecolor='blue', shrink=0.05),
             horizontalalignment='center', verticalalignment='bottom')

    # Gráfico de Precision
    sns.lineplot(ax=axes[1], x=xaxis, y="std", data=data, marker='o', color='r')
    axes[1].set_ylabel("Precision")
    axes[1].set_xlabel('W')
    axes[1].set_xticks(data[xaxis])
    axes[1].set_xticklabels([f"{val}" for val in data[xaxis]])
    axes[1].set_ylim(ylim_pre)
    axes[1].grid(False)
    axes[1].tick_params(direction='in', top=True, right=True, bottom=True, left=True)
    
    plt.tight_layout()
    plt.show()
    if save_plot:
        plt.savefig("acc_pres_plot.png")
        
def dlim_window_scan(dataset, max_window, scale_acc = (0,1), scale_pres=(0,1)):
    resultados = []
    stats = PulseStatistics(simulation=True)
    dlimM = DLIM(4)
    for window in range(4, max_window+2, 2):
        res = stats.evaluate_dataset(
            dataset, detect_func=lambda sig: float(dlimM.t0_get(sig, window)[1])
        )
        res["w"] = window
        resultados.append(res)
    resultado = pd.concat(resultados, ignore_index=True)
    agg2 = stats.aggregated_metrics(resultado, groupby_fields=['w'])
    plot_relacion_acc_pre(agg2, scale_acc, scale_pres)

def histo_con_boxplot(df, variable):
    COLOR_HIST = "#4A90E2"
    COLOR_KDE  = "#D0021B"
    COLOR_BOX  = "#4A90E2"
    data = df[variable].dropna()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
    counts, bin_edges, patches = ax1.hist(
        data,
        bins='fd',
        color=COLOR_HIST,
        alpha=0.65,
        edgecolor="black"
    )

    kde = gaussian_kde(data)
    x_vals = np.linspace(data.min(), data.max(), 400)
    ax1.plot(x_vals, kde(x_vals)*len(data)*(bin_edges[1]-bin_edges[0]),
             color=COLOR_KDE, linewidth=2.2, label="KDE")
    ax1.set_title(f"Histograma + KDE: {variable}")
    ax1.set_xlabel(variable)
    ax1.set_ylabel("Frecuencia")
    
    stats = data.describe()
    textstr = (
        f"N = {int(stats['count'])}\n"
        f"Mean = {stats['mean']:.2f}\n"
        f"Std = {stats['std']:.2f}\n"
        f"Min = {stats['min']:.2f}\n"
        f"Q1 = {stats['25%']:.2f}\n"
        f"Median = {stats['50%']:.2f}\n"
        f"Q3 = {stats['75%']:.2f}\n"
        f"Max = {stats['max']:.2f}"
    )

    props = dict(boxstyle="round", facecolor="white", alpha=0.9)
    ax1.text(0.97, 0.97, textstr, transform=ax1.transAxes,
             fontsize=11, verticalalignment="top",
             horizontalalignment="right", bbox=props)
    ax2.boxplot(
        data,
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor=COLOR_BOX, color="black", alpha=0.6),
        medianprops=dict(color="yellow", linewidth=2),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(marker='o', markersize=4, markerfacecolor="red", alpha=0.4)
    )
    ax2.set_title(f"Boxplot: {variable}")
    ax2.set_ylabel(variable)

    plt.tight_layout()
    plt.show()

    return stats
