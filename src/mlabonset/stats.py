import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc
from joblib import Parallel, delayed


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

        if true_t0 is None:
            raise ValueError("true_t0 must be provided when simulation=True.")

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

        if true_t0 is None:
            raise ValueError("true_t0 must be provided when simulation=True.")

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
        agg.rename(columns={"mean": "mean_abs_error", "std": "std_abs_error"}, inplace=True)

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

    for pulse in pulses:
        if threshold is not None and pulse.max() <= threshold:
            continue

        results.append(
            collect_single_result(pulse, detect_func, true_t0, simulation)
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


