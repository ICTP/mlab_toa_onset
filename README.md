# Pulse Onset Mlab

## Pulse Simulation • Onset Detection • Statistical Analysis

mlabonset is a modular library for waveform simulation, time-of-arrival (ToA/onset) detection, and statistical analysis of pulse-based signals.
It is designed for use in physics experiments, radiation detectors, ADC signal processing, and any application that relies on onset timing extraction.

## Key Features

### Pulse Simulator (simulator)
- Semi-Gaussian pulse generation
- Ramp and angular-ramp signals
- Synthetic dataset generation with noise
- Joblib-powered parallel dataset creation
- Configurable amplitude, tau, SNR, offset, and $t_0$
- Ideal for evaluating detector performance

### Onset Detection (onset)

Includes 3 widely used onset estimation algorithms:

| Method   | Description                         |
| -------- | ----------------------------------- |
| **DLIM** | FIR 2nd-derivative crossover method |
| **DCFD**  | Constant Fraction Discriminator     |
| **DLED** | Leading Edge Discriminator          |

All methods provide:
- Sample-level $t_0$ estimate
- Interpolated $t_0$ approximation
- Optional plotting utilities
- Nanosecond conversion based on ADC sample frequency

### Statistical Engine (stats)

Two operating modes:

#### 1. Simulation mode (simulation=True)

Ground-truth $t_0$ is known → supervised statistics:
- Absolute error
- Signed error
- Normalized error
- TP/FP evaluation
- ROC, TPR/FPR, AUC
- Parameter grid searches
- Heatmaps and histograms

#### 2. Real-data mode (simulation=False)

Ground truth not available → unsupervised:
- Method comparison
- Distribution analysis
- Valid detection fraction

## Installation

Install directly from source:
```bash
pip install .
```

## Quick Start

1. Simulate a pulse
```python
from mlabonset import PulseSimulator

sim = PulseSimulator(500)
pulse = sim.semi_gaussian(t0=200, amplitude=700, tau=6, snr=50)
```

2. Detect onset time (DLIM)
```python
from mlabonset import DLIM

dlim = DLIM(4)
t0_sample, t0_interp = dlim.t0_get(pulse)
print(t0_sample, t0_interp)
```

3. Statistical analysis (synthetic mode)
```python
from mlabonset import PulseStatistics

stats = PulseStatistics(simulation=True)
results = stats.evaluate_dataset(
    dataset,
    detect_func=lambda x: dlim.t0_get(x)[1],
    true_t0=200
)
```

4. Plot histograms
```python
from pulse_tools import plot_overlaid_histograms

plot_overlaid_histograms(results, column="signed_error", hue="method")
```
5. ROC Analysis
```python
thresholds = np.linspace(0.1, 4, 40)
roc = stats.compute_roc_auc(results, true_t0=200, thresholds=thresholds)
PulseStatistics.plot_roc_curve(roc, method_name="DLIM")
```

## Author

by ICTP-MLAB.