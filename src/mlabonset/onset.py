import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Utility Functions
# =============================================================================

def samples_to_ns(t0_samples: float, sampling_frequency_hz: float) -> float:
    """Convert sample index time to nanoseconds.

    Args:
        t0_samples (float): Time-of-arrival in samples.
        sampling_frequency_hz (float): ADC sampling frequency in Hz.

    Returns:
        float: Time-of-arrival in nanoseconds.
    """
    return (t0_samples / sampling_frequency_hz) * 1e9


def ns_to_samples(t0_ns: float, sampling_frequency_hz: float) -> float:
    """Convert nanoseconds to sample index.

    Args:
        t0_ns (float): Time in nanoseconds.
        sampling_frequency_hz (float): ADC sampling frequency in Hz.

    Returns:
        float: Time in samples.
    """
    return (t0_ns * 1e-9) * sampling_frequency_hz


# =============================================================================
# Double Linear Intersection Method (DLIM)
# =============================================================================

class DLIM:
    """Finite-impulse-response based onset detector using second derivative
    filtering and linear intersection approximation.
    """

    def __init__(self, width: int):
        """Initialize the FIR-based onset detector.

        Args:
            width (int): FIR window width (must be even).
        """
        self.width = width
        self.half_width = width // 2
        self.dydx2_coef = self._get_dydx2_coef()
        self.a_coef = self._get_a_coef(self.half_width)
        self.b_coef = self._get_b_coef(self.half_width)
        self.data = None
        self.name = 'DLIM'

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def t0_get(self, data, window: int = None, mode: str = 'normal'):
        """Compute time-of-arrival (t0) using FIR derivative filtering.

        Args:
            data (numpy.ndarray): Input pulse data.
            window (int, optional): FIR window override.
            mode (str): Filter mode ("normal" or "symmetric").

        Returns:
            tuple:
                - t0_sample (float): Discrete t0 sample index.
                - t0_approx (float): Approximate t0 with linear interpolation.
        """
        # Optional window override
        if window is not None:
            self.width = window
            self.half_width = window // 2
            self.dydx2_coef = self._get_dydx2_coef(mode)
            self.a_coef = self._get_a_coef(self.half_width)
            self.b_coef = self._get_b_coef(self.half_width)

        self.data = data
        
        if len(self.data) < len(self.dydx2_coef):  # señal menor que FIR
            return np.nan, np.nan

        if np.all(self.data == 0):  # no hay señal útil
            return np.nan, np.nan

        # Step 1: Find coarse t0 using second derivative FIR
        t0_sample = self._fir_second_derivative()
        
        if not np.isfinite(t0_sample) or t0_sample <= 0:
            return np.nan, np.nan

        # Step 2: Compute fine interpolation if possible
        if t0_sample > self.width:
            _, t0_approx = self._intersect(t0_sample)
            return t0_sample, t0_approx

        return t0_sample, t0_sample

    def t0_get_ns(self, data, sampling_frequency_hz: float, **kwargs):
        """Compute t0 in samples and convert to nanoseconds.

        Args:
            data (numpy.ndarray): Input pulse.
            sampling_frequency_hz (float): Sampling frequency in Hz.
            **kwargs: Passed to t0_get().

        Returns:
            tuple:
                - t0_samples (float)
                - t0_ns (float)
        """
        t0_samp, t0_approx = self.t0_get(data, **kwargs)
        t0_ns = samples_to_ns(t0_approx, sampling_frequency_hz)
        return t0_approx, t0_ns

    def t0_get_dataset(self, data_list, t0_real_list, window=None):
        """Evaluate t0 estimation over a dataset.

        Args:
            data_list (list): List of pulses.
            t0_real_list (list): Ground truth t0 values.
            window (int): FIR window override.

        Returns:
            tuple:
                - estimated_t0 (list)
                - real_t0 (list)
        """
        t0_estimated = []
        t0_real = []

        for data, t0_real_val in zip(data_list, t0_real_list):
            t0_sample, t0_aprox = self.t0_get(data, window=window)
            if t0_aprox != -1:
                t0_estimated.append(t0_aprox)
                t0_real.append(t0_real_val)

        return t0_estimated, t0_real

    # -------------------------------------------------------------------------
    # FIR core
    # -------------------------------------------------------------------------

    def _fir_second_derivative(self) -> int:
        """Compute second derivative using FIR convolution.

        Returns:
            int: Position of the maximum derivative (coarse t0).
        """
        max_index = np.argmax(self.data)
        filtered = np.convolve(self.data[:max_index], self.dydx2_coef, mode='same')
        return int(np.argmax(filtered))

    # -------------------------------------------------------------------------
    # Filter coefficients
    # -------------------------------------------------------------------------

    def _get_dydx2_coef(self, mode='normal'):
        """Generate second-derivative FIR coefficients.

        Args:
            mode (str): 'normal' or 'symmetric'.

        Raises:
            TypeError: If width is odd.

        Returns:
            numpy.ndarray: FIR coefficient array.
        """
        if self.width % 2 != 0:
            raise TypeError("Window width must be an even number.")

        if mode == 'normal':
            return np.array([
                1 - 2 * (i / (self.width / 2 - 1)) if i < self.width / 2
                else -1 + 2 * ((i - self.width / 2) / (self.width / 2 - 1))
                for i in range(self.width)
            ])

        # Symmetric mode
        a = self._get_a_coef(self.half_width)
        return np.append(a, a[::-1])

    def _get_a_coef(self, n: int):
        """Coefficient function A(n)."""
        return np.array([-6 * (1 + n - 2 * (n - i)) / (n**3 - n) for i in range(n)])

    def _get_b_coef(self, n: int):
        """Coefficient function B(n)."""
        return np.array([(4 - 2*n + 6*i) / (n**2 + n) for i in range(n)])

    # -------------------------------------------------------------------------
    # Interpolation utilities
    # -------------------------------------------------------------------------

    def _linear_approx(self, y):
        """Compute linear approximation (slope, intercept).

        Args:
            y (numpy.ndarray): Input data segment.

        Returns:
            tuple: (slope, intercept)
        """
        a = np.convolve(y, self.a_coef, mode='valid')
        b = np.convolve(y, self.b_coef, mode='valid')
        return a[0], b[0]

    def _intersect(self, n0):
        """Compute intersection between two line segments around t0.

        Args:
            n0 (int): Coarse t0 location.

        Returns:
            tuple:
                - q (float): Intersection offset.
                - t0 (float): Refined t0 location.
        """
        left = self.data[n0 - self.half_width: n0]
        right = self.data[n0: n0 + self.half_width]

        m1, b1 = self._linear_approx(left)
        m2, b2 = self._linear_approx(right)

        q = 0 if (m1 - m2) == 0 else (b2 - b1 - m2 * self.half_width) / (m1 - m2)
        t0 = n0 + q - self.half_width

        return q, t0
    
    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------    
    def plot(self, data, sampling_frequency_hz=None, window=None, mode='normal', ylim=None, xlim=None):
        """Plot the input pulse, second derivative, maximum derivative position
        and estimated t0.

        Args:
            data (numpy.ndarray): Input pulse to analyze.
            sampling_frequency_hz (float, optional): If provided, plot t0 in ns.
            window (int, optional): Override FIR window.
            mode (str): Filter mode ("normal" or "symmetric").
        """
        # Compute t0
        t0_sample, t0_est = self.t0_get(data, window=window, mode=mode)

        # Convert to ns if sampling frequency provided
        t0_ns = None
        if sampling_frequency_hz:
            t0_ns = (t0_est / sampling_frequency_hz) * 1e9

        # Get second derivative for visualization
        max_index = np.argmax(data)
        dy2 = np.convolve(data[:max_index], self.dydx2_coef, mode='same')


        plt.figure(figsize=(10, 6))
        plt.plot(data, label="Input Pulse", linewidth=2)
        plt.plot(np.arange(len(dy2)), dy2, label="Second Derivative (d²y/dx²)", alpha=0.7)

        # Mark coarse max position
        plt.axvline(t0_sample, color='red', linestyle='--', label=f"Max derivative = {t0_sample:.2f} samples")

        # Mark interpolated t0
        if t0_est > 0:
            plt.axvline(t0_est, color='green', linestyle=':', label=f"t0 ≈ {t0_est:.2f} samples")

        if t0_ns:
            plt.title(f"DLIM Onset Detection (t0 = {t0_est:.2f} samples, {t0_ns:.1f} ns)")
        else:
            plt.title(f"DLIM Onset Detection (t0 = {t0_est:.2f} samples)")
        
        if ylim != None:
            plt.ylim(ylim)
        if xlim != None:
            plt.xlim(xlim)
        plt.legend()
        plt.grid()
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

# =============================================================================
# DCFD (Digital Constant Fraction Discriminator)
# =============================================================================

class DCFD:
    """Constant fraction discriminator (CFD) onset detector."""

    def __init__(self, fraction: float, delay: int):
        """
        Args:
            fraction (float): Fraction of max amplitude.
            delay (int): Delay in samples.
        """
        self.fraction = fraction
        self.delay = delay
        self.name = "DCFD"

    def t0_get(self, data, fraction=None, delay=None):
        """Compute t0 using constant fraction discrimination.

        Args:
            data (numpy.ndarray): Input pulse.
            fraction (float, optional): Override fraction.
            delay (int, optional): Override delay.

        Returns:
            tuple:
                - x0 (int): Integer t0 crossing.
                - t0 (float): Interpolated t0.
        """
        if fraction is not None:
            self.fraction = fraction
        if delay is not None:
            self.delay = delay

        s_frac = data * self.fraction
        s_delayed = -np.roll(s_frac, -self.delay)
        s_diff = s_frac + s_delayed

        min_index = s_diff.argmin()
        max_index = s_diff.argmax()

        if np.any(s_diff[min_index:max_index] > 0):
            x0 = np.where(s_diff[min_index:max_index] > 0)[0][0] + min_index - 1
            x1 = x0 + 1

            if s_diff[x1] != s_diff[x0]:
                t0 = (0 - s_diff[x1]) / (s_diff[x1] - s_diff[x0]) + x1
                return x0 - self.delay, t0 - self.delay

        return -1, -1
    
    def t0_get_ns(self, data, sampling_frequency_hz: float, fraction=None, delay=None):
        """Compute t0 using CFD and return value in samples and nanoseconds.

        Args:
            data (numpy.ndarray): Input pulse.
            sampling_frequency_hz (float): ADC sampling frequency in Hz.
            fraction (float, optional): Override fraction.
            delay (int, optional): Override delay.

        Returns:
            tuple:
                - t0_samples (float): Estimated t0 in samples.
                - t0_ns (float): Estimated t0 in nanoseconds.
        """
        x0, t0_samples = self.t0_get(
            data,
            fraction=fraction,
            delay=delay
        )

        if t0_samples < 0:
            return -1, -1

        t0_ns = (t0_samples / sampling_frequency_hz) * 1e9
        return t0_samples, t0_ns

    # -----------------------------------------------------------
    # Plot
    # -----------------------------------------------------------
    def plot(self, data, fraction=None, delay=None, sampling_frequency_hz=None):
        """Plot CFD input pulse, delayed pulse, difference signal, and t0.

        Args:
            data (numpy.ndarray): Input pulse.
            fraction (float, optional): Override fraction.
            delay (int, optional): Override delay.
            sampling_frequency_hz (float, optional): Convert t0 to ns.
        """
        # Compute t0
        x0, t0 = self.t0_get(data, fraction=fraction, delay=delay)

        # Convert to ns
        t0_ns = None
        if sampling_frequency_hz:
            t0_ns = (t0 / sampling_frequency_hz) * 1e9

        # CFD internal signals
        frac = data * self.fraction
        delayed = -np.roll(frac, -self.delay)
        diff = frac + delayed


        plt.figure(figsize=(10, 6))
        plt.plot(data, label="Input Pulse", linewidth=2)
        plt.plot(diff, label="CFD Difference", linestyle='--')

        # Mark results
        if x0 >= 0:
            plt.axvline(x0, color='orange', linestyle='--', label=f"x0 = {x0:.2f}")
        if t0 >= 0:
            plt.axvline(t0, color='green', linestyle=':', label=f"t0 = {t0:.2f} samples")
            if t0_ns:
                plt.title(f"CFD Onset (t0 ≈ {t0:.2f} samples, {t0_ns:.1f} ns)")
            else:
                plt.title(f"CFD Onset (t0 ≈ {t0:.2f} samples)")

        plt.legend()
        plt.grid()
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

# =============================================================================
# Digital Leading Edge Discriminator
# =============================================================================

class DLED:
    """Leading-edge discriminator (LED) onset detector."""

    def __init__(self, threshold: float):
        """
        Args:
            threshold (float): Threshold level.
        """
        self.threshold = threshold
        self.name = "DLED"

    def t0_get(self, data, threshold=None):
        """Perform threshold crossing onset detection.

        Args:
            data (numpy.ndarray): Input pulse.
            threshold (float, optional): Override threshold.

        Returns:
            tuple:
                - x0 (int): Discrete t0.
                - t0 (float): Interpolated t0.
        """
        data = np.array(data)

        if threshold is not None:
            self.threshold = threshold

        if self.threshold >= np.max(data):
            return -1, -1

        try:
            x0 = next(i for i, val in enumerate(data) if val > self.threshold)
        except StopIteration:
            return -1, -1

        x1 = x0 - 1

        if x0 > 1 and (data[x1] - data[x0]) != 0:
            y_half = (data[x0] - data[x1]) / 2
            t0 = (y_half - data[x1]) / ((data[x0] - data[x1]) / (x0 - x1)) + x1
            return x0, t0

        return -1, -1

    def t0_get_ns(self, data, sampling_frequency_hz: float, threshold=None):
        """Compute t0 using threshold crossing and return it in samples and ns.

        Args:
            data (numpy.ndarray): Input pulse.
            sampling_frequency_hz (float): ADC sampling frequency in Hz.
            threshold (float, optional): Override threshold.

        Returns:
            tuple:
                - t0_samples (float): t0 in samples.
                - t0_ns (float): t0 in nanoseconds.
        """
        x0, t0_samples = self.t0_get(
            data,
            threshold=threshold
        )

        if t0_samples < 0:
            return -1, -1

        t0_ns = (t0_samples / sampling_frequency_hz) * 1e9
        return t0_samples, t0_ns

    # -----------------------------------------------------------
    # Plot
    # -----------------------------------------------------------
    def plot(self, data, threshold=None, sampling_frequency_hz=None):
        """Plot pulse and threshold crossing.

        Args:
            data (numpy.ndarray): Input pulse.
            threshold (float, optional): Override threshold.
            sampling_frequency_hz (float, optional): Convert t0 to ns.
        """
        # Compute t0
        x0, t0 = self.t0_get(data, threshold=threshold)

        # Convert to ns
        t0_ns = None
        if sampling_frequency_hz:
            t0_ns = (t0 / sampling_frequency_hz) * 1e9

        
        plt.figure(figsize=(10, 6))
        plt.plot(data, label="Input Pulse", linewidth=2)

        th = self.threshold if threshold is None else threshold
        plt.axhline(th, color='orange', linestyle='--', label=f"Threshold = {th}")

        # Mark crossing points
        if x0 >= 0:
            plt.axvline(x0, color='red', linestyle='--', label=f"x0 = {x0}")
        if t0 >= 0:
            plt.axvline(t0, color='green', linestyle=':', label=f"t0 = {t0:.2f} samples")

        if t0_ns:
            plt.title(f"DLED Onset (t0 ≈ {t0:.2f} samples, {t0_ns:.1f} ns)")
        else:
            plt.title(f"DLED Onset (t0 ≈ {t0:.2f} samples)")

        plt.legend()
        plt.grid()
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()
