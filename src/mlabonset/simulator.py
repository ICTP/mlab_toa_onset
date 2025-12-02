import numpy as np
from joblib import Parallel, delayed

PLOG = 0.2784645427610738


class PulseSimulator:
    """Simulates different types of pulse signals, including semi-Gaussian
    and ramp-based pulses. Also includes dataset generation utilities.
    """

    def __init__(self, samples: int):
        """Initialize the simulator.

        Args:
            samples (int): Number of samples of each generated signal.
        """
        self.num_samples = samples
        self.decimals = 2

    # -------------------------------------------------------------------------
    # SEMI-GAUSSIAN PULSE
    # -------------------------------------------------------------------------
    def semi_gaussian(self,
                      t0: float = 5,
                      amplitude: float = 640,
                      snr: float = np.inf,
                      offset: float = 0,
                      tau: float = 2,
                      reflection_scale: float = 0,
                      dt: float = 0,
                      return_dict: bool = False):
        """Generate a semi-Gaussian pulse with an optional reflected component.

        Args:
            t0 (float): Time at which the pulse starts.
            amplitude (float): Peak amplitude of the pulse.
            snr (float): Signal-to-noise ratio. Infinite means no noise.
            offset (float): DC offset of the pulse.
            tau (float): Time constant of the pulse.
            reflection_scale (float): Scaling factor for the reflected pulse.
            dt (float): Time delay between main pulse and reflection.
            return_dict (bool): If True, return a dictionary with metadata.

        Returns:
            numpy.ndarray or dict: Generated pulse or metadata dictionary.
        """

        num = self.num_samples
        pulse = np.zeros(num)

        # Normalization (same as your original)
        normalization = (
            -2 * tau**2 +
            np.exp(2 * (1 + PLOG)) *
            (2 * tau**2 * (1 - 2 * (1 + PLOG) + 2 * (1 + PLOG)**2))
        ) / np.exp(4 * (1 + PLOG))

        for t in range(num):

            if t <= t0:
                pulse[t] = offset
                continue

            # Main component
            main_component = (
                np.exp((-2 * t + t0) / tau) *
                (-2 * np.exp(t0 / tau) * tau**2 +
                 np.exp(t / tau) *
                 (t**2 + t0**2 + 2*t0*tau + 2*tau**2 -
                  2 * t * (t0 + tau)))
            )

            # Reflection component
            reflection_component = (
                np.exp((-2 * t + (t0 + dt)) / tau) *
                (-2 * np.exp((t0 + dt) / tau) * tau**2 +
                 np.exp(t / tau) *
                 (t**2 + (t0 + dt)**2 + 2*(t0 + dt)*tau +
                  2*tau**2 - 2*t*((t0 + dt) + tau)))
            )

            if t <= t0 + dt:
                # Rising before reflection
                pulse[t] = offset + (amplitude * main_component) / normalization
            else:
                # After reflection
                pulse[t] = offset + (
                    amplitude * main_component +
                    reflection_scale * amplitude * reflection_component
                ) / normalization

        # Add noise
        if snr != np.inf:
            sigma = amplitude / snr
            pulse = self._noise_generator(sigma, pulse)

        if not return_dict:
            return pulse

        return {
            "t0": t0,
            "amplitude": amplitude,
            "tau": tau,
            "snr": snr,
            "pulse": pulse,
        }


    # -------------------------------------------------------------------------
    # RAMP PULSE
    # -------------------------------------------------------------------------
    def ramp(self,
             t0: float = 5,
             slope: float = 1,
             sigma: float = 0,
             offset: float = 0,
             return_dict: bool = False):
        """Generate a ramp pulse with optional Gaussian noise.

        Args:
            t0 (float): Time of ramp onset.
            slope (float): Ramp slope.
            sigma (float): Standard deviation of additive noise.
            offset (float): Pulse baseline.
            return_dict (bool): Whether to return metadata.

        Returns:
            numpy.ndarray or dict: Ramp pulse or metadata dictionary.
        """
        pulse = np.zeros(self.num_samples)

        for t in range(self.num_samples):
            if t <= t0:
                pulse[t] = offset
            else:
                pulse[t] = offset + slope * (t - t0)

        if sigma != 0:
            pulse = self._noise_generator(sigma, pulse)

        if not return_dict:
            return pulse

        return {
            "t0": t0,
            "slope": slope,
            "sigma": sigma,
            "pulse": pulse,
        }

    # -------------------------------------------------------------------------
    # RAMP WITH ANGLE
    # -------------------------------------------------------------------------
    def ramp_alpha(self,
                   t0: float = 5,
                   angle_deg: float = 45,
                   sigma: float = 0,
                   offset: float = 0,
                   return_dict: bool = False):
        """Generate a ramp defined by an angle in degrees.

        Args:
            t0 (float): Time where the ramp begins.
            angle_deg (float): Ramp angle in degrees.
            sigma (float): Noise standard deviation.
            offset (float): DC offset.
            return_dict (bool): If True, return metadata.

        Returns:
            numpy.ndarray or dict: Ramp pulse or metadata dictionary.
        """
        pulse = np.zeros(self.num_samples)
        slope = np.tan(np.radians(angle_deg))

        for t in range(self.num_samples):
            if t <= t0:
                pulse[t] = offset
            else:
                pulse[t] = offset + slope * (t - t0)

        if sigma != 0:
            pulse = self._noise_generator(sigma, pulse)

        if not return_dict:
            return pulse

        return {
            "t0": t0,
            "angle_deg": angle_deg,
            "sigma": sigma,
            "pulse": pulse,
        }

    # -------------------------------------------------------------------------
    # UTILS
    # -------------------------------------------------------------------------
    def _noise_generator(self, sigma: float, data: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to a signal.

        Args:
            sigma (float): Noise standard deviation.
            data (numpy.ndarray): Clean input signal.

        Returns:
            numpy.ndarray: Noisy signal.
        """
        noise = np.random.normal(scale=sigma, size=len(data))
        return data + noise

    def _range_generator(self, value, mode: str = 'normal'):
        """Generate a linear or logarithmic range.

        Args:
            value (float or list): Range descriptor: [min, max, step].
            mode (str): 'normal' (linear) or 'log'.

        Returns:
            list or numpy.ndarray: Generated range.
        """
        if isinstance(value, (list, np.ndarray)):
            if mode == 'normal':
                return np.arange(value[0], value[1], value[2]).round(self.decimals)
            return np.logspace(value[0], value[1], value[2]).round(self.decimals)

        return [value]

    def _random_value(self, value_range):
        """Generate a random value given a range.

        Args:
            value_range (list or float): Range or fixed value.

        Returns:
            float: Random sampled value.
        """
        if isinstance(value_range, (list, np.ndarray)):
            return (
                np.random.randint(value_range[0], value_range[1]) +
                round(np.random.random(), 2)
            )
        return value_range

    # -------------------------------------------------------------------------
    # DATASET GENERATORS
    # -------------------------------------------------------------------------
    def dataset_ramp(self,
                     num_pulses: int = 1,
                     t0=5,
                     slope=1,
                     sigma=0,
                     offset=0,
                     round_decimals: int = 2,
                     jobs = -2):
        """Generate a dataset of ramp pulses.

        Args:
            num_pulses (int): Number of pulses per parameter combination.
            t0: Ramp onset or range.
            slope: Ramp slope or range.
            sigma: Noise level or range.
            offset: Offset or range.
            round_decimals (int): Rounding precision.
            jobs (int): number of cores for parallel generation.

        Returns:
            list: Collection of ramp pulses.
        """
        self.decimals = round_decimals

        dataset = Parallel(n_jobs=jobs)(
            delayed(self.ramp)(
                t0=t0_i,
                slope=slope_i,
                sigma=sigma_i,
                offset=offset,
                return_dict=True
            )
            for slope_i in self._range_generator(slope, mode='log')
            for sigma_i in self._range_generator(sigma)
            for t0_i in self._range_generator(t0)
            for _ in range(num_pulses)
        )
        return dataset

    def dataset_ramp_alpha(self,
                           num_pulses: int = 1,
                           t0=5,
                           angle_deg=45,
                           sigma=0,
                           offset=0,
                           round_decimals: int = 2,
                           jobs = -2):
        """Generate a dataset of angled ramp pulses.

        Args:
            num_pulses (int): Number of pulses per configuration.
            t0: Onset or range.
            angle_deg: Angle of ramp or range.
            sigma: Noise or range.
            offset: DC offset.
            round_decimals (int): Precision.
            jobs (int): number of cores for parallel generation.

        Returns:
            list: Dataset of ramp-alpha pulses.
        """
        self.decimals = round_decimals

        dataset = Parallel(n_jobs=jobs)(
            delayed(self.ramp_alpha)(
                t0=t0_i,
                angle_deg=angle_i,
                sigma=sigma_i,
                offset=offset,
                return_dict=True
            )
            for angle_i in self._range_generator(angle_deg)
            for sigma_i in self._range_generator(sigma)
            for t0_i in self._range_generator(t0)
            for _ in range(num_pulses)
        )
        return dataset

    def dataset_semigauss(self,
                          num_pulses: int = 1,
                          t0=50,
                          amplitude=1,
                          tau=2,
                          snr=np.inf,
                          offset=0,
                          round_decimals: int = 2,
                          jobs = -2):
        """Generate a dataset of semi-Gaussian pulses.

        Args:
            num_pulses (int): Number of pulses per combination.
            t0: Onset or range.
            amplitude: Pulse amplitude or range.
            tau: Time constant or range.
            snr: SNR or range (logarithmic).
            offset: DC offset.
            round_decimals (int): Precision.
            jobs (int): number of cores for parallel generation.

        Returns:
            list: List of generated pulses with metadata.
        """
        self.decimals = round_decimals

        dataset = Parallel(n_jobs=jobs)(
            delayed(self.semi_gaussian)(
                t0=t0_i,
                amplitude=amp_i,
                snr=snr_i,
                offset=offset,
                tau=tau_i,
                return_dict=True
            )
            for tau_i in self._range_generator(tau)
            for amp_i in self._range_generator(amplitude)
            for snr_i in self._range_generator(snr, mode='log')
            for t0_i in self._range_generator(t0)
            for _ in range(num_pulses)
        )
        return dataset

    def dataset_random_semigaussian(
            self,
            num_pulses=100,
            t0=50,
            amplitude=640,
            tau=2,
            snr=np.inf,
            offset=0,
            seed=None,
        ):
            """
            Generate a DataFrame of semi-Gaussian pulses with random parameters.

            This behaves like `dataset_semigauss` but uses random sampling
            instead of deterministic ranges.

            Args:
                number_pulses (int): Number of pulses to generate.
                t0 (float or [min, max]): Fixed or uniform-random t0.
                amp (float or [min, max]): Fixed or uniform-random amplitude.
                tau (float or [min, max]): Fixed or uniform-random tau.
                snr (float or [min_exp, max_exp] for log10 sampling, or float): 
                    - float → fixed SNR
                    - list of 2 → log10 sampling range
                offset (float or [min, max]): Fixed or uniform-random offset.
                seed (int): Optional random seed.

            Returns:
                pandas.DataFrame: Each row contains parameters and 'pulse'.
            """

            import pandas as pd

            if seed is not None:
                np.random.seed(seed)

            def rand_param(x):
                """Uniform for real ranges, log for SNR-like ranges."""
                if isinstance(x, (list, tuple)) and len(x) == 2:
                    return np.random.uniform(x[0], x[1])
                else:
                    return x

            def rand_snr(x):
                if isinstance(x, (list, tuple)) and len(x) == 2:
                    # log10 sampling
                    log_value = np.random.uniform(x[0], x[1])
                    return 10 ** log_value
                else:
                    return x

            records = []

            for _ in range(num_pulses):
            
                t0_i = rand_param(t0)
                amp_i = rand_param(amplitude)
                tau_i = rand_param(tau)
                snr_i = rand_snr(snr)
                off_i = rand_param(offset)

                pulse = self.semi_gaussian(
                    t0=t0_i,
                    amplitude=amp_i,
                    tau=tau_i,
                    snr=snr_i,
                    offset=off_i,
                    return_dict=False
                )

                records.append(
                    {
                        "t0": round(t0_i, 3),
                        "amplitude": round(amp_i, 3),
                        "tau": round(tau_i, 3),
                        "snr": snr_i,
                        "offset": off_i,
                        "pulse": pulse,
                    }
                )

            return pd.DataFrame(records)

