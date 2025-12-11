import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from typing import Union, Callable, Dict, List, Optional, Any, Tuple
from enum import Enum

PLOG = 0.2784645427610738


class OutputMode(Enum):
    """Enumeration for output signal modes."""
    ADC = "adc"
    VOLTS = "volts"


class PulseSimulator:
    """Simulates different types of pulse signals with flexible dataset generation.
    
    Supports built-in pulse models (semi-Gaussian, ramp, CR-RC) and custom 
    user-defined pulse functions. Provides unified dataset generation capabilities
    with configurable sampling frequency, output modes (ADC/Volts), and saturation.
    """

    AVAILABLE_MODELS = ['semi_gaussian', 'ramp', 'cr_rc']

    def __init__(self, 
                 num_samples: int,
                 sampling_freq: float = 1.0,
                 output_mode: Union[str, OutputMode] = OutputMode.ADC,
                 adc_bits: int = 14,
                 voltage_range: Tuple[float, float] = (0.0, 2.0)):
        """Initialize the pulse simulator with acquisition parameters.

        Args:
            num_samples: Number of samples in each generated signal.
            sampling_freq: Sampling frequency in MHz (MegaHertz).
                Used to generate time vector in nanoseconds.
                Example: 100 MHz = 10 ns per sample.
            output_mode: Signal output mode, either:
                - 'adc' or OutputMode.ADC: Output in ADC counts
                - 'volts' or OutputMode.VOLTS: Output in voltage units
            adc_bits: Number of ADC bits (only used in ADC mode).
                Determines maximum value: 2^adc_bits - 1.
                Common values: 12 bits (4095), 14 bits (16383), 16 bits (65535).
            voltage_range: Tuple (min_voltage, max_voltage) for saturation in Volts mode.
                Example: (0.0, 2.0) means signals saturate at 0V and 2V.

        Raises:
            ValueError: If sampling_freq <= 0, adc_bits < 1, or invalid voltage_range.

        Examples:
            >>> # High-speed ADC acquisition (100 MHz, 14-bit)
            >>> sim = PulseSimulator(
            ...     num_samples=200,
            ...     sampling_freq=100.0,  # MHz
            ...     output_mode='adc',
            ...     adc_bits=14
            ... )
            
            >>> # Voltage mode oscilloscope simulation
            >>> sim = PulseSimulator(
            ...     num_samples=500,
            ...     sampling_freq=50.0,   # MHz
            ...     output_mode='volts',
            ...     voltage_range=(-1.0, 1.0)
            ... )
        """
        if sampling_freq <= 0:
            raise ValueError(f"sampling_freq must be positive, got {sampling_freq}")
        if adc_bits < 1:
            raise ValueError(f"adc_bits must be >= 1, got {adc_bits}")
        if voltage_range[0] >= voltage_range[1]:
            raise ValueError(
                f"voltage_range must be (min, max) with min < max, "
                f"got {voltage_range}"
            )

        self.num_samples = num_samples
        self.sampling_freq = sampling_freq  # MHz
        self.decimals = 2

        if isinstance(output_mode, str):
            output_mode = output_mode.lower()
            if output_mode == 'adc':
                self.output_mode = OutputMode.ADC
            elif output_mode in ['volts', 'voltage', 'v']:
                self.output_mode = OutputMode.VOLTS
            else:
                raise ValueError(
                    f"Invalid output_mode '{output_mode}'. "
                    f"Must be 'adc' or 'volts'."
                )
        elif isinstance(output_mode, OutputMode):
            self.output_mode = output_mode
        else:
            raise ValueError(
                f"output_mode must be string or OutputMode enum, "
                f"got {type(output_mode)}"
            )

        self.adc_bits = adc_bits
        self.adc_max = (2 ** adc_bits) - 1
        self.adc_min = 0

        self.voltage_min, self.voltage_max = voltage_range

        self.sampling_period_ns = 1000.0 / sampling_freq
        self.time_vector = np.arange(num_samples) * self.sampling_period_ns

    def get_time_vector(self) -> np.ndarray:
        """Get the time vector in nanoseconds for the x-axis.

        Returns:
            1D numpy array with time values in nanoseconds.

        Examples:
            >>> sim = PulseSimulator(num_samples=5, sampling_freq=100.0)
            >>> sim.get_time_vector()
            array([  0.,  10.,  20.,  30.,  40.])  # 100 MHz = 10 ns/sample
        """
        return self.time_vector.copy()

    def _apply_saturation(self, signal: np.ndarray) -> np.ndarray:
        """Apply saturation limits based on output mode.

        Args:
            signal: Input signal array.

        Returns:
            Signal clipped to valid range (ADC counts or voltage).

        Notes:
            - ADC mode: Clips to [0, 2^adc_bits - 1]
            - Volts mode: Clips to [voltage_min, voltage_max]
        """
        if self.output_mode == OutputMode.ADC:
            return np.clip(signal, self.adc_min, self.adc_max)
        else:  # VOLTS mode
            return np.clip(signal, self.voltage_min, self.voltage_max)

    def semi_gaussian(self,
                      t0: float = 5.0,
                      amplitude: float = 640.0,
                      tau: float = 2.0,
                      snr: float = np.inf,
                      offset: float = 0.0,
                      reflection_scale: float = 0.0,
                      dt: float = 0.0) -> np.ndarray:
        """Generate a semi-Gaussian pulse with optional reflected component.

        Args:
            t0: Time of pulse onset in SAMPLES (not nanoseconds).
                Example: t0=50 means pulse starts at sample index 50.
            amplitude: Peak amplitude in current output mode units.
                - ADC mode: amplitude in ADC counts
                - Volts mode: amplitude in volts
            tau: Time constant controlling pulse width (samples).
            snr: Signal-to-noise ratio. Use np.inf for noiseless signal.
            offset: DC offset (baseline) in current output mode units.
                - ADC mode: offset in ADC counts
                - Volts mode: offset in volts
            reflection_scale: Scaling factor for reflected pulse component.
            dt: Time delay between main pulse and reflection (samples).

        Returns:
            Generated pulse signal as 1D numpy array (with saturation applied).

        Notes:
            - t0 is always in samples, independent of sampling frequency
            - Use get_time_vector() to get corresponding time in nanoseconds
            - Output is automatically saturated based on output_mode
        """
        pulse = np.zeros(self.num_samples)

        normalization = (
            -2 * tau**2 +
            np.exp(2 * (1 + PLOG)) *
            (2 * tau**2 * (1 - 2 * (1 + PLOG) + 2 * (1 + PLOG)**2))
        ) / np.exp(4 * (1 + PLOG))

        for t in range(self.num_samples):
            if t <= t0:
                pulse[t] = offset
                continue

            main_component = (
                np.exp((-2 * t + t0) / tau) *
                (-2 * np.exp(t0 / tau) * tau**2 +
                 np.exp(t / tau) *
                 (t**2 + t0**2 + 2*t0*tau + 2*tau**2 -
                  2 * t * (t0 + tau)))
            )

            reflection_component = (
                np.exp((-2 * t + (t0 + dt)) / tau) *
                (-2 * np.exp((t0 + dt) / tau) * tau**2 +
                 np.exp(t / tau) *
                 (t**2 + (t0 + dt)**2 + 2*(t0 + dt)*tau +
                  2*tau**2 - 2*t*((t0 + dt) + tau)))
            )

            if t <= t0 + dt:
                pulse[t] = offset + (amplitude * main_component) / normalization
            else:
                pulse[t] = offset + (
                    amplitude * main_component +
                    reflection_scale * amplitude * reflection_component
                ) / normalization

        if snr != np.inf:
            sigma = amplitude / snr
            pulse = self._add_noise(sigma, pulse)

        pulse = self._apply_saturation(pulse)

        return pulse

    def ramp(self,
             t0: float = 5.0,
             slope: Optional[float] = None,
             angle_deg: Optional[float] = None,
             sigma: float = 0.0,
             offset: float = 0.0) -> np.ndarray:
        """Generate a ramp pulse with optional Gaussian noise.
        
        The ramp can be specified either by slope or angle. If both are provided,
        angle_deg takes precedence.

        Args:
            t0: Time of ramp onset in SAMPLES (not nanoseconds).
            slope: Ramp slope (rise per sample) in output mode units.
                Ignored if angle_deg is provided.
            angle_deg: Ramp angle in degrees. If provided, overrides slope.
            sigma: Standard deviation of additive Gaussian noise in output units.
            offset: DC offset (baseline) in current output mode units.

        Returns:
            Generated ramp pulse as 1D numpy array (with saturation applied).
            
        Raises:
            ValueError: If neither slope nor angle_deg is provided.

        Notes:
            - t0 is always in samples
            - slope and offset units depend on output_mode (ADC counts or volts)
            - Output is automatically saturated
        """
        if angle_deg is not None:
            slope = np.tan(np.radians(angle_deg))
        elif slope is None:
            raise ValueError("Either 'slope' or 'angle_deg' must be provided")

        pulse = np.zeros(self.num_samples)

        for t in range(self.num_samples):
            if t <= t0:
                pulse[t] = offset
            else:
                pulse[t] = offset + slope * (t - t0)

        if sigma > 0:
            pulse = self._add_noise(sigma, pulse)

        pulse = self._apply_saturation(pulse)

        return pulse

    def cr_rc(self,
              t0: float = 50.0,
              amplitude: float = 1.0,
              tau_rise: float = 9.448,
              tau_fall: float = 61.030,
              snr: float = np.inf,
              offset: float = 450.0) -> np.ndarray:
        """Generate a CR-RC (Capacitor-Resistor / Resistor-Capacitor) pulse.
        
        This model represents the response of a CR-RC circuit, commonly used in
        nuclear detector electronics. The pulse exhibits an exponential rise
        followed by an exponential decay, characterized by two time constants.
        

        Args:
            t0: Time of pulse onset in SAMPLES (not nanoseconds).
            amplitude: Peak amplitude scaling factor in current output mode units.
            tau_rise: Rise time constant (samples). Smaller values = faster rise.
            tau_fall: Fall time constant (samples). Larger values = slower decay.
            snr: Signal-to-noise ratio. Use np.inf for noiseless signal.
            offset: DC offset (baseline) in current output mode units.

        Returns:
            Generated CR-RC pulse as 1D numpy array (with saturation applied).
        """
        t = np.arange(self.num_samples, dtype=np.float64)
        
        pulse = np.full(self.num_samples, offset, dtype=np.float64)
        
        mask = t > t0
        
        dt = t[mask] - t0
        
        exponential_rise = np.exp(-dt / tau_rise)
        exponential_fall = np.exp(-dt / tau_fall)
        pulse[mask] += amplitude * (exponential_fall - exponential_rise)
        
        if snr != np.inf:
            sigma = amplitude / snr
            pulse = self._add_noise(sigma, pulse)
        
        pulse = self._apply_saturation(pulse)
        
        return pulse

    def generate_dataset(self,
                        model: Union[str, Callable] = 'semi_gaussian',
                        num_pulses: int = 1,
                        sampling_mode: str = 'grid',
                        params: Optional[Dict[str, Any]] = None,
                        random_seed: Optional[int] = None,
                        n_jobs: int = -2,
                        return_dataframe: bool = True,
                        include_time_vector: bool = False) -> Union[pd.DataFrame, List[Dict]]:
        """Generate a dataset of pulses using any pulse model.
        
        This is the unified entry point for all dataset generation. Supports
        built-in models, custom user-defined functions, and both grid-based
        and random sampling strategies.

        Args:
            model: Pulse model to use. Can be:
                - 'semi_gaussian': Built-in semi-Gaussian pulse
                - 'ramp': Built-in ramp pulse
                - 'cr_rc': Built-in CR-RC pulse
                - Callable: Custom user-defined pulse function
            num_pulses: Number of pulses to generate per parameter combination.
            sampling_mode: Strategy for parameter sampling:
                - 'grid': Cartesian product of all parameter ranges
                - 'random': Random sampling from parameter ranges
            params: Dictionary of parameters. Each value can be:
                - Single value: Fixed parameter
                - List [min, max, step]: Range for grid mode
                - List [min, max]: Range for random mode
                Special handling for 'snr': uses logarithmic sampling
            random_seed: Random seed for reproducibility (random mode only).
            n_jobs: Number of parallel jobs (-1: all cores, -2: all but one).
            return_dataframe: If True, return pandas DataFrame. Otherwise list of dicts.
            include_time_vector: If True, add time_ns column with time vector.

        Returns:
            Dataset as DataFrame or list of dictionaries, each containing
            parameters and the generated 'pulse' array.

        Raises:
            ValueError: If model is invalid or params are incompatible with mode.

        Notes:
            - All time-based parameters (t0, tau, etc.) are in SAMPLES
            - Amplitude/offset units depend on output_mode setting
            - All pulses are automatically saturated
            - Time vector can be retrieved with get_time_vector()

        Examples:
            >>> sim = PulseSimulator(
            ...     num_samples=100,
            ...     sampling_freq=100.0,
            ...     output_mode='adc',
            ...     adc_bits=14
            ... )
            
            # Generate semi-Gaussian dataset with grid sampling
            >>> dataset = sim.generate_dataset(
            ...     model='semi_gaussian',
            ...     num_pulses=5,
            ...     sampling_mode='grid',
            ...     params={
            ...         't0': [10, 20, 5],
            ...         'amplitude': [1000, 2000, 500],
            ...         'tau': 2.0,
            ...         'offset': 500
            ...     },
            ...     include_time_vector=True
            ... )
            
            # Generate ramp dataset with random sampling
            >>> dataset = sim.generate_dataset(
            ...     model='ramp',
            ...     num_pulses=100,
            ...     sampling_mode='random',
            ...     params={
            ...         't0': [5, 15],
            ...         'angle_deg': [30, 60],
            ...         'sigma': 0.5
            ...     },
            ...     random_seed=42
            ... )
        """
        if params is None:
            params = {}

        pulse_func = self._get_pulse_function(model)
        
        self._validate_params(params, sampling_mode)

        if sampling_mode == 'grid':
            dataset = self._generate_grid_dataset(
                pulse_func, num_pulses, params, n_jobs
            )
        elif sampling_mode == 'random':
            dataset = self._generate_random_dataset(
                pulse_func, num_pulses, params, random_seed, n_jobs
            )
        else:
            raise ValueError(
                f"Invalid sampling_mode '{sampling_mode}'. "
                f"Must be 'grid' or 'random'."
            )

        if include_time_vector:
            time_vec = self.get_time_vector()
            for record in dataset:
                record['time_ns'] = time_vec

        if return_dataframe:
            df = pd.DataFrame(dataset)
            df.attrs['sampling_freq_mhz'] = self.sampling_freq
            df.attrs['sampling_period_ns'] = self.sampling_period_ns
            df.attrs['output_mode'] = self.output_mode.value
            df.attrs['num_samples'] = self.num_samples
            if self.output_mode == OutputMode.ADC:
                df.attrs['adc_bits'] = self.adc_bits
                df.attrs['adc_max'] = self.adc_max
            else:
                df.attrs['voltage_range'] = (self.voltage_min, self.voltage_max)
            return df
        
        return dataset

    def _get_pulse_function(self, model: Union[str, Callable]) -> Callable:
        """Get the pulse generation function from model specification.
        
        Args:
            model: Model name (string) or custom callable.
            
        Returns:
            Pulse generation function.
            
        Raises:
            ValueError: If model string is not recognized.
            TypeError: If model is not string or callable.
        """
        if isinstance(model, str):
            if model not in self.AVAILABLE_MODELS:
                raise ValueError(
                    f"Unknown model '{model}'. "
                    f"Available models: {self.AVAILABLE_MODELS}"
                )
            return getattr(self, model)
        elif callable(model):
            return model
        else:
            raise TypeError(
                f"Model must be string or callable, got {type(model)}"
            )

    def _validate_params(self, params: Dict, sampling_mode: str) -> None:
        """Validate parameter dictionary based on sampling mode.
        
        Args:
            params: Parameter dictionary to validate.
            sampling_mode: 'grid' or 'random'.
            
        Raises:
            ValueError: If parameters are invalid for the given mode.
        """
        for key, value in params.items():
            if isinstance(value, (list, tuple)):
                if sampling_mode == 'grid' and len(value) != 3:
                    raise ValueError(
                        f"Grid mode requires [min, max, step] for '{key}', "
                        f"got {value}"
                    )
                elif sampling_mode == 'random' and len(value) != 2:
                    raise ValueError(
                        f"Random mode requires [min, max] for '{key}', "
                        f"got {value}"
                    )

    def _generate_grid_dataset(self,
                               pulse_func: Callable,
                               num_pulses: int,
                               params: Dict,
                               n_jobs: int) -> List[Dict]:
        """Generate dataset using grid (Cartesian product) sampling.
        
        Args:
            pulse_func: Pulse generation function.
            num_pulses: Number of pulses per parameter combination.
            params: Parameter specifications.
            n_jobs: Number of parallel jobs.
            
        Returns:
            List of dictionaries containing parameters and pulses.
        """
        param_grids = {}
        for key, value in params.items():
            if isinstance(value, (list, tuple)):
                if key == 'snr':
                    param_grids[key] = np.logspace(
                        value[0], value[1], int(value[2])
                    ).round(self.decimals)
                else:
                    param_grids[key] = np.arange(
                        value[0], value[1] + value[2]/2, value[2]
                    ).round(self.decimals)
            else:
                param_grids[key] = [value]

        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())
        
        combinations = []
        self._build_combinations(param_values, [], combinations)
        
        def generate_single(param_combo):
            param_dict = dict(zip(param_names, param_combo))
            pulse = pulse_func(**param_dict)
            result = param_dict.copy()
            result['pulse'] = pulse
            return result
        
        all_combos = combinations * num_pulses
        
        dataset = Parallel(n_jobs=n_jobs)(
            delayed(generate_single)(combo) for combo in all_combos
        )
        
        return dataset

    def _build_combinations(self, lists, current, result):
        """Recursively build Cartesian product of parameter lists."""
        if not lists:
            result.append(tuple(current))
            return
        
        for item in lists[0]:
            self._build_combinations(lists[1:], current + [item], result)

    def _generate_random_dataset(self,
                                 pulse_func: Callable,
                                 num_pulses: int,
                                 params: Dict,
                                 random_seed: Optional[int],
                                 n_jobs: int) -> List[Dict]:
        """Generate dataset using random sampling.
        
        Args:
            pulse_func: Pulse generation function.
            num_pulses: Total number of pulses to generate.
            params: Parameter specifications.
            random_seed: Random seed for reproducibility.
            n_jobs: Number of parallel jobs.
            
        Returns:
            List of dictionaries containing parameters and pulses.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        def generate_single(_):
            param_dict = {}
            for key, value in params.items():
                if isinstance(value, (list, tuple)):
                    if key == 'snr':
                        log_val = np.random.uniform(value[0], value[1])
                        param_dict[key] = 10 ** log_val
                    else:
                        param_dict[key] = np.random.uniform(value[0], value[1])
                else:
                    param_dict[key] = value
            
            pulse = pulse_func(**param_dict)
            result = param_dict.copy()
            result['pulse'] = pulse
            
            for key in result:
                if key != 'pulse' and isinstance(result[key], (int, float)):
                    result[key] = round(result[key], 3)
            
            return result

        dataset = Parallel(n_jobs=n_jobs)(
            delayed(generate_single)(i) for i in range(num_pulses)
        )
        
        return dataset

    def _add_noise(self, sigma: float, signal: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to a signal.

        Args:
            sigma: Standard deviation of noise in current output mode units.
            signal: Clean input signal.

        Returns:
            Noisy signal.
        """
        noise = np.random.normal(scale=sigma, size=len(signal))
        return signal + noise