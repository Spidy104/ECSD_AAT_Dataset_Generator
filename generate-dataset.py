#!/usr/bin/env python3
"""
generate-dataset.py

Generates simulated solar data with advanced physics modeling for prototyping and testing.

This is a simplified version compared to the Indian seasonal dataset, but still includes:
- Realistic solar physics (air mass, panel temperature)
- Advanced battery model (SoC tracking, temperature effects)
- Enhanced cloud modeling with smooth transitions
- Data quality validation
- Comprehensive feature engineering for TinyML

Outputs:
- Individual day CSVs: raw_day_001.csv, raw_day_002.csv, etc.
- Combined CSV: all_days_raw.csv
- Normalized CSV ready for TinyML: all_days_normalized.csv
- Dataset summary with statistics

Run: python generate-dataset.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
import time


@dataclass
class SimulationConfig:
    """Configuration parameters for solar simulation"""
    # Sampling
    fs: float = 1/5.0  # 0.2 Hz (every 5s)
    daylight_hours: int = 12
    num_days: int = 8
    seed_base: int = 42

    # Cloud parameters
    cloud_event_rate: float = 1.0 / 400.0
    cloud_mean_duration_s: float = 60.0
    cloud_depth_mean: float = 0.45
    cloud_depth_std: float = 0.12
    cloud_transition_samples: int = 5

    # Atmospheric turbulence
    sigma_turb: float = 0.04

    # Battery model
    V_min: float = 3.2
    V_max: float = 4.2
    battery_capacity_mAh: float = 2600.0
    battery_voltage_nominal: float = 3.7
    P_idle_mW: float = 50.0
    P_solar_max_mW: float = 500.0
    charge_efficiency: float = 0.85

    # Solar panel parameters
    panel_area_m2: float = 0.05  # 50 cm²
    panel_efficiency: float = 0.20  # 20%
    NOCT: float = 45.0  # Nominal Operating Cell Temperature (°C)
    temp_coeff_power: float = -0.004  # Power temperature coefficient

    # Environmental
    ambient_temp_C: float = 25.0
    temp_variation_C: float = 8.0

    # Prediction horizon
    horizon_seconds: int = 60

    # Output
    output_dir: str = "solar_minimal_dataset"

    @property
    def sample_interval_s(self) -> int:
        return int(1.0 / self.fs)

    @property
    def N_per_day(self) -> int:
        return int(self.daylight_hours * 3600 * self.fs)

    @property
    def horizon_samples(self) -> int:
        return int(self.horizon_seconds * self.fs)

    @property
    def battery_energy_Wh(self) -> float:
        return self.battery_capacity_mAh * self.battery_voltage_nominal / 1000.0


class SolarDataGenerator:
    """Generates realistic solar irradiance and battery data with advanced physics"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(exist_ok=True)

    def calculate_air_mass(self, sun_elevation: np.ndarray) -> np.ndarray:
        """
        Calculate air mass coefficient for atmospheric attenuation.
        
        Air mass (AM) = path length through atmosphere relative to zenith.
        Uses Kasten-Young formula for accuracy at low sun angles.
        """
        zenith_angle = np.pi / 2 - sun_elevation
        z_deg = np.degrees(zenith_angle)
        
        am = np.where(
            z_deg < 85,
            1.0 / (np.cos(zenith_angle) + 0.50572 * np.power(96.07995 - z_deg, -1.6364)),
            10.0
        )
        
        return np.clip(am, 1.0, 10.0)

    def clear_sky_irradiance(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate normalized clear-sky irradiance with realistic solar physics.
        
        Returns:
            irradiance: Normalized irradiance [0, 1]
            sun_elevation: Sun elevation angle in radians
        """
        t = np.arange(N) / self.config.fs
        x = t / (self.config.daylight_hours * 3600.0)

        # Solar elevation angle (sinusoidal approximation)
        max_elevation = np.radians(70)  # Maximum sun elevation
        sun_elevation = max_elevation * np.sin(np.clip(np.pi * x, 0, np.pi))

        # Base sinusoidal irradiance
        sun = np.sin(np.clip(np.pi * x, 0, np.pi))

        # Air mass atmospheric attenuation
        air_mass = self.calculate_air_mass(sun_elevation)
        optical_depth = 0.20  # Moderate atmospheric clarity
        atmospheric_transmission = np.exp(-optical_depth * air_mass)
        
        sun = sun * atmospheric_transmission

        return np.clip(sun, 0.0, 1.0), sun_elevation

    def generate_cloud_events(self, N: int, rng: np.random.RandomState) -> np.ndarray:
        """
        Generate cloud attenuation with Poisson events and smooth transitions.
        """
        attenuation = np.ones(N)
        i = 0

        while i < N:
            if rng.rand() < self.config.cloud_event_rate:
                # Cloud event starts
                duration = max(1, int(rng.exponential(
                    self.config.cloud_mean_duration_s * self.config.fs)))
                depth = float(np.clip(
                    rng.normal(self.config.cloud_depth_mean,
                               self.config.cloud_depth_std),
                    0.05, 0.95
                ))

                end = min(N, i + duration)
                trans = self.config.cloud_transition_samples

                # Smooth cloud entry
                for j in range(min(trans, end - i)):
                    fade = j / trans
                    attenuation[i + j] *= (1.0 - fade * (1.0 - depth))

                # Full cloud
                attenuation[i + trans:end - trans] *= depth

                # Smooth cloud exit
                for j in range(min(trans, end - i)):
                    fade = 1.0 - (j / trans)
                    idx = end - trans + j
                    if idx < N:
                        attenuation[idx] *= (1.0 - fade * (1.0 - depth))

                i = end
            else:
                i += 1

        return attenuation

    def add_atmospheric_effects(self, base: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Add clouds and high-frequency turbulence"""
        N = len(base)

        # High-frequency turbulence (atmospheric scintillation)
        turbulence = 1.0 + rng.normal(0.0, self.config.sigma_turb, size=N)

        # Cloud events
        cloud_atten = self.generate_cloud_events(N, rng)

        irradiance = base * turbulence * cloud_atten
        return np.clip(irradiance, 0.0, 1.0)

    def calculate_panel_temperature(
        self, irradiance: np.ndarray, sun_elevation: np.ndarray
    ) -> np.ndarray:
        """
        Calculate solar panel temperature with thermal dynamics.
        
        T_panel = T_ambient + (NOCT - 20) * (Irradiance / 800 W/m²)
        Includes thermal inertia for realistic temperature changes.
        """
        N = len(irradiance)
        panel_temp = np.zeros(N)
        panel_temp[0] = self.config.ambient_temp_C
        
        # Thermal time constant (5 minutes)
        thermal_tau = 300.0
        alpha = self.config.sample_interval_s / thermal_tau
        
        for t in range(1, N):
            irr_W_m2 = irradiance[t] * 1000.0
            T_steady = self.config.ambient_temp_C + (self.config.NOCT - 20.0) * (irr_W_m2 / 800.0)
            panel_temp[t] = panel_temp[t-1] + alpha * (T_steady - panel_temp[t-1])
        
        return panel_temp

    def simulate_battery_dynamics(
        self, irradiance: np.ndarray, sun_elevation: np.ndarray, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate realistic battery charging/discharging with advanced physics.

        Returns:
            ldr: LDR sensor readings (with noise)
            battery_voltage: Battery voltage over time
            battery_soc: State of Charge (0-100%)
            panel_temp: Panel temperature (°C)
        """
        N = len(irradiance)

        # LDR sensor with quantization noise
        ldr = np.clip(irradiance + rng.normal(0, 0.01, size=N), 0.0, 1.0)

        # Panel temperature simulation
        panel_temp = self.calculate_panel_temperature(irradiance, sun_elevation)

        # Battery simulation with SoC tracking
        battery_voltage = np.zeros(N)
        battery_soc = np.zeros(N)
        battery_voltage[0] = 3.6  # Initial voltage
        battery_soc[0] = 50.0  # Initial SoC (50%)

        capacity_mAh = self.config.battery_capacity_mAh

        for t in range(1, N):
            current_v = battery_voltage[t - 1]
            current_soc = battery_soc[t - 1]

            # Temperature effect on battery
            temp_factor = 1.0 - abs(self.config.ambient_temp_C - 25.0) * 0.002

            # Panel efficiency vs temperature
            panel_temp_factor = 1.0 + self.config.temp_coeff_power * (panel_temp[t] - 25.0)
            panel_temp_factor = np.clip(panel_temp_factor, 0.7, 1.0)

            # Power balance
            P_solar = ldr[t] * self.config.P_solar_max_mW * panel_temp_factor
            P_net = P_solar * self.config.charge_efficiency * temp_factor - self.config.P_idle_mW

            # SoC-dependent voltage
            soc_normalized = current_soc / 100.0

            if P_net > 0:  # Charging
                charge_factor = 1.0 - 0.7 * soc_normalized
                dv = (P_net / 10000.0) * charge_factor
                
                current_mA = P_net / current_v
                dsoc = (current_mA * self.config.sample_interval_s / 3600.0) / capacity_mAh * 100.0
                dsoc *= charge_factor
            else:  # Discharging
                discharge_factor = 1.0 + 0.5 * (1.0 - soc_normalized)
                dv = (P_net / 8000.0) * discharge_factor
                
                current_mA = abs(P_net) / current_v
                dsoc = -(current_mA * self.config.sample_interval_s / 3600.0) / capacity_mAh * 100.0
                dsoc *= discharge_factor

            # Add noise and self-discharge
            dv += rng.normal(0, 0.002) - 0.0001

            battery_voltage[t] = np.clip(
                current_v + dv, self.config.V_min, self.config.V_max)
            battery_soc[t] = np.clip(current_soc + dsoc, 0.0, 100.0)

        return ldr, battery_voltage, battery_soc, panel_temp

    def generate_day(self, day_idx: int) -> pd.DataFrame:
        """Generate one day of data"""
        rng = np.random.RandomState(self.config.seed_base + day_idx)
        N = self.config.N_per_day

        # Generate irradiance with sun elevation
        clear_sky, sun_elevation = self.clear_sky_irradiance(N)
        irradiance = self.add_atmospheric_effects(clear_sky, rng)

        # Simulate sensors and battery
        ldr, battery_voltage, battery_soc, panel_temp = self.simulate_battery_dynamics(
            irradiance, sun_elevation, rng
        )

        # Time in seconds since sunrise
        time_s = np.arange(N) * self.config.sample_interval_s

        # Future target (horizon steps ahead) - use NaN for unavailable data
        future_ldr = np.full(N, np.nan)
        horizon = self.config.horizon_samples
        
        if horizon < N:
            future_ldr[:-horizon] = ldr[horizon:]
            future_ldr[-horizon:] = np.nan
        else:
            future_ldr[:] = np.nan

        return pd.DataFrame({
            'day': day_idx + 1,
            'time_s': time_s,
            'irradiance_true': irradiance,
            'ldr': ldr,
            'battery_voltage': battery_voltage,
            'battery_soc': battery_soc,
            'panel_temp_C': panel_temp,
            'future_ldr': future_ldr
        })

    def generate_dataset(self):
        """Generate complete dataset with all days"""
        print("=" * 70)
        print("SOLAR DATASET GENERATOR (ADVANCED)")
        print("=" * 70)
        print(f"Generating {self.config.num_days} days of solar data...")
        print(f"Samples per day: {self.config.N_per_day}")
        print(f"Sample interval: {self.config.sample_interval_s}s")
        print(f"Prediction horizon: {self.config.horizon_seconds}s ({self.config.horizon_samples} samples)")
        print("-" * 70)

        all_days = []
        start_time = time.time()

        for d in range(self.config.num_days):
            df = self.generate_day(d)

            # Save individual day
            fname = self.output_path / f"raw_day_{d+1:03d}.csv"
            df.to_csv(fname, index=False)

            all_days.append(df)
            print(f"  Day {d+1:3d}: {len(df):5d} samples -> {fname.name}")

        # Combine all days
        df_all = pd.concat(all_days, ignore_index=True)
        df_all.to_csv(self.output_path / "all_days_raw.csv", index=False)
        print(f"\n[COMBINED] all_days_raw.csv ({len(df_all)} samples)")

        # Create normalized version
        df_normalized = self.create_normalized_dataset(df_all)
        df_normalized.to_csv(self.output_path / "all_days_normalized.csv", index=False)
        print(f"[NORMALIZED] all_days_normalized.csv")

        # Feature count
        feature_count = len([c for c in df_normalized.columns if c not in ['day', 'time_s']])
        print(f"\nNormalized Features: {feature_count} (excluding metadata)")

        # Generate summary statistics
        self.generate_summary(df_all, df_normalized)

        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"COMPLETE! Generated in {elapsed:.1f}s")
        print(f"Output directory: {self.output_path.resolve()}")
        print(f"{'='*70}")

    def create_normalized_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create normalized dataset with engineered features for TinyML"""
        df_norm = df.copy()

        # Time-of-day features (sine/cosine encoding)
        seconds_in_day = 24 * 3600
        tod_frac = (df_norm['time_s'] % seconds_in_day) / seconds_in_day
        df_norm['time_sin'] = np.sin(2 * np.pi * tod_frac)
        df_norm['time_cos'] = np.cos(2 * np.pi * tod_frac)

        # Normalized LDR (already 0-1)
        df_norm['ldr_norm'] = df_norm['ldr'].clip(0, 1)
        df_norm['irradiance_norm'] = df_norm['irradiance_true'].clip(0, 1)

        # Normalized battery voltage
        df_norm['battery_norm'] = (
            (df_norm['battery_voltage'] - self.config.V_min) /
            (self.config.V_max - self.config.V_min)
        ).clip(0, 1)

        # Normalized battery SoC
        df_norm['battery_soc_norm'] = (df_norm['battery_soc'] / 100.0).clip(0, 1)

        # Normalized panel temperature (0-70°C range)
        df_norm['panel_temp_norm'] = (df_norm['panel_temp_C'] / 70.0).clip(0, 1)

        # Rate of change features (useful for prediction)
        df_norm['ldr_diff'] = df_norm.groupby('day')['ldr_norm'].diff().fillna(0).clip(-0.5, 0.5)
        df_norm['battery_diff'] = df_norm.groupby('day')['battery_norm'].diff().fillna(0).clip(-0.1, 0.1)

        # Moving averages for trend detection
        df_norm['ldr_ma_short'] = df_norm.groupby('day')['ldr_norm'].transform(
            lambda x: x.rolling(window=6, min_periods=1, center=True).mean()
        )
        df_norm['ldr_ma_long'] = df_norm.groupby('day')['ldr_norm'].transform(
            lambda x: x.rolling(window=24, min_periods=1, center=True).mean()
        )

        # Variance indicator
        df_norm['ldr_std_short'] = (
            df_norm.groupby('day')['ldr_norm']
            .transform(lambda x: x.rolling(window=12, min_periods=1).std().fillna(0))
            .clip(0, 0.3)
        )

        # Target variable (keep NaNs)
        df_norm['future_ldr_norm'] = df_norm['future_ldr']

        # Select final columns
        columns = [
            'day', 'time_s', 'time_sin', 'time_cos',
            'ldr_norm', 'battery_norm', 'battery_soc_norm', 'panel_temp_norm',
            'irradiance_norm', 'ldr_diff', 'battery_diff',
            'ldr_ma_short', 'ldr_ma_long', 'ldr_std_short',
            'future_ldr_norm'
        ]

        return df_norm[columns]

    def generate_summary(self, df_raw: pd.DataFrame, df_norm: pd.DataFrame):
        """Generate and save summary statistics"""
        summary = {
            'Total samples': len(df_raw),
            'Total days': self.config.num_days,
            'Samples per day': self.config.N_per_day,
            'Sample rate': f"{self.config.fs} Hz ({self.config.sample_interval_s}s interval)",
            'LDR mean': f"{df_raw['ldr'].mean():.3f}",
            'LDR std': f"{df_raw['ldr'].std():.3f}",
            'Battery mean': f"{df_raw['battery_voltage'].mean():.3f} V",
            'Battery std': f"{df_raw['battery_voltage'].std():.3f} V",
            'Battery range': f"[{df_raw['battery_voltage'].min():.3f}, {df_raw['battery_voltage'].max():.3f}] V",
            'SoC mean': f"{df_raw['battery_soc'].mean():.1f}%",
            'Panel temp mean': f"{df_raw['panel_temp_C'].mean():.1f}°C"
        }

        print("\n" + "=" * 70)
        print("DATASET SUMMARY")
        print("=" * 70)
        for key, value in summary.items():
            print(f"{key:20s}: {value}")
        print("=" * 70)

        # Save summary
        with open(self.output_path / "dataset_summary.txt", 'w') as f:
            f.write("SOLAR DATASET SUMMARY (ADVANCED)\n")
            f.write("=" * 70 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")


def main():
    """Main entry point"""
    config = SimulationConfig()
    generator = SolarDataGenerator(config)
    generator.generate_dataset()


if __name__ == "__main__":
    main()
