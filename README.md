# Solar Energy Dataset Generator

A Python-based toolkit for generating realistic synthetic solar irradiance and battery voltage datasets for TinyML applications, energy harvesting research, and solar power prediction models.

## Overview

This project provides two dataset generators that simulate realistic solar panel behavior, battery dynamics, and environmental conditions:

1. **Indian Seasonal Dataset** - Comprehensive dataset modeling India's three distinct seasons (Summer, Monsoon, Winter)
2. **Minimal Dataset** - Simplified dataset for quick prototyping and testing

Both generators produce time-series data suitable for machine learning applications, particularly for edge devices and TinyML deployments.

## Features

### Indian Seasonal Dataset (`generate-indian-dataset.py`)

- **Multi-Season Simulation**: Models Summer (April-June), Monsoon (July-September), and Winter (November-February)
- **Realistic Weather Patterns**: Season-specific cloud coverage, humidity, and temperature effects
- **35 Days of Data**: 10 summer days, 15 monsoon days, 10 winter days
- **302,400 Samples**: High-resolution time-series data
- **Advanced Features**:
  - Season-specific irradiance factors
  - Variable cloud event rates and durations
  - Humidity effects on battery performance
  - Temperature-dependent charging efficiency
  - Atmospheric absorption modeling

### Minimal Dataset (`generate-dataset.py`)

- **Simplified Simulation**: Generic solar patterns without seasonal variations
- **8 Days of Data**: Consistent conditions for baseline testing
- **69,120 Samples**: Sufficient for initial model development
- **Core Features**:
  - Clear-sky irradiance modeling
  - Poisson-distributed cloud events
  - Basic battery charge/discharge dynamics
  - Atmospheric turbulence

## Project Structure

```
ecsd_aat/
├── generate-indian-dataset.py          # Indian seasonal dataset generator
├── generate-dataset.py                 # Minimal dataset generator
├── solar_india_dataset/                # Indian seasonal dataset output
│   ├── all_days_raw.csv               # Combined raw data (all 35 days)
│   ├── all_days_normalized.csv        # Normalized features for TinyML
│   ├── raw_day_001_summer.csv         # Individual day files
│   ├── raw_day_002_summer.csv
│   ├── ...
│   ├── raw_day_035_winter.csv
│   ├── season_summer.csv              # Season-specific aggregated data
│   ├── season_monsoon.csv
│   ├── season_winter.csv
│   └── dataset_summary.txt            # Statistical summary
├── solar_minimal_dataset/              # Minimal dataset output
│   ├── all_days_raw.csv               # Combined raw data (all 8 days)
│   ├── all_days_normalized.csv        # Normalized features for TinyML
│   ├── raw_day_001.csv                # Individual day files
│   ├── ...
│   ├── raw_day_008.csv
│   └── dataset_summary.txt            # Statistical summary
├── solar_india_dataset.zip             # Compressed archive
└── solar_minimal_dataset.zip           # Compressed archive
```

## Data Format

### Raw Data Columns

#### Indian Seasonal Dataset
| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `day` | int | 1-35 | Day number |
| `season` | string | summer/monsoon/winter | Season identifier |
| `time_s` | int | 0-43200 | Time in seconds since sunrise (12-hour daylight) |
| `irradiance_true` | float | 0.0-1.0 | Ground truth normalized irradiance |
| `ldr` | float | 0.0-1.0 | LDR sensor reading (with noise) |
| `battery_voltage` | float | 3.2-4.2 | Battery voltage in volts |
| `battery_soc` | float | 0-100 | Battery State of Charge (%) |
| `panel_temp_C` | float | 15-70 | Solar panel temperature (°C) |
| `future_ldr` | float | 0.0-1.0 | LDR value 60 seconds ahead (prediction target) |

#### Minimal Dataset
| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `day` | int | 1-8 | Day number |
| `time_s` | int | 0-43200 | Time in seconds since sunrise |
| `irradiance_true` | float | 0.0-1.0 | Ground truth normalized irradiance |
| `ldr` | float | 0.0-1.0 | LDR sensor reading (with noise) |
| `battery_voltage` | float | 3.2-4.2 | Battery voltage in volts |
| `battery_soc` | float | 0-100 | Battery State of Charge (%) |
| `panel_temp_C` | float | 15-70 | Solar panel temperature (°C) |
| `future_ldr` | float | 0.0-1.0 | LDR value 60 seconds ahead (prediction target) |

### Normalized Data (TinyML-Ready)

The normalized datasets include engineered features optimized for machine learning:

#### Indian Seasonal Dataset Features
- `time_sin`, `time_cos` - Cyclical time encoding
- `ldr_norm` - Normalized LDR reading [0, 1]
- `battery_norm` - Normalized battery voltage [0, 1]
- `battery_soc_norm` - Normalized State of Charge [0, 1]
- `panel_temp_norm` - Normalized panel temperature [0, 1]
- `irradiance_norm` - Normalized irradiance [0, 1]
- `ldr_diff` - Rate of change in LDR
- `battery_diff` - Rate of change in battery voltage
- `ldr_ma_short` - Short-term moving average (30s window)
- `ldr_ma_long` - Long-term moving average (2min window)
- `ldr_std_short` - Short-term variance indicator
- `season_summer`, `season_monsoon`, `season_winter` - One-hot season encoding
- `future_ldr_norm` - Prediction target

#### Minimal Dataset Features
- `time_sin`, `time_cos` - Cyclical time encoding
- `ldr_norm` - Normalized LDR reading [0, 1]
- `battery_norm` - Normalized battery voltage [0, 1]
- `battery_soc_norm` - Normalized State of Charge [0, 1]
- `panel_temp_norm` - Normalized panel temperature [0, 1]
- `irradiance_norm` - Normalized irradiance [0, 1]
- `ldr_diff` - Rate of change in LDR
- `battery_diff` - Rate of change in battery voltage
- `ldr_ma_short` - Short-term moving average (30s window)
- `ldr_ma_long` - Long-term moving average (2min window)
- `ldr_std_short` - Short-term variance indicator
- `future_ldr_norm` - Prediction target

## Installation

### Requirements

- Python 3.7+
- NumPy
- Pandas

### Setup

```bash
# Clone or download the project
cd ecsd_aat

# Install dependencies
pip install numpy pandas

# Or use requirements.txt if available
pip install -r requirements.txt
```

## Usage

### Generate Indian Seasonal Dataset

```bash
python generate-indian-dataset.py
```

**Output:**
- 35 individual day CSV files (10 summer, 15 monsoon, 10 winter)
- Combined raw dataset: `all_days_raw.csv` (302,400 samples)
- Normalized dataset: `all_days_normalized.csv`
- Season-specific files: `season_summer.csv`, `season_monsoon.csv`, `season_winter.csv`
- Summary statistics: `dataset_summary.txt`

**Configuration:**
Edit the `SimulationConfig` class in the script to customize:
- Number of days per season
- Sampling rate (default: 0.2 Hz, every 5 seconds)
- Battery parameters
- Prediction horizon
- Season-specific parameters (irradiance, cloud patterns, temperature)

### Generate Minimal Dataset

```bash
python generate-dataset.py
```

**Output:**
- 8 individual day CSV files
- Combined raw dataset: `all_days_raw.csv` (69,120 samples)
- Normalized dataset: `all_days_normalized.csv`
- Summary statistics: `dataset_summary.txt`

**Configuration:**
Edit the `SimulationConfig` class to customize:
- Number of days (default: 8)
- Sampling rate (default: 0.2 Hz)
- Cloud event parameters
- Battery parameters
- Prediction horizon

### Sample Terminal Output

#### Indian Seasonal Dataset Generation

```
======================================================================
INDIAN SOLAR DATASET GENERATOR (ADVANCED)
======================================================================
Location: India (simulated for Central India region)
Samples per day: 8640
Sample interval: 5s
Prediction horizon: 60s
Seasons (days): Summer=10, Monsoon=15, Winter=10
Total days: 35
----------------------------------------------------------------------

[SUMMER] April-June
  Day   1:  8640 samples -> raw_day_001_summer.csv
  Day   2:  8640 samples -> raw_day_002_summer.csv
  ...
  Day  10:  8640 samples -> raw_day_010_summer.csv

[MONSOON] July-September
  Day  11:  8640 samples -> raw_day_011_monsoon.csv
  Day  12:  8640 samples -> raw_day_012_monsoon.csv
  ...
  Day  25:  8640 samples -> raw_day_025_monsoon.csv

[WINTER] November-February
  Day  26:  8640 samples -> raw_day_026_winter.csv
  Day  27:  8640 samples -> raw_day_027_winter.csv
  ...
  Day  35:  8640 samples -> raw_day_035_winter.csv

[COMBINED] all_days_raw.csv (302400 samples)
[NORMALIZED] all_days_normalized.csv

Normalized Features: 16 (excluding metadata)
  - Time encoding (sin/cos)
  - LDR, Battery, Panel Temperature (normalized)
  - Rate of change features
  - Moving averages (short & long term)
  - Variance indicators
  - Season one-hot encoding
======================================================================
[SEASON] season_summer.csv (86400 samples)
[SEASON] season_monsoon.csv (129600 samples)
[SEASON] season_winter.csv (86400 samples)

======================================================================
DATASET SUMMARY
======================================================================
Total samples:                 302400
Total days:                    35
Sample rate:                   0.2 Hz (5s)

Per-Season Statistics:

  SUMMER:
    Days: 10
    Avg LDR: 0.520
    Avg Battery: 4.046 V
    Avg SoC: 54.5%
    Avg Panel Temp: 54.2°C
    Irradiance range: [0.000, 0.936]

  MONSOON:
    Days: 15
    Avg LDR: 0.214
    Avg Battery: 3.881 V
    Avg SoC: 50.8%
    Avg Panel Temp: 34.7°C
    Irradiance range: [0.000, 0.495]

  WINTER:
    Days: 10
    Avg LDR: 0.339
    Avg Battery: 3.992 V
    Avg SoC: 52.8%
    Avg Panel Temp: 28.6°C
    Irradiance range: [0.000, 0.639]

Overall Statistics:
  LDR: mean=0.337, std=0.239
  Battery: mean=3.960V, range=[3.200, 4.200]V
  SoC: mean=52.4%, range=[49.1, 59.2]%
======================================================================

======================================================================
COMPLETE! Generated in 14.7s
Output directory: /home/dibo/my_stuff/project_shit/ecsd_aat/solar_india_dataset
======================================================================
```

**Key Observations**:
- **Seasonal Variation**: Monsoon shows 59% reduction in LDR compared to summer (0.214 vs 0.520)
- **Temperature Effects**: Panel temperature varies from 28.6°C (winter) to 54.2°C (summer)
- **Battery Performance**: Lower SoC during monsoon (50.8%) due to reduced solar input
- **Generation Speed**: 20,550 samples/second
- **Data Quality**: All values within physical bounds, smooth transitions

#### Minimal Dataset Generation

```
======================================================================
SOLAR DATASET GENERATOR (ADVANCED)
======================================================================
Generating 8 days of solar data...
Samples per day: 8640
Sample interval: 5s
Prediction horizon: 60s (12 samples)
----------------------------------------------------------------------
  Day   1:  8640 samples -> raw_day_001.csv
  Day   2:  8640 samples -> raw_day_002.csv
  Day   3:  8640 samples -> raw_day_003.csv
  Day   4:  8640 samples -> raw_day_004.csv
  Day   5:  8640 samples -> raw_day_005.csv
  Day   6:  8640 samples -> raw_day_006.csv
  Day   7:  8640 samples -> raw_day_007.csv
  Day   8:  8640 samples -> raw_day_008.csv

[COMBINED] all_days_raw.csv (69120 samples)
[NORMALIZED] all_days_normalized.csv

Normalized Features: 13 (excluding metadata)

======================================================================
DATASET SUMMARY
======================================================================
Total samples       : 69120
Total days          : 8
Samples per day     : 8640
Sample rate         : 0.2 Hz (5s interval)
LDR mean            : 0.478
LDR std             : 0.272
Battery mean        : 4.031 V
Battery std         : 0.355 V
Battery range       : [3.200, 4.200] V
SoC mean            : 54.5%
Panel temp mean     : 39.9°C
======================================================================

======================================================================
COMPLETE! Generated in 2.3s
Output directory: /home/dibo/my_stuff/project_shit/ecsd_aat/solar_minimal_dataset
======================================================================
```

**Key Observations**:
- **Consistent Conditions**: No seasonal variations, ideal for baseline testing
- **Realistic Physics**: Panel temperature (39.9°C) and battery SoC (54.5%) show proper modeling
- **Generation Speed**: 30,052 samples/second (faster due to simpler configuration)
- **Feature Parity**: Includes all advanced features (SoC, panel temp) despite being "minimal"

#### Example Usage Script Output

```
======================================================================
SOLAR DATASET EXAMPLE USAGE
======================================================================
✓ Loaded normalized dataset: 302400 samples

======================================================================
DATASET ANALYSIS
======================================================================

Shape: (302400, 19)
Columns: ['day', 'season', 'time_s', 'time_sin', 'time_cos', 'ldr_norm', 
          'battery_norm', 'battery_soc_norm', 'panel_temp_norm', 
          'irradiance_norm', 'ldr_diff', 'battery_diff', 'ldr_ma_short', 
          'ldr_ma_long', 'ldr_std_short', 'season_summer', 'season_monsoon', 
          'season_winter', 'future_ldr_norm']

Seasons: ['summer' 'monsoon' 'winter']

Samples per season:
monsoon    129600
summer      86400
winter      86400

LDR (normalized) Statistics:
  Mean: 0.3371
  Std:  0.2386
  Min:  0.0000
  Max:  0.9593

Battery (normalized) Statistics:
  Mean: 0.7595
  Std:  0.4046

Missing values:
future_ldr_norm    420
======================================================================

======================================================================
SEASONAL COMPARISON
======================================================================

SUMMER:
  Avg LDR: 0.5196
  Avg Battery: 0.8455
  Samples: 86400

MONSOON:
  Avg LDR: 0.2138
  Avg Battery: 0.6806
  Samples: 129600

WINTER:
  Avg LDR: 0.3394
  Avg Battery: 0.7920
  Samples: 86400
======================================================================

======================================================================
MACHINE LEARNING PREPARATION
======================================================================

Feature columns (15):
  - time_sin, time_cos
  - ldr_norm, battery_norm, battery_soc_norm, panel_temp_norm
  - irradiance_norm, ldr_diff, battery_diff
  - ldr_ma_short, ldr_ma_long, ldr_std_short
  - season_summer, season_monsoon, season_winter

Target variable: future_ldr_norm
Valid samples: 301980 / 302400 (99.9%)
Feature matrix shape: (301980, 15)

✓ Data ready for training!
  Example usage:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
======================================================================

✓ Dataset successfully loaded and prepared!

Next steps:
  1. Split data into train/test sets
  2. Train your model (e.g., neural network, random forest)
  3. Evaluate performance on test set
  4. Deploy to edge device (TinyML)
======================================================================
```

**Key Observations**:
- **Data Completeness**: 99.9% of samples valid for ML training (only 420 NaN at prediction horizon boundary)
- **Feature Engineering**: 15 features automatically prepared for TinyML
- **Seasonal Balance**: Monsoon has 50% more samples (15 days vs 10 days for summer/winter)
- **Ready for ML**: Data properly normalized, no preprocessing needed



### Indian Seasonal Dataset

- **Total Samples**: 302,400
- **Total Days**: 35
- **Sample Rate**: 0.2 Hz (5-second intervals)
- **Daylight Hours**: 12 hours per day

**Season Breakdown:**
- **Summer** (10 days): High irradiance (avg LDR: 0.626), minimal clouds, high battery voltage (avg: 4.107V)
- **Monsoon** (15 days): Low irradiance (avg LDR: 0.307), heavy clouds, moderate battery voltage (avg: 3.998V)
- **Winter** (10 days): Moderate irradiance (avg LDR: 0.426), occasional clouds, good battery voltage (avg: 4.066V)

### Minimal Dataset

- **Total Samples**: 69,120
- **Total Days**: 8
- **Sample Rate**: 0.2 Hz (5-second intervals)
- **Average LDR**: 0.617
- **Average Battery**: 4.104V
- **Battery Range**: 3.2V - 4.2V

## Use Cases

1. **TinyML Model Training**: Train lightweight neural networks for solar energy prediction on microcontrollers
2. **Energy Harvesting Research**: Simulate and test energy management algorithms
3. **Battery Management Systems**: Develop and validate battery charging/discharging strategies
4. **Weather Pattern Analysis**: Study the impact of seasonal variations on solar energy
5. **Edge Computing**: Test real-time prediction models for IoT devices
6. **Educational Purposes**: Learn about solar energy systems and time-series forecasting

## Technical Details & Algorithm Logic

This section provides detailed explanations of the physics-based models and algorithms used in the dataset generators.

### 1. Solar Irradiance Modeling

#### Clear-Sky Irradiance

The base solar irradiance follows a sinusoidal pattern representing the sun's path across the sky:

```
I_base(t) = sin(π × t / T_day)
```

Where:
- `I_base(t)` = Normalized irradiance at time t [0, 1]
- `t` = Time since sunrise (seconds)
- `T_day` = Total daylight duration (43,200 seconds = 12 hours)

#### Air Mass Calculation

Air mass (AM) represents the path length of sunlight through the atmosphere. We use the **Kasten-Young formula** for accuracy:

```
AM = 1 / (cos(θ_z) + 0.50572 × (96.07995 - θ_z_deg)^(-1.6364))
```

Where:
- `θ_z` = Zenith angle (radians) = π/2 - elevation_angle
- `θ_z_deg` = Zenith angle in degrees
- AM is capped at 10 for very low sun angles

**Physical Significance**: AM = 1 when sun is directly overhead (zenith), AM = 2 at 60° from zenith. Higher AM means more atmospheric attenuation.

#### Atmospheric Transmission

Using the **Beer-Lambert Law** for atmospheric absorption:

```
T_atm = exp(-τ × AM)
```

Where:
- `T_atm` = Atmospheric transmission [0, 1]
- `τ` = Optical depth (depends on season/weather)
  - Summer: τ = 0.15 (clear, dry air)
  - Monsoon: τ = 0.35 (humid, hazy air)
  - Winter: τ = 0.20 (moderate clarity)

**Final Clear-Sky Irradiance**:
```
I_clear(t) = I_base(t) × T_atm
```

### 2. Cloud Modeling

#### Cloud Event Generation

Clouds are modeled as **Poisson-distributed events** with exponential durations:

```
P(cloud at sample i) = λ
Duration ~ Exponential(μ_duration)
Depth ~ Normal(μ_depth, σ_depth)
```

Where:
- `λ` = Cloud event rate (probability per sample)
  - Summer: 1/800 (rare clouds)
  - Monsoon: 1/200 (frequent clouds)
  - Winter: 1/500 (moderate clouds)
- `μ_duration` = Mean cloud duration (seconds)
- `μ_depth` = Mean cloud optical depth (attenuation factor)

#### Cloud Transition Smoothing

To avoid unrealistic step changes, cloud entry/exit use **smooth transitions**:

```
For cloud entry (fade-in over n samples):
  I(t) = I_clear(t) × [1 - (i/n) × (1 - depth)]  for i = 0 to n

For cloud exit (fade-out over n samples):
  I(t) = I_clear(t) × [1 - ((n-i)/n) × (1 - depth)]  for i = 0 to n
```

This creates gradual, realistic cloud shadows.

#### Atmospheric Turbulence

High-frequency variations from atmospheric scintillation:

```
I_turb(t) = I(t) × [1 + N(0, σ_turb)]
```

Where `N(0, σ_turb)` is Gaussian noise with standard deviation σ_turb (0.025-0.08 depending on season).

### 3. Solar Panel Temperature Model

Panel temperature affects efficiency. We use the **NOCT-based model** with thermal inertia:

#### Steady-State Temperature

```
T_panel_ss = T_ambient + (NOCT - 20) × (G / 800)
```

Where:
- `T_panel_ss` = Steady-state panel temperature (°C)
- `T_ambient` = Ambient air temperature (°C)
- `NOCT` = Nominal Operating Cell Temperature (45°C for typical panels)
- `G` = Irradiance in W/m² (normalized irradiance × 1000)

#### Thermal Dynamics (First-Order Response)

Panels don't heat/cool instantly. We model thermal inertia:

```
T_panel(t) = T_panel(t-1) + α × [T_panel_ss(t) - T_panel(t-1)]
```

Where:
- `α = Δt / τ_thermal` = Thermal response coefficient
- `Δt` = Sample interval (5 seconds)
- `τ_thermal` = Thermal time constant (300 seconds = 5 minutes)

This creates realistic gradual temperature changes.

#### Temperature Effect on Power

Panel efficiency decreases with temperature:

```
η(T) = η_ref × [1 + β × (T_panel - 25)]
```

Where:
- `η(T)` = Temperature-dependent efficiency
- `η_ref` = Reference efficiency at 25°C (20%)
- `β` = Temperature coefficient (-0.004 per °C, i.e., -0.4%/°C)

**Result**: A 45°C panel produces ~8% less power than at 25°C.

### 4. Battery Model (Advanced)

#### State of Charge (SoC) Tracking

We use **Coulomb counting** to track battery charge:

```
SoC(t) = SoC(t-1) + (I_net × Δt) / (C_battery × 3600) × 100%
```

Where:
- `SoC(t)` = State of Charge at time t (0-100%)
- `I_net` = Net current (mA) = (P_net / V_battery)
- `Δt` = Time step (seconds)
- `C_battery` = Battery capacity (2600 mAh)

#### Power Balance

```
P_net = P_solar × η_charge × f_temp - P_idle
```

Where:
- `P_solar` = Solar panel power = I_norm × P_max × f_panel_temp
- `η_charge` = Charging efficiency (85%)
- `f_temp` = Battery temperature factor
- `P_idle` = Idle power consumption (50 mW)

#### Voltage-SoC Relationship

Li-ion battery voltage is non-linear with SoC. We model this:

**During Charging**:
```
dV = (P_net / 10000) × [1 - 0.7 × (SoC/100)]
```

The `[1 - 0.7 × SoC_norm]` term represents **reduced charging rate at high SoC** (constant-voltage phase).

**During Discharging**:
```
dV = (P_net / 8000) × [1 + 0.5 × (1 - SoC/100)]
```

The `[1 + 0.5 × (1 - SoC_norm)]` term represents **voltage sag at low SoC**.

#### Temperature Effects

**Battery Performance**:
```
f_temp = 1 - |T_ambient - 25| × 0.002
```

Batteries perform optimally at 25°C. Performance degrades at higher/lower temperatures.

**Humidity Self-Discharge** (Monsoon effect):
```
V_loss = humidity_factor × 0.0001 per sample
```

High humidity (0.85 in monsoon) increases self-discharge rate.

### 5. Sensor Modeling

#### LDR (Light Dependent Resistor)

The LDR measures irradiance with noise:

```
LDR(t) = clip(I_true(t) + N(0, 0.01), 0, 1)
```

Where `N(0, 0.01)` represents **measurement noise** (1% standard deviation).

### 6. Feature Engineering for TinyML

#### Time Encoding (Cyclical)

To represent time's cyclical nature:

```
time_sin = sin(2π × t / T_day)
time_cos = cos(2π × t / T_day)
```

This allows the model to understand that 23:59 is close to 00:01.

#### Rate of Change Features

```
ldr_diff(t) = ldr_norm(t) - ldr_norm(t-1)
battery_diff(t) = battery_norm(t) - battery_norm(t-1)
```

Clipped to [-0.5, 0.5] and [-0.1, 0.1] respectively to prevent outliers.

#### Moving Averages

**Short-term trend** (30-second window, 6 samples):
```
ldr_ma_short(t) = mean(ldr_norm[t-3:t+3])
```

**Long-term trend** (2-minute window, 24 samples):
```
ldr_ma_long(t) = mean(ldr_norm[t-12:t+12])
```

#### Variance Indicator

Signal stability measure:
```
ldr_std_short(t) = std(ldr_norm[t-12:t])
```

Low variance = stable conditions, high variance = rapidly changing (clouds passing).

### 7. Seasonal Parameters (Indian Dataset)

| Parameter | Summer | Monsoon | Winter | Physical Basis |
|-----------|--------|---------|--------|----------------|
| Irradiance Factor | 1.0 | 0.55 | 0.70 | Monsoon clouds reduce solar by 45% |
| Cloud Rate (λ) | 1/800 | 1/200 | 1/500 | Monsoon has 4× more clouds than summer |
| Cloud Depth (μ) | 0.20 | 0.55 | 0.35 | Monsoon clouds are thicker |
| Turbulence (σ) | 0.025 | 0.08 | 0.035 | Monsoon has more atmospheric turbulence |
| Ambient Temp | 38°C | 28°C | 18°C | Based on Indian climate data |
| Humidity | 0.25 | 0.85 | 0.45 | Monsoon is very humid |

### 8. Data Normalization

All features are normalized to [0, 1] range for TinyML efficiency:

```
battery_norm = (V - V_min) / (V_max - V_min)
             = (V - 3.2) / (4.2 - 3.2)

soc_norm = SoC / 100

panel_temp_norm = T_panel / 70  (assuming max 70°C)
```

### 9. Prediction Target

The target variable is **future LDR value** at a prediction horizon (default: 60 seconds):

```
future_ldr(t) = ldr(t + horizon)
```

For samples where `t + horizon` exceeds the day length, the target is `NaN` (unavailable data).

This formulation enables **time-series forecasting** models to predict solar irradiance 60 seconds ahead, useful for energy management decisions.

---

## Advanced Features Summary

| Feature | Indian Dataset | Minimal Dataset | Description |
|---------|---------------|-----------------|-------------|
| **Air Mass Calculation** | ✓ | ✓ | Kasten-Young formula for atmospheric attenuation |
| **Panel Temperature** | ✓ | ✓ | NOCT-based model with thermal inertia |
| **Battery SoC Tracking** | ✓ | ✓ | Coulomb counting with efficiency curves |
| **Temperature Compensation** | ✓ | ✓ | Battery & panel performance vs temperature |
| **Seasonal Variations** | ✓ | ✗ | Three distinct seasons with different parameters |
| **Humidity Effects** | ✓ | ✗ | Self-discharge rate increases with humidity |
| **Multi-Cloud Types** | ✓ | ✓ | Realistic cloud optical properties |
| **Data Quality Metrics** | ✓ | ✓ | Validation and statistics |



## Data Quality

- **Reproducible**: Fixed random seeds ensure consistent generation
- **Validated**: Physical constraints enforced (voltage limits, irradiance bounds)
- **Noise Modeling**: Realistic sensor noise and measurement uncertainty
- **Smooth Transitions**: Cloud events use gradual fade-in/fade-out
- **No Missing Values**: Complete time-series (except prediction horizon NaNs)

## License

This project is provided as-is for research and educational purposes.

## Contributing

Suggestions and improvements are welcome. Consider:
- Additional weather patterns (fog, dust storms)
- More sophisticated battery models
- Integration with real solar irradiance data
- Support for different geographic locations
- Multi-panel configurations

## Citation

If you use this dataset generator in your research, please cite:

```
Solar Energy Dataset Generator
https://github.com/Spidy104/ECSD_AAT_Dataset_Generator
```

## Contact

For questions, issues, or collaboration opportunities, please open an issue on the project repository.

---

**Generated with ❤️ for the TinyML and renewable energy research community**
