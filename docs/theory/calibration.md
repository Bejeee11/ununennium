# Radiometric Calibration and Correction

## 1. The Physical Measurement Chain

Satellite remote sensing measures **Radiance** at the sensor, but physical models usually require **Reflectance** at the surface. Understanding this chain is vital for building robust ("physics-aware") models that generalize across time and sensors.

$$ \text{DN} \xrightarrow{\text{Gain/Offset}} L_{TOA} \xrightarrow{\text{Solar Angle}} \rho_{TOA} \xrightarrow{\text{Atmosphere}} \rho_{BOA} $$

### 1.1 Digital Number (DN)
The raw quantized integer value stored in the file (e.g., 12-bit uint).
*   *Dependent on:* Sensor gain, quantization.

### 1.2 Top-of-Atmosphere (TOA) Radiance ($L_{TOA}$)
Physical energy flux measured at the sensor ($W \cdot m^{-2} \cdot sr^{-1} \cdot \mu m^{-1}$).

$$ L_\lambda = \text{gain} \cdot \text{DN} + \text{offset} $$

### 1.3 TOA Reflectance ($\rho_{TOA}$)
Normalized unitless ratio of reflected to incident solar energy. Corrects for illumination geometry (solar zenith angle $\theta_s$) and Earth-Sun distance ($d$).

$$ \rho_{TOA} = \frac{\pi \cdot L_\lambda \cdot d^2}{\text{E}_{\text{sun},\lambda} \cdot \cos(\theta_s)} $$

Where $\text{E}_{\text{sun}}$ is mean solar exoatmospheric irradiances.

### 1.4 Bottom-of-Atmosphere (BOA) Reflectance ($\rho_{BOA}$)
Surface reflectance, corrected for atmospheric absorption (water vapor, ozone) and scattering (Rayleigh, Aerosol/Mie). Also known as **Surface Reflectance (SR)** or **L2A**.

$$ \rho_{BOA} \approx \frac{\rho_{TOA} - \rho_{atm}}{T \cdot (1 + S \cdot \rho_{BOA})} $$
(Simplified radiative transfer equation)

---

## 2. Deep Learning Implications

### 2.1 Why L2A (BOA) Matters
A CNN trained on L1C (TOA) data learns to implicitly model the atmosphere. If tested on a day with different aerosol load (hazy vs. clear), it fails.
*   **Invariant Feature:** Surface Reflectance ($\rho_{BOA}$) is an intrinsic property of the material.
*   **Action:** Always prefer L2A data for training rigorous models.

### 2.2 Calibration Loss
When training generative models (e.g., Cloud Removal GANs), the loss function must respect physical bounds.
*   $\rho \in [0, 1]$ (mostly).
*   **Spectral Consistency:** The ratio between bands (e.g., NDVI) must be preserved.

$$ \mathcal{L}_{spectral} = \| \frac{B_{NIR} - B_{RED}}{B_{NIR} + B_{RED}} - \text{GT}_{NDVI} \|_1 $$

---

## 3. Sensor Cross-Calibration

Combining Landsat-8 and Sentinel-2 requires Harmonization.

### 3.1 Spectral Response Functions (SRF)
Each sensor has slightly different band pass filters.
*   *Sentinel-2 Red:* 665nm center, 30nm width.
*   *Landsat-8 Red:* 655nm center, 37nm width.

### 3.2 SBAF (Spectral Band Adjustment Factor)
To use Landsat data in a Sentinel model, we apply a linear transformation derived from hyperspectral reference data (e.g., Hyperion).

$$ \rho_{S2} = a \cdot \rho_{L8} + b $$

Ununennium's `preprocessing.harmonization` module includes theoretically derived coefficients for common sensor pairs (S2, L8, L9, PlanetScope).

---

## 4. Radar (SAR) Calibration

Synthetic Aperture Radar requires different physics.

### 4.1 Radiometric Terrain Correction (RTC)
SAR geometry leads to foreshortening and layover. RTC normalizes the backscatter $\sigma^0$ by the true illuminated area of each pixel on the DEM.

$$ \sigma^0_{RTC} = \sigma^0 \cdot \frac{A_{flat}}{A_{local}} $$

### 4.2 Speckle Statistics
SAR noise (Speckle) is multiplicative, adhering to a Gamma distribution (multi-look).
*   **Log-Transform:** $y = \log(x)$ stabilizes variance, making noise additive and Gaussian-like, which is better for standard CNNs (MSE Loss).

> **Ununennium Default:** All SAR inputs are assumed to be $\gamma^0$ (Gamma Naught) RTC corrected and Log-scaled dB.
