# Geodesic Calculations in Earth Observation (EO)

## 1. The Geodetic foundation of EO

Satellite imagery does not exist on a planar Cartesian surface but on an irregular oblate spheroid (geoid). Treating Earth as a sphere or, worse, a plane, introduces significant metric errors in area, distance, and direction calculations, particularly at high latitudes and over large separate regions.

**Ununennium** implements rigorous ellipsoidal mathematics to ensure sub-meter accuracy for all derived metrics.

---

## 2. Earth Models

### 2.1 The Ellipsoid (WGS84)

The Earth is modeled as an oblate spheroid defined by:
*   **Semi-major axis ($a$):** 6,378,137.0 meters (Equatorial radius)
*   **Semi-minor axis ($b$):** 6,356,752.3142 meters (Polar radius)
*   **Inverse flattening ($1/f$):** 298.257223563

**Flattening ($f$):**

$$ f = \frac{a - b}{a} $$

**Eccentricity squared ($e^2$):**

$$ e^2 = \frac{a^2 - b^2}{a^2} = 2f - f^2 $$

### 2.2 The Geoid (EGM96/EGM2008)

The Geoid is the equipotential surface of the Earth's gravity field that creates the "Mean Sea Level." It deviates from the WGS84 ellipsoid by up to $\pm 100$ meters.
*   **Ellipsoidal Height ($h$):** Height above WGS84 ellipsoid (GPS native).
*   **Orthometric Height ($H$):** Height above Geoid (MSL).
*   **Geoid Undulation ($N$):** Separation between ellipsoid and geoid.

$$ h = H + N $$

---

## 3. Distance Calculations

### 3.1 Haversine Formula (Spherical Approximation)

For quick approximations (error ~0.3-0.5%), we assume a spherical Earth with radius $R \approx 6371$ km.

$$ a = \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos \phi_1 \cdot \cos \phi_2 \cdot \sin^2\left(\frac{\Delta\lambda}{2}\right) $$
$$ c = 2 \cdot \text{atan2}\left(\sqrt{a}, \sqrt{1-a}\right) $$
$$ d = R \cdot c $$

Where $\phi$ is latitude and $\lambda$ is longitude in radians.

### 3.2 Vincenty's Formulae (Ellipsoidal Truth)

For high-precision applications (Ununennium default), we use Vincenty's iterative algorithm on the ellipsoid. It is accurate to within 0.5mm.

**Direct Problem:** Given $\phi_1, \lambda_1, \alpha_1, s$, find $\phi_2, \lambda_2, \alpha_2$.
**Inverse Problem:** Given $\phi_1, \lambda_1, \phi_2, \lambda_2$, find $s, \alpha_1, \alpha_2$.

The core iteration solves for $\lambda$ difference ($L$) on the auxiliary sphere:

$$ \sin \sigma = \sqrt{(\cos U_2 \sin \Delta\lambda)^2 + (\cos U_1 \sin U_2 - \sin U_1 \cos U_2 \cos \Delta\lambda)^2} $$
$$ \cos \sigma = \sin U_1 \sin U_2 + \cos U_1 \cos U_2 \cos \Delta\lambda $$

Where $U$ is the reduced latitude: $\tan U = (1-f) \tan \phi$.

---

## 4. Area Calculations on the Ellipsoid

Calculating the area of a polygon defined by geodetic coordinates on an ellipsoid is non-trivial. Planar projection (e.g., UTM) distorts area. Ununennium uses **Albers Equal-Area Conic** projection dynamically or **Karney's Algorithm** for direct ellipsoidal integration.

**Geodesic Polygon Area:**

$$ Area = | \sum_{i=0}^{N-1} ( \lambda_{i+1} - \lambda_i ) \cdot ( \sin \phi_{i+1} + \sin \phi_i ) \cdot \frac{R^2}{2} | \quad \text{(Spherical approx)} $$

For the ellipsoid, the area is calculated by summing the areas of trapezoidal segments under the geodesic curve, corrected for the radius of curvature in the prime vertical ($N(\phi)$) and meridian ($M(\phi)$).

$$ M(\phi) = \frac{a(1-e^2)}{(1-e^2 \sin^2\phi)^{3/2}}, \quad N(\phi) = \frac{a}{\sqrt{1-e^2 \sin^2\phi}} $$

---

## 5. Distortion Analysis in Deep Learning

Using projected coordinate systems (PCS) like Web Mercator (EPSG:3857) feeds distorted logic into Convolutional Neural Networks (CNNs).

### 5.1 Tissot's Indicatrix

Tissot's indicatrix visualizes local distortion. A circle on the globe transforms into an ellipse on the map.

*   **Mercator:** Preserves shape (conformal), distorts area massively at poles.
    *   *Effect:* A model trained on global data thinks Greenland > Africa.
*   **Gall-Peters:** Preserves area, distorts shape.
    *   *Effect:* Objects are sheared, confusing rotation-variant kernels.

### 5.2 Ununennium's "GeoTensor" Solution

Our `GeoTensor` class dynamically handles correction factors during training:

1.  **Pixel-wise Area Maps:** An auxiliary input channel containing the real-world area ($m^2$) of each pixel.
2.  **Loss Weighting:** Small-area pixels (high latitudes in Mercator) are down-weighted; large-area pixels (equator) are up-weighted to equalize physical importance.

```python
# Ununennium Internal Logic
def compute_area_weights(latitudes, resolution):
    """Compute correction factor for Mercator distortion."""
    # Scale factor k = 1 / cos(lat)
    k = 1.0 / torch.cos(torch.deg2rad(latitudes))
    real_area = (resolution * k) ** 2
    return real_area / real_area.mean()
```

## 6. Coordinate Reference Systems (CRS) Taxonomy

Ununennium supports the full PROJ database.

| Type | Examples | Property Preserved | Use Case |
|------|----------|--------------------|----------|
| **Geographic** | WGS84 (EPSG:4326) | 3D Position | Raw Storage, GPS |
| **Projected (Cylindrical)** | UTM, Mercator | Angles (Conformal) | Navigation, Local Mapping |
| **Projected (Conic)** | Albers, Lambert | Area | Statistical Analysis, Macroscopic Training |
| **Projected (Azimuthal)** | Stereographic | Direction | Polar Regions |

> **Best Practice:** Always reproject datasets to an Equal-Area projection (e.g., Albers EPSG:6933, 9822) before calculating summary statistics or training segmentation models where class balance assumes area balance.
