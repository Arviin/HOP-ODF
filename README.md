# HOP–ODF Visualization

Synthetic visualization of fibre orientation and the corresponding  
**Orientation Distribution Function (ODF)** for prescribed  
**Hermans Orientation Parameter (HOP)** values.

---
## What This Repository Does

This repository provides a controlled, synthetic visualization of fibre orientation statistics and their relationship to the Hermans Orientation Parameter (HOP).

It generates synthetic fibre cross-sections (line segments within a disk) together with their corresponding Orientation Distribution Functions (ODFs), for prescribed HOP values spanning:

- transverse alignment (HOP < 0),
- isotropic orientation (HOP = 0),
- axial alignment (HOP > 0).

The figure layout is:

- **Left column:** Real-space fibre configuration in a 2D cross-section (reference direction indicated by an arrow)
- **Right column:** Orientation Distribution Function (ODF) shown in polar coordinates  
  (using the standard undirected-fibre mapping α = 2θ to represent θ ∈ [0°, 180°) on a full 0–360° polar axis)

---

## Scientific Context

The purpose of this project is to build physical intuition for how a scalar orientational order parameter (HOP) relates to the full angular distribution of fibre orientations.

The Hermans Orientation Parameter,

\[
HOP = \frac{1}{2} \left( 3 \langle \cos^2\phi \rangle - 1 \right),
\]

represents the second Legendre moment of the orientation distribution relative to a reference axis. While HOP provides a compact scalar measure of orientational order, it does not uniquely define the full ODF. Different angular distributions may yield identical HOP values.

This repository therefore illustrates:

- how increasing |HOP| sharpens the angular distribution,
- how the sign of HOP distinguishes axial vs transverse preferential alignment,
- and how scalar order parameters relate to full angular statistics.

---

## Relation to Ongoing Research

This work is part of ongoing research on the application of **directional neutron dark-field imaging (DFI)** to probe anisotropic microstructures in cellulose-based materials.

Directional neutron dark-field imaging is sensitive to small-angle scattering anisotropy, which is directly related to underlying fibre orientation distributions. Understanding the relationship between HOP and the full ODF is therefore essential for interpreting anisotropic dark-field signals and linking imaging contrast to physical microstructure.

The present code provides a synthetic and reproducible framework to support that interpretation.


---

## Interpretation of HOP

The Hermans Orientation Parameter is defined as:

\[
HOP = \frac{1}{2}\left(3\langle \cos^2\phi \rangle - 1\right)
\]

Its theoretical range is:

- **−0.5** → perfect perpendicular alignment  
- **0** → isotropic (random orientation)  
- **1** → perfect alignment with the reference direction  

Interpretation:

- **HOP > 0**: preferential alignment with the reference axis  
- **HOP = 0**: no preferred direction  
- **HOP < 0**: preferential alignment perpendicular to the reference axis  

---

## Reproducibility

- Dependencies are pinned via **Pixi** (`pixi.toml` and `pixi.lock`)
- Random seeds are fixed for deterministic outputs
- Figures are generated algorithmically from synthetic orientation distributions

---

## Run with Pixi (Recommended)

```bash
pixi run hop-odf

```
---

## Author

**Arvin (Fazel) Mirzaei**  
Postdoctoral Researcher  ; fazel.mirzaei@psi.ch
Paul Scherrer Institute (PSI), Switzerland  

This project was developed as part of research activities related to fibre orientation analysis and manuscript figure preparation.  

Paul Scherrer Institute (PSI) is Switzerland’s largest research institute for natural and engineering sciences, focusing on cutting-edge research in energy, materials, and health.
