# HOP–ODF Visualization

Synthetic visualization of fibre orientation and the corresponding  
**Orientation Distribution Function (ODF)** for prescribed  
**Hermans Orientation Parameter (HOP)** values.

---

## What This Repository Does

This project generates synthetic fibre cross-sections (line segments inside a disk) together with their corresponding ODFs.

The figure layout is:

- **Left column:** Fibre cross-section (reference direction shown by arrow)
- **Right column:** Orientation Distribution Function (ODF) in polar coordinates  
  (using α = 2θ for undirected fibres)

The purpose is to build physical intuition for how scalar orientational order (HOP) relates to the full angular distribution.

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
