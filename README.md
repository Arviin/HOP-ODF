\# HOP–ODF Visualization (Synthetic)



This repository generates \*\*synthetic\*\* fibre cross-sections (line segments in a disk) and the corresponding

\*\*Orientation Distribution Functions (ODF)\*\* for prescribed \*\*Hermans Orientation Parameter (HOP)\*\* values,

including negative HOP values (perpendicular preference).



The figure is intended for manuscript-quality illustration and for building intuition:

\- \*\*Left column:\*\* fibre cross-section (reference direction shown by arrow)

\- \*\*Right column:\*\* ODF shown in polar form (α = 2θ for undirected fibres)



\## Reproducibility

\- Dependencies are pinned via \*\*Pixi\*\* (`pixi.toml` + `pixi.lock`).

\- Random seeds are fixed for deterministic outputs.



\## Run with Pixi (recommended)

```bash

pixi run hop-odf

\## Interpretation of HOP



The Hermans Orientation Parameter (HOP) ranges from:



\- \*\*−0.5\*\* → perfect perpendicular alignment  

\- \*\*0\*\* → isotropic (random orientation)  

\- \*\*1\*\* → perfect alignment with the reference direction  



Positive HOP values indicate preferential alignment with the reference axis,

while negative values indicate preferential alignment perpendicular to it.

HOP = (3⟨cos²φ⟩ − 1) / 2



