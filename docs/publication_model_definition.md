# OXYTIB model definition

## Architecture

OXYTIB is one atmospheric O₂ triple-isotope model. Its deterministic core
couples an altitude-resolved oxygen, ozone and carbon-dioxide photochemical
column to a conservative global atmospheric-O₂ and biological-turnover budget.
A precomputed model grid provides faster evaluation by interpolation, checked
against direct calculations using the same physical formulation.

The machine-readable authority is
`model_data/publication_model_contract_v1.json`. It pins the model and surface
identifiers, operating domain, entry points, uncertainty contract, validation
evidence, and checksums of the defining scientific files. Python and JSON
source checksums use LF-normalized text for consistent verification across
Windows and Linux checkouts.

## Reporting

Forward results report the raw mechanistic state. Modern observations,
including Pack (2021), provide independent validation targets. The isotope
composition solver reports Δ′¹⁷O at a reference slope of 0.528 and conventional
δ¹⁸O on VSMOW; the core forward endpoint also exposes logarithmic δ′¹⁸O.
Input uncertainties propagate to isotope intervals with their numerical
convergence diagnostics and input constraints retained in the export.

## Uncertainty

Measurement/proxy, parameter, numerical, and structural uncertainty are
reported as separate layers. Literature ranges define the parameter layer,
and direct calculations withheld from the interpolation grid constrain its
numerical error. Comparisons with Young
et al. (2014), Liu et al. (2021), Cao and Bao (2013), Luz et al. (1999), the
Yang-Banerjee ice-core record, Brandon et al. (2020), and Pack (2021) provide
validation evidence with source-specific scopes.

Cloud-free Clima radiative-convective-equilibrium experiments quantify a
non-probabilistic climate-architecture sensitivity at high pCO₂. Their results
remain a structural evidence layer alongside the central temperature
architecture.

## Accepted scope

The integrated audit in `docs/publication_model_acceptance.md` accepts OXYTIB
for steady forward calculations, constrained one-coordinate inference, isotope
fields, and the declared time-response experiments within the operational
domain.

Independent pCO₂, pO₂, or GPP constraints select the coordinate family used in
an inversion. Precision varies across the domain and is lowest in the extreme
high-pCO₂/low-GPP region. Long-term carbon and oxygen inventories enter through
the declared boundary conditions and transient experiments.

## Scientific presentation

The model is presented through its deterministic architecture, inputs,
observational and published-model validation, and four separated uncertainty
layers. Historical reconstruction calculations document the lineage from
Young et al. (2014) and serve as validation provenance for OXYTIB.
