# OXYTIB model definition

Local integration status: `unreleased-20261021`. The corrected chemistry and
fixed marine configuration have passed local integrated scientific and application
validation on Windows/Anaconda. Linux/container qualification remains a separate
software-release gate. The archived
v0.1.0 release remains reproducible at its Git tag and DOI
`10.5281/zenodo.22819964`. The current working tree is not that archived release.

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

Excited oxygen is quenched by N2 and O2 in the non-exchange quenching family.
CO2 removal is counted once in the complete CO2 reaction family, including its
exchange and non-exchange products. The forcing scale, 3.2726126929006267, uses
the original 400 ppm CO2-isoflux normalization with this accounting. The isotope
column spans 10-73 km on the fixed modern-reference atmospheric construction.

## Biological configurations

`modern_reference` is the default mixed terrestrial-marine budget.
`marine_endmember` is a fixed marine-only limiting configuration with its own
live budget and interpolation grid. Both support forward calculations, isotope
fields, proxy inversions, posterior integration and the declared time-response
experiments. The registry is `model_data/biosphere_configurations_v1.json`.

The marine configuration combines surface/interior uptake shares 0.95/0.05,
alpha18 values 0.978/0.988 and logarithmic exponents 0.514/0.516. Ratios are mixed
before deriving the effective exponent. Dissolved/air equilibrium is applied
once at a fixed 20 C and reference seawater composition. Its biological
parameters are fixed; its source-isoflux endpoints are propagated separately.
Biological parameter uncertainty for this endmember has not been quantified.
The photosynthesis experiment integrates carbon and oxygen together with shared
production and respiration fluxes. Slow geological O2 fluxes retain their balanced
initial background. The marine selection supplies the marine isotope budget;
ocean carbon exchange retains its first-order atmospheric-box parameterization.
Weathering and ocean-uptake coefficients use multiplier 1.00, with a separate
constant balancing flux calculated for the entered initial state. The rationale,
equations and sensitivity input are in [the carbon-budget definition](photosynthesis_carbon_budget.md).

Production is gross photosynthetic O2 expressed in Pg C-equivalent per year:
the dominant-O2 production flux is G*1e15/12.0107 mol/year. Respiration closes
the prescribed-O2 budget. External carbon productivity estimates require their
source-consistent conversion to this oxygen-equivalent convention.

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

Archived cloud-free Clima experiments used the preceding chemistry. Their
numerical offsets are excluded from this corrected model's uncertainty layer.
A climate sensitivity for this chemistry requires a separately regenerated,
consistently normalized calculation; it remains outside this integration.

## Validation status

The current contract marks acceptance evidence as pending regeneration. The
existing `docs/publication_model_acceptance.md` and bundled literature reports
describe the archived release and do not establish acceptance of this revision.
The integration gate remains closed until applicable current-model evidence
and full workflow checks have been reconciled.

Independent pCO₂, pO₂, or GPP constraints select the coordinate family used in
an inversion. Precision varies across the domain and is lowest in the extreme
high-pCO₂/low-GPP region. Long-term carbon and oxygen inventories enter through
the declared boundary conditions and transient experiments.

## Scientific presentation

The model is presented through its deterministic architecture, inputs,
observational and published-model validation, and four separated uncertainty
layers. Historical reconstruction calculations document the lineage from
Young et al. (2014) and serve as validation provenance for OXYTIB.
