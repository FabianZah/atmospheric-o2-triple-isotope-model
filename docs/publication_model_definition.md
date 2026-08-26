# Publication Model Definition

## One model

The manuscript and public interface use one atmospheric O2 triple-isotope
model. Its deterministic core is the resolved Photochem R1-R7 atmospheric
column coupled to a conservative global atmospheric-O2 and biological-turnover
budget. The validated output surface is a numerical accelerator for that same
model; it is not a fitted surrogate with independent equations.

The machine-readable authority is
`model_data/publication_model_contract_v1.json`. It pins the model and surface
identifiers, operating domain, entry points, uncertainty contract, validation
evidence, and checksums of the defining data files.

## Reporting

The raw model state is the primary forward result. A Pack (2021)-referenced
result may additionally be reported as the observed modern value plus the
unchanged modelled scenario-minus-modern differential. This reference-frame
translation neither modifies the model equations nor creates a second model.
Exports retain the raw modern state, raw scenario state, differential,
observation-referenced state, and modern structural residual.

## Uncertainty

Measurement/proxy, parameter, numerical, and structural uncertainty remain
separate. Literature parameter ranges enter the declared parameter layer.
Numerical holdouts constrain interpolation error. Comparisons with Young et
al. (2014), Liu et al. (2021), Cao and Bao (2013), Luz et al. (1999), the
Yang-Banerjee ice-core record, Brandon et al. (2020), and Pack (2021) are
validation evidence with source-specific scopes.

The Clima experiments are retained as non-probabilistic structural end
members. Their cloud-free RCE configuration is not sufficiently constrained
to replace the central temperature architecture, so their high-pCO2 response
is not applied as damping, an offset, or a second prediction branch.

## Accepted Scope

The integrated acceptance audit in `docs/publication_model_acceptance.md`
finds no release-blocking numerical, conservation, modern-reference, surface,
inverse, posterior, or uncertainty-provenance failures. The model is accepted
for steady forward calculations, constrained inversions, and the declared
step-response experiments within its operational domain.

This acceptance does not imply unique recovery of pCO2, GPP, and pO2 from one
isotope value, a fully simultaneous long-term carbon-oxygen cycle, or uniform
precision in the extreme high-pCO2/low-GPP corner.

## Manuscript Structure

The Methods section describes the one deterministic model and its inputs. The
Validation section compares its behavior with observations and published
models. The Uncertainty section propagates the four declared layers without
combining uncalibrated inter-model spread into a Gaussian confidence interval.
Historical reconstruction presets remain software provenance and validation
tools, not separate models in the manuscript analysis.
