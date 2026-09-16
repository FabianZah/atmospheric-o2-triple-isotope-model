# Forward Isotope Constraints

The model solver's **O2 isotope composition** option evaluates the same
steady-state output surface as the fixed forward calculation. Inputs are
pCO2 (ppm), absolute GPP (PgC/year), and pO2 (PAL). The interface converts
GPP from percent modern using its declared 290 PgC/year reference.

Each input may be fixed, normally distributed, or uniform over an exact range.
Constraints are independent. Normal distributions are restricted to their
center +/-4 sigma and the accepted model domain, then renormalized, matching
the inverse solver. The workbook records these effective bounds.

Outputs are atmospheric O2 Delta-prime-17O (lambda=0.528) and conventional
delta18O on VSMOW, both in per mil. Conversion from delta-prime-18O uses
`1000 * expm1(delta18_prime / 1000)` for every evaluation before calculating
statistics. Fixed inputs return the deterministic prediction. Uncertain inputs
return medians, means, standard deviations, and equal-tailed 95% propagated
intervals. These intervals describe entered input uncertainty in the central
model; model-parameter and structural uncertainty remain separate.

Propagation uses two reproducible scrambled Sobol sequences, refined by
doubling. Quantiles (2.5%, 50%, 97.5%), means, and standard deviations are
checked between replicates and against the preceding refinement. Both checks
must pass on two successive refinements, with tolerance 0.0001 per mil plus
0.0002 times each isotope's 95% interval width. These are empirical numerical
convergence checks, rather than a rigorous error bound. The calculation raises
an explicit error if it reaches the sample limit without convergence.

## Python

From the repository root, with the documented dependencies installed:

```python
import sys
sys.path.insert(0, "code")
from forward_isotope_constraints import ForwardIsotopeConstraints, predict_isotopes
from updated_constrained_pco2_posterior import CoordinateConstraint

result = predict_isotopes(ForwardIsotopeConstraints(
    pco2_constraint=CoordinateConstraint("fixed", center=800),
    gpp_constraint=CoordinateConstraint("normal", center=174, sigma=29),
    po2_constraint=CoordinateConstraint("range", lower=0.8, upper=1.2),
))
print(result["isotopes"])
```

## HTTP API

`POST /api/v1/forward/isotopes` accepts `pco2_constraint`, `gpp_constraint`,
and `po2_constraint`, each containing `kind` and either `center`,
`center` plus `sigma`, or `lower` plus `upper`.
`POST /api/v1/export/isotopes.xlsx` accepts the same request and exports the
results, input constraints, effective bounds, numerical checks, and provenance.
The existing fixed-input `/api/v1/forward` contract is unchanged.
