# Constrained coordinate inference

The publication interface infers one of pCO2, GPP, or pO2 only after the other
two coordinates have been constrained independently. Each coordinate constraint
is represented as one of:

- `fixed`: an exact value;
- `normal`: a Gaussian density defined by a central value and 1-sigma;
- `range`: uniform density between exact lower and upper bounds.

Gaussian coordinate constraints are evaluated over central value +/- 4 sigma,
clipped to the accepted output-surface domain, and renormalized over that
support. The solved coordinate receives uniform prior density in its reported
units over the accepted model domain. A logarithmic pCO2 plot axis is only a
display transformation; it does not imply a log-uniform prior.

The isotope likelihood is Gaussian. Direct-air inference uses both atmospheric
O2 Delta-prime-17O and conventional delta-18O when both measurements and their
analytical uncertainties are supplied. Spherule inference uses the atmospheric
Delta-prime-17O target and analytical uncertainty obtained from the Zahnow et
al. (2025) conversion.

When either constraining coordinate is uncertain, the corresponding joint
posterior is evaluated. If both constraints are uncertain, the coordinate not
shown in the two-dimensional field is marginalized. Every result contains the
solved-coordinate marginal density, median, and equal-tailed 95% credible
interval. The interface plots this marginal distribution and shades its 95%
interval.

If the solved-coordinate posterior occupies fewer than eight intervals of the
initial quadrature grid, the solver repeats the quadrature over the negligible-
tail support identified by the first pass. It redistributes the requested solved
axis points without increasing the API payload or workbook size. This adaptive
numerical refinement uses the same likelihood and prior densities. The result
records the initial and final solved axis bounds and sizes. A posterior that
reaches an actual input or operational-domain edge is not truncated away and
remains explicitly boundary-sensitive.

This probability calculation includes the central updated model, analytical
isotope uncertainty, and the user-supplied constraints. It does not silently
include structural model discrepancy. Structural sensitivity remains a
separate validation result.

The typed endpoint is `POST /api/v1/inference/coordinate`. The earlier
`POST /api/v1/inference/pco2` endpoint remains available for compatibility.
The public interface can repeat the typed request through
`POST /api/v1/export/coordinate.xlsx` to produce a workbook containing model
identity, input constraints, the solved marginal posterior, and the joint
probability field when one is calculated.
