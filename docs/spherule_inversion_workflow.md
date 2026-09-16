# I-Type Cosmic-Spherule Inversion

The public web application and HTTP API use the single model pinned by
`model_data/publication_model_contract_v1.json`. The proxy workflow does not
select a separate spherule model or a historical reconstruction branch.

## Proxy conversion

Measured I-type cosmic-spherule isotope values are converted to atmospheric
O2 with Zahnow et al. (2025), Eq. 3, using the calibration of Fischer et al.
(2021):

```text
Delta-prime-17O_air = Delta-prime-17O_spherule
                      + 0.0285 * delta-18O_spherule - 1.005
```

All isotope values are in per mil. Spherule `delta-18O` is reported relative
to VSMOW and all `Delta-prime-17O` values use lambda = 0.528. The measured
spherule `delta-18O` must be supplied for each sample; the Fischer et al.
Table 5 mean is not used as a model default.

The analytical one-sigma uncertainty passed to the atmospheric inversion is

```text
sigma_air = sqrt(sigma_Delta17_spherule^2
                 + (0.0285 * sigma_delta18_spherule)^2)
```

The slope and intercept uncertainties reported by Fischer et al. are retained
as a separate calibration-sensitivity envelope because their covariance is
not reported. That envelope is not combined with analytical error or labelled
as a confidence interval.

## Scientific inference

One of `pCO2`, GPP, or `pO2` is solved at a time. The other two quantities must
be supplied as fixed values, Gaussian one-sigma constraints, or bounded
ranges. The result is therefore conditional on independent information; one
isotope measurement is not treated as an independent determination of all
three quantities.

The reported probability distribution contains:

- the propagated spherule analytical likelihood;
- the explicitly entered constraints on the other two coordinates;
- a bounded uniform prior for the solved coordinate in its reported units.

It does not include a log-uniform prior or a probabilistic whole-domain
structural-model error. Measurement/proxy, parameter, numerical, and
structural uncertainty remain separate under
`model_data/uncertainty/updated_o2_uncertainty_layers_v1.json`.

## Operational domain

The accepted model domain is:

| coordinate | minimum | maximum |
|---|---:|---:|
| pCO2 | 50 ppm | 60,000 ppm |
| GPP | 18.256 PgC/yr (6.30% modern) | 850 PgC/yr (293.10% modern) |
| pO2 | 0.10 PAL | 2.00 PAL |

Numerical extrapolation outside this domain is rejected. A boundary-sensitive
result means appreciable posterior probability reaches a model-domain edge;
it does not create or imply a solution beyond that edge.

## Public API sequence

The independent server exposes the same calculation used by the browser UI:

1. `POST /api/v1/proxy/spherule-to-air` converts the measured spherule and
   propagates analytical uncertainty.
2. `POST /api/v1/inference/coordinate` solves the selected coordinate while
   integrating the two declared constraints.
3. `POST /api/v1/export/coordinate.xlsx` exports the posterior, constraints,
   proxy inputs, model identifiers, and provenance in one workbook.

The API schema is available at `/docs` while the server is running. Server
installation and deployment are documented in `docs/server_deployment.md`.

## Acceptance gate

`validation/test_spherule_inversion_acceptance.py` performs an end-to-end
known-answer test with realistic analytical uncertainties of 0.060 per mil in
spherule `Delta-prime-17O` and 0.500 per mil in spherule `delta-18O`. It checks
recovery of pCO2, GPP, and pO2 under independent constraints, exact proxy
conversion, normalized probability mass, and both pCO2 domain-edge density
modes.
The test runs in continuous integration with the public API contract.
