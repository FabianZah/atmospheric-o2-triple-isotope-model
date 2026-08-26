# Changelog

All notable changes to the released software are documented here.

## 0.1.0 - Release candidate

- Name the public model OXYTIB: Atmospheric oxygen triple-isotope budget and
  inference model.
- Define one literature-updated atmospheric O₂ triple-isotope publication
  model over 0.1–2.0 PAL O₂, 50–60,000 ppm CO₂, and
  18.256–850 Pg C yr⁻¹ GPP.
- Provide steady forward calculations, constrained one-coordinate inversions,
  probability fields, isotope fields, and declared time-response experiments.
- Add state-step, photosynthesis-step, and prescribed gradual-pCO₂ transient
  experiments with data and metadata export.
- Add a stable console interface for forward, inverse, posterior, and transient
  calculations in JSON, CSV, or XLSX form.
- Provide a same-origin FastAPI browser interface and versioned HTTP API.
- Add persistent light and dark appearances and self-contained legends in
  exported scientific PNG figures.
- Export constrained solutions as CSV or XLSX and time-response experiments
  as provenance-bearing XLSX workbooks.
- Distribute separated measurement, parameter, numerical, and structural
  uncertainty metadata without combining them into an unsupported global
  confidence interval.
- Include compact observational and multi-model validation evidence, the
  historical Young et al. (2014) response anchor, and release regression tests.
- Include hardened Docker, Caddy, and shared-Traefik deployment templates with
  bounded request size, compute concurrency, traffic rate, and container
  resources.

The date and archival DOI will be added when version 0.1.0 is released.
