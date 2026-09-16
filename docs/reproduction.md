# Reproduce scientific comparisons

Install the validation dependencies using the production constraints:

```powershell
python -m pip install -r code/requirements-dev.txt -c code/requirements-api-lock.txt
```

The console, API and comparisons use the
same OXYTIB model and versioned numerical input surfaces.

## Bundled-input calculations

```powershell
python run_model.py reproduce --list
python run_model.py reproduce
```

The default sequence recalculates eight comparisons. Each can also be named
individually after `reproduce`:

| Name | Calculation | Inputs |
|---|---|---|
| `fig8` | OXYTIB response and comparison with Young Fig. 8 | Digitized curve coordinates and OXYTIB surface |
| `cao-bao` | Cao and Bao (2013) equations and OXYTIB comparison | Published source constraints in `model_data/literature/` |
| `luz` | Luz et al. (1999) productivity equation and OXYTIB inversion | Published isotope and CO2 table |
| `banerjee` | GPP inference for pristine measured ice-core pairs | Normalized USAP-DC workbook table |
| `yang` | CO2 age matching, isotope tracking and blocked validation | Yang isotope table and Bereiter CO2 composite |
| `brandon` | Termination V transient productivity comparison | Brandon isotope table and Bereiter CO2 composite |
| `uncertainty` | Separate measurement, parameter and structural uncertainty checks | Versioned model and uncertainty inputs |
| `liu` | OXYTIB comparison in own-reference and matched-turnover coordinates | Archived external Liu model results, with OXYTIB predictions excluded from the reference table |

JSON, CSV and PNG files are written under `outputs/`. The reproduction manifest
records input and source-code hashes, package versions, command logs, durations
and completion status. Each run has its own timestamped directory;
`outputs/reproduction/manifest.json` describes the most recently completed run.
A failed calculation or numerical acceptance gate stops the sequence and
records an incomplete run. Rerunning an individual comparison replaces its
generated outputs.
Each subprocess has a one-hour limit; `--timeout-seconds` adjusts this limit
for local reproduction. Solver tolerances and scientific acceptance thresholds
are independent of this execution limit.

The Yang calculation preserves the source's modern-air-relative isotope
coordinate and reference slope. Its age-block cross-validation estimates the
reference offset independently of held-out data. Model response amplitudes
remain fixed. The Banerjee comparison uses measured CO2 and preserves the
authors' sample exclusions; its proxy-reconstructed CO2 column is not a
validation observation.

## Full numerical tests and Young comparison

```powershell
python -m pytest validation -q
python validation/audit_young_acceptance_gate.py
```

The tests cover conservation, inversion, isotope conversion, uncertainty
integration, transients, API contracts and exports. The comparison with
Young et al. (2014) uses a separate source-derived validation calculation. Some POSIX process tests run
on Linux rather than Windows.

## Climate and chemistry inputs

The evidence manifest names the generator for each of its 15 reports; all
those generators and their local Python dependencies are supplied. The 20
native atmospheric arrays for the climate comparison are bundled under
`validation/reference_data/climate/` (about 3 MB), outside the web-container
build. Their archived SHA-256 hashes are checked before use. Run:

```powershell
python run_model.py reproduce climate
```

This calculates local oxygen-isotope exchange responses for CO2 and excited
atomic oxygen, O(1D), and solves the global O2 isotope budget
for the four oxygen/pressure experiments. It is optional in the default
sequence because it takes longer than the surface-based comparisons.

Lower-level entry points are also supplied:

- `validation/merge_liu_2021_gpp_grid.py --input-directory PATH` reads either
  native Liu `column.json` results or the bundled fixed-reference grid and
  recalculates the OXYTIB comparisons.
- `validation/audit_clima_global_o2_response.py --manifest MANIFEST
  --artifact-directory DIRECTORY` constructs these isotope-exchange responses
  from Photochem atmospheric profiles and solves the global isotope budget. `--po2-pal` selects
  the oxygen level; run `--help` for output and bundle options.
- `validation/audit_clima_pressure_conventions.py` and
  `validation/audit_clima_po2_cross.py` compare the resulting climate reports.
- `validation/audit_marine_o2_accessibility.py` consumes the Liu comparison,
  scorecard and Yang reports for the marine-access sensitivity.

Native Photochem and Clima atmospheric generation uses external scientific
programs and source licenses. The bundled arrays are their numerical outputs,
not their program source. `reproduce climate` recomputes OXYTIB isotope responses
from these prescribed atmospheres; regeneration of the atmospheric profiles
themselves is a distinct external-model calculation. Similarly, `reproduce liu`
recalculates OXYTIB predictions against fixed Liu results, rather than claiming
to rerun Liu's model. Input metadata preserves this distinction.

## Evidence aggregation

`python run_model.py validate` performs a runtime smoke calculation and checks
the acceptance decision against the hash-verified evidence bundle. Its generated
reports go to `outputs/`, leaving tracked documentation intact.

The full scorecard and its supporting independent grid-interpolation, ice-core
tracking and transient reports can be regenerated explicitly:

```powershell
python run_model.py reproduce numerical
python run_model.py reproduce scorecard
```

`numerical` tests the physical solver against surface interpolation, including
the separate oxygen-18 grid and the low-CO2 extension. `scorecard` includes that
work plus the observational prerequisites and can take substantially longer
than the default comparison sequence. These routes preserve the source
validation thresholds and write reports under `outputs/`. They are separate
from the fast API/runtime installation check.

Regenerate the required comparisons before assembling an evidence bundle with
`validation/export_validation_evidence_bundle.py`. An aggregate report and a
fresh observational comparison have distinct provenance and should be cited
accordingly.
