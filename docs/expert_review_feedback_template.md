# OXYTIB expert review feedback

## Reviewer and build

- Reviewer:
- Date:
- Scientific field or role:
- Tested Git tag or commit:
- Hosted or local interface:
- Browser and operating system:
- Local Python version, if applicable:

## Overall assessment

- Recommended decision: accept / accept after changes / repeat expert review
- Confidence in scientific interpretation: high / moderate / low
- Confidence in reproducibility: high / moderate / low
- Most important strength:
- Most important concern:

## Test record

| Review area | Pass / issue / not tested | Notes or exported evidence |
|---|---|---|
| Scope, notation, and references | | |
| Known-answer forward calculation | | |
| Forward-inverse closure | | |
| Fixed, 1σ, and range constraints | | |
| I-type cosmic-spherule conversion | | |
| Isotope field and contours | | |
| Time-response experiments | | |
| XLSX and PNG exports | | |
| Local validation | | |

## Findings

Copy this block once for each finding.

### Finding title

- Severity: release blocker / major / minor / suggestion
- Interface or command:
- Inputs:
- Expected behavior:
- Observed behavior:
- Scientific or practical consequence:
- Reproduction steps:
- Attached evidence:

## Scientific review

- Are the model purpose and conditional nature of inference clear?
- Are the physical assumptions and operational limits stated appropriately?
- Are isotope notation, units, reference scales, and PAL/GPP definitions clear?
- Are the uncertainty displays scientifically interpretable?
- Do any trends, transients, or boundary responses appear physically implausible?
- Are the declared validation qualifications sufficient and fair?
- Is any claim stronger than the evidence presented?
- Which additional validation would most improve confidence?

## Interface and reproducibility review

- Could a first-time user complete the main calculation without guidance?
- Did long calculations communicate progress adequately?
- Were rejected or boundary-limited solutions understandable?
- Could every exported result be interpreted independently?
- Did `python run_model.py validate` complete successfully, if tested locally?
- Were any inputs, results, or browser settings retained unexpectedly?

## Final recommendation

State the changes required before OXYTIB 0.1.0 and separate them from useful
post-release extensions.
