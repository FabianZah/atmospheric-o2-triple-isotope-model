# OXYTIB 0.1.0 release-candidate checklist

This checklist separates automated acceptance, expert review, archival
metadata, and deployment. A permanent v0.1.0 tag and DOI are created only after
all release gates are complete.

## Automated scientific and software gates

| Gate | Required result | Current status |
|---|---|---|
| `python run_model.py validate` | zero release blockers | pass |
| Integrated scientific acceptance | accepted steady, inverse, and declared transients | pass |
| Modern Δ′¹⁷O and δ¹⁸O checks | within declared observational gates | pass |
| Dense operational-domain audit | finite and monotonic | pass |
| Forward-inverse closure | all three coordinates recovered | pass |
| Public API and browser contracts | regression suite passes | pass |
| Production container known answer | health and identity checks pass | pass on current deployment |
| Historical Young response gate | 26 of 26 checks pass | pass |

The authoritative scientific details and qualifications are in
[`publication_model_acceptance.md`](publication_model_acceptance.md). The table
above is a release summary and does not replace that report.

## Expert review gates

- [ ] Three domain experts receive the same tagged release candidate.
- [ ] Reviewers record the tested tag or commit.
- [ ] All release blockers and major findings are resolved or explicitly
      adjudicated.
- [ ] Scientific wording, notation, uncertainty, and scope are accepted.
- [ ] At least one independent local installation and validation succeeds.
- [ ] Solver, proxy conversion, isotope field, transients, and exports are
      covered collectively by the reviews.
- [ ] Reviewer changes are rerun through the complete automated test suite.

## Archival metadata gates

- [ ] Final software authors and order are confirmed.
- [ ] Author ORCID identifiers are added where available.
- [ ] Version 0.1.0 release date is added to the changelog and citation files.
- [ ] Repository URL, license, and citation records are verified.
- [ ] The final Git tag and GitHub release are created from a passing commit.
- [ ] Zenodo archives that exact release and issues the DOI.
- [ ] The DOI is added to citation records and the public How to cite panel.
- [ ] A metadata-only follow-up tag, if needed for the DOI, is documented
      without altering the archived scientific implementation.

## Deployment gates

- [ ] The release image is built from the final tag.
- [ ] Image digest and preceding deployment are recorded for rollback.
- [ ] Container health, logs, and known-answer verification pass.
- [ ] The canonical `/oxytib/` interface, API documentation, one solver run,
      one transient, and one export pass on the public server.
- [ ] No obsolete public route exposes a second model instance.

## Release decision

Release v0.1.0 only when automated gates pass, expert-review findings are
closed, citation metadata are final, and the tagged image passes deployment
verification. Suggestions that do not affect scientific validity,
reproducibility, security, or the documented core workflows may be scheduled
for a later version.
