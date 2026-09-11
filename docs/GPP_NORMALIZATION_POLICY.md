# GPP units and normalization

The physical model uses absolute global gross primary production in
Pg C per year. The browser reports GPP as a percentage of a defined
modern reference:

**100% modern GPP = 290 Pg C per year**, following Liang et al. (2023).

The conversion is `absolute GPP = percent modern / 100 * 290`.
Thus 25% modern corresponds to 72.5 Pg C per year. Console inputs in
Pg C per year are already absolute and need no percentage conversion.

## Literature comparisons

The normalization utilities also retain Young et al. (2014)'s
365.125 Pg C per year scale, Beerling (1999)'s 367.527 Pg C per year
gross oxygen-production equivalent, and explicit custom references.
These support literature comparisons. Percentages based on different
references are converted to absolute production before comparison.

Liang et al. (2023)'s reference uncertainty of ±30 Pg C per year is a
separate normalization sensitivity. A user-entered GPP uncertainty or range
is the constraint used by the public inference; the reference uncertainty is
not silently added to it.

## Reporting

Report the percentage together with its reference and the resulting absolute
GPP. This keeps biological throughput unambiguous across studies.
The model's absolute domain is 18.256264-850 Pg C per year, corresponding to
approximately 6.30-293.10% of the browser reference.

The implementation is `code/gpp_normalization.py`. Source conventions and
reference metadata accompany calculations that use the normalization utility.
