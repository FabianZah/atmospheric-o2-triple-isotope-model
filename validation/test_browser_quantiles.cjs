/* Analytic checks for the browser's posterior zoom quantiles. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../web/app.js'), 'utf8');
const start = source.indexOf('function posteriorQuantile(');
const end = source.indexOf('\nfunction drawMarginalPosterior(', start);
assert(start >= 0 && end > start);
const scope = {};
vm.runInNewContext(source.slice(start, end), scope);
const quantile = scope.posteriorQuantile;
const weights = axis => axis.map((x, i) =>
  (axis[Math.min(axis.length - 1, i + 1)] - axis[Math.max(0, i - 1)]) / 2);
const close = (actual, expected, tolerance = 1e-10) =>
  assert(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);

let checks = 0;
for (const axis of [[0, .01, .13, .7, .99, 1], [50, 80, 500, 2000, 60000]]) {
  for (const q of [0, .001, .025, .5, .975, .999, 1]) {
    close(quantile(axis, weights(axis), q), axis[0] + q * (axis.at(-1) - axis[0]));
    checks += 1;
  }
}
const axis = [0, .01, .13, .7, .99, 1];
for (const increasing of [true, false]) {
  const mass = weights(axis).map((w, i) => w * 2 * (increasing ? axis[i] : 1 - axis[i]));
  for (const q of [.001, .025, .5, .975, .999]) {
    const expected = increasing ? Math.sqrt(q) : 1 - Math.sqrt(1 - q);
    close(quantile(axis, mass, q), expected);
    close(quantile(axis, mass.map(m => 7 * m), q), expected);
    checks += 2;
  }
}
const gapAxis = [0, 1, 2, 3, 4, 5];
const gapMass = weights(gapAxis).map((w, i) => w * [1, 1, 0, 0, 1, 1][i]);
close(quantile(gapAxis, gapMass, .5), 2);
console.log(`${checks + 1} browser quantile checks passed`);
