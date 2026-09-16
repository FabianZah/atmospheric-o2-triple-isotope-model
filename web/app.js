"use strict";

const MODERN_GPP = 290.0;
const APPLICATION_BASE_PATH = (() => {
  const script = document.currentScript;
  if (!script?.src) return "";
  const pathname = new URL(script.src, window.location.href).pathname;
  const assetMarker = "/assets/";
  const markerIndex = pathname.lastIndexOf(assetMarker);
  return markerIndex >= 0 ? pathname.slice(0, markerIndex) : "";
})();
const state = {
  source: "air",
  solveFor: "pCO2",
  constraintModes: { pCO2: "fixed", GPP: "fixed", pO2: "fixed" },
  lastResult: null,
  lastTransient: null,
  metadata: null,
};
const $ = (id) => document.getElementById(id);

function preferredTheme() {
  try {
    const stored = window.localStorage.getItem("oxytib-theme");
    if (stored === "light" || stored === "dark") return stored;
  } catch (_error) {
    // Browser storage may be unavailable in privacy-restricted sessions.
  }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme, persist = true) {
  const dark = theme === "dark";
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  const toggle = $("theme-toggle");
  if (toggle) {
    toggle.checked = dark;
    toggle.setAttribute("aria-label", dark ? "Use light appearance" : "Use dark appearance");
  }
  const label = $("theme-label");
  if (label) label.textContent = dark ? "Light" : "Dark";
  if (persist) {
    try {
      window.localStorage.setItem("oxytib-theme", dark ? "dark" : "light");
    } catch (_error) {
      // The active theme still applies for this page.
    }
  }
}

function applicationUrl(path) {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${APPLICATION_BASE_PATH}${normalized}`;
}

function number(id) {
  return Number($(id).value);
}

function finite(value, label) {
  if (!Number.isFinite(value)) throw new Error(`${label} must be a finite number.`);
  return value;
}

function detailMessage(payload) {
  if (typeof payload?.detail === "string") return payload.detail;
  if (Array.isArray(payload?.detail)) return payload.detail.map((item) => item.msg).join("; ");
  return "The calculation could not be completed.";
}

function publicErrorMessage(error) {
  const message = error?.message || "The calculation could not be completed.";
  if (!/outside (?:the )?(?:model|mechanistic response) domain/i.test(message)) return message;

  const domain = state.metadata?.operational_domain || {};
  if (/GPP/i.test(message)) {
    const bounds = domain.GPP_percent_of_290_PgC_per_year || { minimum: 6.3, maximum: 293.1 };
    return `GPP must remain within ${format(bounds.minimum, 1)}–${format(bounds.maximum, 1)}% modern.`;
  }
  if (/pCO2/i.test(message)) {
    const bounds = domain.pCO2_ppm || { minimum: 50, maximum: 60000 };
    return `pCO₂ must remain within ${format(bounds.minimum, 0)}–${format(bounds.maximum, 0)} ppm.`;
  }
  if (/pO2/i.test(message)) {
    const bounds = domain.pO2_PAL || { minimum: 0.1, maximum: 2 };
    return `pO₂ must remain within ${format(bounds.minimum, 2)}–${format(bounds.maximum, 2)} PAL.`;
  }
  return "The entered constraints are outside the model domain.";
}

async function modelFetch(path, { onComputeState, ...options } = {}) {
  const tracked = options.method === "POST" && typeof onComputeState === "function";
  const token = tracked ? crypto.randomUUID() : null;
  const polling = new AbortController();
  let stopped = false;
  let timer;
  const poll = async () => {
    let delay = 5000;
    try {
      const response = await fetch(applicationUrl(`/api/v1/compute-status/${token}`), {
        signal: polling.signal, cache: "no-store",
      });
      if (response.ok) {
        const progress = await response.json();
        if (!stopped && ["waiting", "running"].includes(progress.state)) onComputeState(progress.state);
      } else if (response.status === 429) delay = 15000;
    } catch (_) {
      // A failed status check does not interrupt the calculation request.
    }
    if (!stopped) timer = window.setTimeout(poll, delay);
  };
  if (tracked) timer = window.setTimeout(poll, 300);
  try {
    return await fetch(applicationUrl(path), {
      ...options,
      headers: {
        "Content-Type": "application/json", ...(options.headers || {}),
        ...(tracked ? { "X-OXYTIB-Request-ID": token } : {}),
      },
    });
  } finally {
    stopped = true;
    window.clearTimeout(timer);
    polling.abort();
  }
}

async function api(path, options = {}) {
  const response = await modelFetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(detailMessage(payload));
  return payload;
}

function format(value, digits = 3) {
  if (!Number.isFinite(value)) return "not resolved";
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: 0,
  }).format(value);
}

function formatFixed(value, digits) {
  if (!Number.isFinite(value)) return "not resolved";
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);
}

function formatCoordinate(coordinate, value) {
  if (coordinate === "pCO2") return `${format(value, value >= 1000 ? 0 : 1)} ppm`;
  if (coordinate === "GPP") return `${format((100 * value) / MODERN_GPP, 1)}% modern`;
  return `${format(value, 2)} PAL`;
}

function validateSurfaceDomain(xmin, xmax, ymin, ymax) {
  const domain = state.metadata?.operational_domain || {};
  const pco2 = domain.pCO2_ppm || { minimum: 50, maximum: 60000 };
  const gpp = domain.GPP_PgC_per_year || { minimum: 18.256264, maximum: 850 };
  const gppPercent = domain.GPP_percent_of_290_PgC_per_year || {
    minimum: 100 * gpp.minimum / MODERN_GPP,
    maximum: 100 * gpp.maximum / MODERN_GPP,
  };
  if (xmin < pco2.minimum || xmax > pco2.maximum) {
    throw new Error(`pCO₂ limits must remain within ${format(pco2.minimum, 0)}–${format(pco2.maximum, 0)} ppm.`);
  }
  if (ymin < gppPercent.minimum || ymax > gppPercent.maximum) {
    throw new Error(
      `GPP limits must remain within ${format(gppPercent.minimum, 1)}–${format(gppPercent.maximum, 1)}% modern `
      + `(${format(gpp.minimum, 2)}–${format(gpp.maximum, 0)} Pg C yr⁻¹).`,
    );
  }
}

function coordinateLabel(coordinate) {
  if (coordinate === "pCO2") return "pCO<sub>2</sub>";
  if (coordinate === "pO2") return "pO<sub>2</sub>";
  return "GPP";
}

function currentForwardState() {
  return {
    p_o2_pal: finite(number("po2"), "pO2"),
    p_co2_ppm: finite(number("pco2"), "pCO2"),
    gpp_pgC_per_year: finite(number("gpp"), "GPP") * MODERN_GPP / 100,
  };
}

function coordinateConstraint(coordinate) {
  const mode = state.constraintModes[coordinate];
  const prefix = { pCO2: "pco2", GPP: "gpp", pO2: "po2" }[coordinate];
  const factor = coordinate === "GPP" ? MODERN_GPP / 100 : 1;
  const label = coordinate;
  if (mode === "fixed") {
    return { kind: "fixed", center: finite(number(prefix), label) * factor };
  }
  if (mode === "normal") {
    const sigma = finite(number(`${prefix}-sigma`), `${label} 1σ`) * factor;
    if (sigma <= 0) throw new Error(`${label} 1σ must be positive.`);
    return {
      kind: "normal",
      center: finite(number(prefix), label) * factor,
      sigma,
    };
  }
  const lower = finite(number(`${prefix}-lower`), `${label} lower bound`) * factor;
  const upper = finite(number(`${prefix}-upper`), `${label} upper bound`) * factor;
  if (upper <= lower) throw new Error(`${label} lower bound must be below its upper bound.`);
  return { kind: "range", lower, upper };
}

function constraintText(coordinate, constraint) {
  const display = (value) => formatCoordinate(coordinate, value);
  const label = coordinateLabel(coordinate);
  if (constraint.kind === "fixed") return `${label} = ${display(constraint.center)}, fixed`;
  if (constraint.kind === "normal") return `${label} = ${display(constraint.center)} ± ${display(constraint.sigma)} (1σ)`;
  return `${label} = ${display(constraint.lower)}–${display(constraint.upper)}, exact range`;
}

function setConstraintMode(coordinate, mode) {
  state.constraintModes[coordinate] = mode;
  const prefix = { pCO2: "pco2", GPP: "gpp", pO2: "po2" }[coordinate];
  document.querySelectorAll(`#${prefix}-constraint-mode button`).forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  $(`${prefix}-normal-fields`).classList.toggle("hidden", mode !== "normal");
  $(`${prefix}-range-fields`).classList.toggle("hidden", mode !== "range");
  const labels = {
    pCO2: ["pCO<sub>2</sub> <span>ppm</span>", "pCO<sub>2</sub> central value <span>ppm</span>"],
    GPP: ["GPP <span>% modern</span>", "GPP central value <span>% modern</span>"],
    pO2: ["pO<sub>2</sub> <span>PAL</span>", "pO<sub>2</sub> central value <span>PAL</span>"],
  };
  $(`${prefix}-value-label`).innerHTML = labels[coordinate][mode === "normal" ? 1 : 0];
  updateCoordinateControls();
}

function requiredSulfateNumber(id, label) {
  if (!$(id).value.trim()) throw new Error(`${label} is required.`);
  return finite(number(id), label);
}

function sulfateConstraint(prefix, label, factor) {
  const kind = $(`${prefix}-mode`).value;
  const value = (suffix, title) => requiredSulfateNumber(`${prefix}${suffix}`, `${label} ${title}`) * factor;
  if (kind === "range") {
    const lower = value("-lower", "lower bound");
    const upper = value("-upper", "upper bound");
    if (lower >= upper) throw new Error(`${label} requires lower < upper.`);
    return { kind, lower, upper };
  }
  const center = value("", "value");
  if (kind === "fixed") return { kind, center };
  const sigma = value("-sigma", "1σ");
  if (sigma <= 0) throw new Error(`${label} 1σ must be positive.`);
  return { kind, center, sigma };
}

const sulfateTreatments = {
  none: "No fractionation assumed",
  peng_2026_irreversible: "Irreversible O₂ incorporation (Peng et al., 2026; 25 °C)",
  peng_2026_equilibrium: "Equilibrated O₂ incorporation (Peng et al., 2026; 25 °C)",
  specified: "Custom isotope shifts",
};

function updateSulfateControls() {
  for (const prefix of ["sulfate-f", "sulfate-b"]) {
    const mode = $(`${prefix}-mode`).value;
    $(`${prefix}-center-fields`).classList.toggle("hidden", mode === "range");
    $(`${prefix}-normal-fields`).classList.toggle("hidden", mode !== "normal");
    $(`${prefix}-range-fields`).classList.toggle("hidden", mode !== "range");
  }
  const treatment = $("sulfate-fractionation").value;
  const peng = treatment.startsWith("peng_");
  $("sulfate-incorporation-inputs").classList.toggle("hidden", peng);
  $("sulfate-pathway-fraction").classList.toggle("hidden", !peng);
  $("sulfate-advanced").classList.toggle("hidden", treatment !== "specified");
  $("sulfate-treatment-note").textContent = peng ? "Sulfite oxidation · Peng et al., 2026" : "";
  $("sulfate-treatment-note").classList.toggle("hidden", !peng);
}

function sulfateObservation() {
  const assumption = $("sulfate-fractionation").value;
  if (!Object.hasOwn(sulfateTreatments, assumption)) throw new Error("Select an incorporation treatment.");
  let transfer = {};
  if (assumption === "specified") {
    const log18 = requiredSulfateNumber("sulfate-log18", "δ′¹⁸O transfer shift");
    const shift17 = requiredSulfateNumber("sulfate-shift17", "Δ′¹⁷O transfer shift");
    transfer = {
      alpha18_air_to_sulfate: finite(Math.exp(log18 / 1000), "α18"),
      alpha17_air_to_sulfate: finite(Math.exp((shift17 + .528 * log18) / 1000), "α17"),
      theta_air_to_sulfate: null,
    };
  }
  return {
    measured_cap_delta17_permil: requiredSulfateNumber("sulfate-d17", "Sulfate Δ′¹⁷O"),
    measured_delta18_permil: requiredSulfateNumber("sulfate-d18", "Sulfate δ¹⁸O"),
    cap_delta17_sigma_permil: requiredSulfateNumber("sulfate-d17-sigma", "Sulfate Δ′¹⁷O 1σ"),
    delta18_sigma_permil: requiredSulfateNumber("sulfate-d18-sigma", "Sulfate δ¹⁸O 1σ"),
    isotope_error_correlation: 0,
    incorporation: assumption.startsWith("peng_") ? { kind: "fixed", center: .25 }
      : sulfateConstraint("sulfate-f", "Air-derived oxygen fraction", 0.01),
    background: sulfateConstraint("sulfate-b", "Non-air oxygen Δ′¹⁷O", 1),
    fractionation_treatment: assumption,
    ...transfer,
  };
}

async function isotopeTarget(onComputeState) {
  if (state.source === "sulfate") {
    const sulfate = sulfateObservation();
    return { source: "Sulfate", sulfate, sigma: sulfate.cap_delta17_sigma_permil };
  }
  if (state.source === "air") {
    return {
      target: finite(number("air-d17"), "Air Δ′17O"),
      sigma: finite(number("air-sigma"), "Analytical uncertainty"),
      delta18: finite(number("air-d18"), "Air δ18O"),
      delta18Sigma: finite(number("air-d18-sigma"), "Air δ18O analytical uncertainty"),
      source: "Direct air O2",
    };
  }
  const spherule = {
    delta17: finite(number("spherule-d17"), "Spherule Δ′17O"),
    delta17Sigma: finite(number("spherule-d17-sigma"), "Spherule Δ′17O uncertainty"),
    delta18: finite(number("spherule-d18"), "Spherule δ18O"),
    delta18Sigma: finite(number("spherule-d18-sigma"), "Spherule δ18O uncertainty"),
  };
  const payload = await api("/api/v1/proxy/spherule-to-air", {
    method: "POST",
    onComputeState,
    body: JSON.stringify({
      cap_delta17_spherule_permil: spherule.delta17,
      delta18_spherule_permil: spherule.delta18,
      cap_delta17_sigma_permil: spherule.delta17Sigma,
      delta18_sigma_permil: spherule.delta18Sigma,
      include_calibration_sensitivity: true,
    }),
  });
  const result = payload.result;
  $("converted-air").innerHTML = `Air O<sub>2</sub> Δ′<sup>17</sup>O<sub>0.528</sub> = ${format(result.cap_delta17_air_o2_permil, 3)} ± ${format(result.analytical_sigma_permil, 3)}‰`;
  return {
    target: result.cap_delta17_air_o2_permil,
    sigma: result.analytical_sigma_permil,
    delta18: null,
    delta18Sigma: null,
    source: "I-type cosmic spherule",
    spherule,
  };
}

function setBusy(button, busy, busyText) {
  if (!button.dataset.label) button.dataset.label = button.textContent;
  button.disabled = busy;
  button.textContent = busy ? busyText : button.dataset.label;
}

function beginCalculationProgress({ panelId, labelId, elapsedId, busyRegionId, buttonId, message }) {
  const panel = $(panelId);
  const label = $(labelId);
  const elapsed = $(elapsedId);
  const startedAt = performance.now();
  label.textContent = message;
  panel.classList.remove("hidden");
  $(busyRegionId).setAttribute("aria-busy", "true");

  const updateElapsed = () => {
    const seconds = Math.max(0, Math.floor((performance.now() - startedAt) / 1000));
    elapsed.textContent = `${seconds} ${seconds === 1 ? "second" : "seconds"} elapsed`;
  };
  updateElapsed();
  const timer = window.setInterval(updateElapsed, 500);
  const finish = () => {
    window.clearInterval(timer);
    panel.classList.add("hidden");
    $(busyRegionId).removeAttribute("aria-busy");
  };
  finish.onState = (status) => {
    label.textContent = status === "waiting" ? "Waiting for calculation capacity" : message;
    setBusy($(buttonId), true, status === "waiting" ? "Waiting…" : "Calculating…");
  };
  return finish;
}

function beginTransientProgress(type) {
  return beginCalculationProgress({
    panelId: "transient-progress",
    labelId: "transient-progress-label",
    elapsedId: "transient-progress-elapsed",
    busyRegionId: "view-transient",
    buttonId: "run-transient",
    message: type === "photosynthesis"
      ? "Solving coupled photosynthesis, carbon, and O₂ response"
      : "Solving atmospheric isotope response",
  });
}

function beginSolverProgress() {
  return beginCalculationProgress({
    panelId: "solver-progress",
    labelId: "solver-progress-label",
    elapsedId: "solver-progress-elapsed",
    busyRegionId: "view-solver",
    buttonId: "run-solver",
    message: state.solveFor === "isotopes" ? "Predicting atmospheric O₂ isotopes" : "Calculating constrained solution",
  });
}

function updateCoordinateControls() {
  const isForward = state.solveFor === "isotopes";
  $("observed-isotope-inputs").classList.toggle("hidden", isForward);
  $("pco2").disabled = state.solveFor === "pCO2";
  $("gpp").disabled = state.solveFor === "GPP";
  $("po2").disabled = state.solveFor === "pO2";
  for (const coordinate of ["pCO2", "GPP", "pO2"]) {
    const prefix = { pCO2: "pco2", GPP: "gpp", pO2: "po2" }[coordinate];
    const hideValue = coordinate !== state.solveFor && state.constraintModes[coordinate] === "range";
    $(`${prefix}-value-label`).classList.toggle("hidden", hideValue);
    $(prefix).classList.toggle("hidden", hideValue);
    $(`${prefix}-constraint-definition`).classList.toggle("hidden", coordinate === state.solveFor);
  }
  $("result-title").innerHTML = isForward ? "Atmospheric O₂ steady state" : `${coordinateLabel(state.solveFor)} solution`;
  const label = isForward ? "Calculate isotope composition" : "Calculate constrained solution";
  $("run-solver").dataset.label = label;
  if (!$("run-solver").disabled) $("run-solver").textContent = label;
}

function isotopeConstraintText(target) {
  if (target.sulfate) {
    const s = target.sulfate;
    const describe = (c, factor, digits, unit) => {
      const v = (x) => `${format(x * factor, digits)}${unit}`;
      if (c.kind === "range") return `${v(c.lower)}–${v(c.upper)}, exact range`;
      if (c.kind === "normal") return `${v(c.center)} ± ${v(c.sigma)} (1σ)`;
      return `${v(c.center)}, fixed`;
    };
    return `Sulfate Δ′<sup>17</sup>O<sub>0.528</sub> = ${format(s.measured_cap_delta17_permil, 3)} ± ${format(s.cap_delta17_sigma_permil, 3)}‰`
      + `<br>Sulfate δ<sup>18</sup>O<sub>VSMOW</sub> = ${format(s.measured_delta18_permil, 3)} ± ${format(s.delta18_sigma_permil, 3)}‰`
      + `<br>Air-derived oxygen = ${describe(s.incorporation, 100, 2, "%")}`
      + `<br>Non-air oxygen Δ′<sup>17</sup>O<sub>0.528</sub> = ${describe(s.background, 1, 3, "‰")}`
      + `<br>O₂ transfer: ${sulfateTreatments[s.fractionation_treatment]}`
      + (s.fractionation_treatment === "specified"
        ? `<br>δ′<sup>18</sup>O shift = ${format(1000 * Math.log(s.alpha18_air_to_sulfate), 3)}‰; Δ′<sup>17</sup>O shift = ${format(1000 * (Math.log(s.alpha17_air_to_sulfate) - .528 * Math.log(s.alpha18_air_to_sulfate)), 3)}‰`
        : "");
  }
  if (target.spherule) {
    const source = target.spherule;
    return `Spherule Δ′<sup>17</sup>O<sub>0.528</sub> = ${format(source.delta17, 3)} ± ${format(source.delta17Sigma, 3)}‰`
      + `<br>Spherule δ<sup>18</sup>O<sub>VSMOW</sub> = ${format(source.delta18, 3)} ± ${format(source.delta18Sigma, 3)}‰`
      + `<br>Converted air O<sub>2</sub> Δ′<sup>17</sup>O<sub>0.528</sub> = ${format(target.target, 3)} ± ${format(target.sigma, 3)}‰`;
  }
  const d17 = `Δ′<sup>17</sup>O<sub>0.528</sub> = ${format(target.target, 3)} ± ${format(target.sigma, 3)}‰`;
  if (!Number.isFinite(target.delta18)) return d17;
  return `${d17}<br>δ<sup>18</sup>O<sub>VSMOW</sub> = ${format(target.delta18, 3)} ± ${format(target.delta18Sigma, 3)}‰`;
}

function renderConstrainedCoordinate(result, target, inputs, constraints, request) {
  $("forward-d18-result").classList.add("hidden");
  const coordinate = result.solve_for;
  const [low, high] = result.equal_tailed_credible_interval;
  const central = result.posterior_median;
  const strongEdgeMode = result.solve_boundary_sensitive
    && result.solve_mode_at_boundary
    && result.solve_boundary_probability_mass >= 0.5;
  $("solver-empty").classList.add("hidden");
  $("solver-result").classList.remove("hidden");
  if (strongEdgeMode) {
    const boundaryIndex = result.solve_boundary_direction === "lower" ? 0 : result.solve_axis.length - 1;
    const boundary = formatCoordinate(coordinate, result.solve_axis[boundaryIndex]);
    $("result-coordinate").innerHTML = `${coordinateLabel(coordinate)} constrained solution`;
    $("result-value").textContent = "No interior solution";
    $("result-interval").textContent = `Compatibility increases toward the ${boundary} domain boundary`;
  } else {
    $("result-coordinate").innerHTML = `${coordinateLabel(coordinate)} posterior median`;
    $("result-value").textContent = formatCoordinate(coordinate, central);
    $("result-interval").textContent = `${formatCoordinate(coordinate, low)}–${formatCoordinate(coordinate, high)} 95% credible interval`;
  }
  const isotopeLines = isotopeConstraintText(target).split("<br>");
  const coordinateLines = Object.entries(constraints)
    .map(([name, constraint]) => constraintText(name, constraint));
  $("result-constraints").innerHTML = [...isotopeLines, ...coordinateLines].join("<br>");
  const hasField = Array.isArray(result.field_density)
    && Array.isArray(result.field_probability_mass)
    && Array.isArray(result.field_x_axis);
  $("solver-marginal").classList.toggle("hidden", hasField);
  if (hasField) {
    clearPlot("solver-marginal-canvas");
  } else {
    $("solver-marginal-title").innerHTML = strongEdgeMode
      ? `${coordinateLabel(coordinate)} constrained compatibility`
      : `${coordinateLabel(coordinate)} probability distribution`;
    drawMarginalPosterior(
      result.solve_axis,
      result.solve_marginal_density,
      result.solve_marginal_probability_mass,
      coordinate,
      low,
      high,
      central,
      strongEdgeMode,
    );
  }
  let boundaryNote = "";
  if (strongEdgeMode) {
    const [domainLow, domainHigh] = result.final_solve_bounds;
    boundaryNote = `The entered isotope and fixed-coordinate constraints do not identify an interior ${coordinate} solution within the accepted ${formatCoordinate(coordinate, domainLow)}–${formatCoordinate(coordinate, domainHigh)} domain.`;
  } else if (result.solve_boundary_sensitive) {
    boundaryNote = `Posterior probability reaches the accepted ${coordinate} boundary, so the interval is boundary-limited.`;
  }
  $("solver-method-note").textContent = boundaryNote;
  $("solver-method-note").classList.toggle("hidden", !boundaryNote);

  $("solver-probability").classList.toggle("hidden", !hasField);
  if (hasField) {
    $("solver-probability-title").innerHTML = `${coordinateLabel(result.field_x_coordinate)}–${coordinateLabel(result.field_y_coordinate)} probability field`;
    drawProbabilityField(
      "solver-probability-canvas",
      result.field_x_coordinate,
      result.field_y_coordinate,
      result.field_x_axis,
      result.field_y_axis,
      result.field_density,
      result.field_probability_mass,
      result.field_hpd_mask,
      result.field_hpd_density_threshold,
    );
    const marginalized = Object.keys(constraints).find((name) => (
      constraints[name].kind !== "fixed"
      && name !== result.field_x_coordinate
      && name !== result.field_y_coordinate
    ));
    const marginalizedText = marginalized
      ? `The displayed solution accounts for the entered uncertainty in ${coordinateLabel(marginalized)} by combining results across its allowed values.`
      : "";
    $("solver-probability-caption").innerHTML = marginalizedText.trim();
  }
  $("download-result-xlsx").disabled = false;
  state.lastResult = {
    result, target, inputs, constraints, coordinate, request,
    solution: { central, low, high, intervalKind: "95% credible" },
  };
}

function renderForwardIsotopes(result, constraints, request) {
  $("solver-empty").classList.add("hidden");
  $("solver-result").classList.remove("hidden");
  $("forward-d18-result").classList.remove("hidden");
  const uncertain = result.uncertain_inputs.length > 0;
  $("result-coordinate").innerHTML = `Δ′<sup>17</sup>O<sub>0.528</sub>${uncertain ? " median" : ""}`;
  const d17 = result.isotopes.cap_delta17_prime_permil;
  const d18 = result.isotopes.delta18_conventional_permil;
  const interval = (value) => value.interval95
    ? `${format(value.interval95[0], 3)} to ${format(value.interval95[1], 3)}‰ · 95% propagated interval`
    : "";
  $("result-value").textContent = `${format(d17.median, 3)}‰`;
  $("result-interval").textContent = interval(d17);
  $("forward-d18-label").innerHTML = `δ<sup>18</sup>O<sub>VSMOW</sub>${uncertain ? " median" : ""}`;
  $("forward-d18-value").textContent = `${format(d18.median, 3)}‰`;
  $("forward-d18-interval").textContent = interval(d18);
  $("result-constraints").innerHTML = Object.entries(constraints)
    .map(([coordinate, constraint]) => constraintText(coordinate, constraint)).join("<br>");
  ["solver-marginal", "solver-probability", "solver-method-note"].forEach((id) => $(id).classList.add("hidden"));
  ["solver-marginal-canvas", "solver-probability-canvas"].forEach(clearPlot);
  $("download-result-xlsx").disabled = false;
  state.lastResult = { result, request, constraints, coordinate: "isotopes", forward: true };
}

async function runSolver() {
  const button = $("run-solver");
  $("solver-error").textContent = "";
  $("solver-result").classList.add("hidden");
  $("solver-empty").classList.remove("hidden");
  $("download-result-xlsx").disabled = true;
  state.lastResult = null;
  setBusy(button, true, "Calculating…");
  const endProgress = beginSolverProgress();
  const solveFor = state.solveFor;
  const solverControls = [...document.querySelectorAll("#view-solver .control-panel input, #view-solver .control-panel select, #view-solver .control-panel button")];
  const disabledBefore = solverControls.map((control) => control.disabled);
  solverControls.forEach((control) => { control.disabled = true; });
  try {
    if (solveFor === "isotopes") {
      const constraints = Object.fromEntries(["pCO2", "GPP", "pO2"].map((coordinate) => [coordinate, coordinateConstraint(coordinate)]));
      const request = { pco2_constraint: constraints.pCO2, gpp_constraint: constraints.GPP, po2_constraint: constraints.pO2 };
      const payload = await api("/api/v1/forward/isotopes", {
        method: "POST", onComputeState: endProgress.onState, body: JSON.stringify(request),
      });
      renderForwardIsotopes(payload.result, constraints, request);
      return;
    }
    const target = await isotopeTarget(endProgress.onState);
    const inputs = currentForwardState();
    if (target.sigma <= 0) throw new Error("A positive Δ′¹⁷O analytical uncertainty is required.");
    if (Number.isFinite(target.delta18) && target.delta18Sigma <= 0) {
      throw new Error("A positive δ¹⁸O analytical uncertainty is required for direct air.");
    }
    const constraints = {};
    for (const coordinate of ["pCO2", "GPP", "pO2"]) {
      if (coordinate !== state.solveFor) constraints[coordinate] = coordinateConstraint(coordinate);
    }
    const request = {
      solve_for: state.solveFor,
      ...(target.sulfate ? { sulfate: target.sulfate } : {
        target_air_cap_delta17_permil: target.target,
        measurement_sigma_permil: target.sigma,
      }),
      ...(Number.isFinite(target.delta18) ? {
        target_air_delta18_conventional_permil: target.delta18,
        delta18_measurement_sigma_permil: target.delta18Sigma,
      } : {}),
      pco2_grid_size: 181,
      gpp_grid_size: 81,
      po2_grid_size: 17,
    };
    const fieldNames = { pCO2: "pco2_constraint", GPP: "gpp_constraint", pO2: "po2_constraint" };
    for (const [coordinate, constraint] of Object.entries(constraints)) {
      request[fieldNames[coordinate]] = constraint;
    }
    const payload = await api("/api/v1/inference/coordinate", {
      method: "POST",
      onComputeState: endProgress.onState,
      body: JSON.stringify(request),
    });
    renderConstrainedCoordinate(payload.result, target, inputs, constraints, request);
  } catch (error) {
    $("solver-error").textContent = publicErrorMessage(error);
  } finally {
    endProgress();
    solverControls.forEach((control, index) => { control.disabled = disabledBefore[index]; });
    setBusy(button, false);
  }
}

async function downloadWorkbook() {
  if (!state.lastResult) return;
  const button = $("download-result-xlsx");
  const { request, target, coordinate } = state.lastResult;
  $("solver-error").textContent = "";
  setBusy(button, true, "Preparing…");
  try {
    const isForward = state.lastResult.forward;
    const context = isForward ? {} : {
      isotope_source: target.source,
      ...(target.spherule ? {
        spherule: {
          cap_delta17_permil: target.spherule.delta17,
          cap_delta17_sigma_permil: target.spherule.delta17Sigma,
          delta18_permil: target.spherule.delta18,
          delta18_sigma_permil: target.spherule.delta18Sigma,
        },
      } : {}),
    };
    const response = await modelFetch(isForward ? "/api/v1/export/isotopes.xlsx" : "/api/v1/export/coordinate.xlsx", {
      method: "POST",
      onComputeState: (status) => setBusy(button, true, status === "waiting" ? "Waiting…" : "Preparing…"),
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(isForward ? request : { inference: request, context }),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(detailMessage(payload));
    }
    const blob = await response.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = isForward ? "oxytib_isotope_prediction.xlsx" : `oxytib_${coordinate}_solution.xlsx`;
    link.click();
    URL.revokeObjectURL(link.href);
  } catch (error) {
    $("solver-error").textContent = publicErrorMessage(error);
  } finally {
    setBusy(button, false);
  }
}

function enablePlotExport(canvasId) {
  const button = document.querySelector(`.plot-export[data-canvas="${canvasId}"]`);
  if (button) button.disabled = false;
}

function clearPlot(canvasId) {
  const canvas = $(canvasId);
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  const button = document.querySelector(`.plot-export[data-canvas="${canvasId}"]`);
  if (button) button.disabled = true;
}

function downloadPlot(button) {
  const canvas = $(button.dataset.canvas);
  if (!canvas || button.disabled) return;
  canvas.toBlob((blob) => {
    if (!blob) return;
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${button.dataset.filename}.png`;
    link.click();
    URL.revokeObjectURL(link.href);
  }, "image/png");
}

function canvasContext(id, aspectRatio = 1.75) {
  const canvas = $(id);
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(520, canvas.clientWidth || 760);
  const height = width / aspectRatio;
  canvas.style.aspectRatio = String(aspectRatio);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  return { canvas, ctx, width, height };
}

const posteriorColorStops = [
  [242, 246, 246], [170, 207, 201], [67, 153, 147], [0, 111, 113], [22, 35, 91],
];
const isotopeColorStops = [[22, 35, 91], [33, 102, 172], [35, 169, 157], [242, 200, 75]];

function heatRgb(value, stops = posteriorColorStops) {
  const scaled = Math.max(0, Math.min(1, value)) * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(scaled));
  const fraction = scaled - index;
  return stops[index].map((channel, i) => Math.round(channel + fraction * (stops[index + 1][i] - channel)));
}

function heatColor(value, stops = posteriorColorStops) {
  return `rgb(${heatRgb(value, stops).join(",")})`;
}

function drawColorLegend(ctx, x, y, width, stops, title, lowLabel, highLabel, titleFontSize = 10) {
  const gradient = ctx.createLinearGradient(x, 0, x + width, 0);
  stops.forEach((color, index) => {
    gradient.addColorStop(index / (stops.length - 1), `rgb(${color.join(",")})`);
  });
  ctx.fillStyle = "#17212b";
  ctx.font = `600 ${titleFontSize}px system-ui`;
  ctx.textAlign = "left";
  ctx.fillText(title, x, y - 5);
  ctx.fillStyle = gradient;
  ctx.fillRect(x, y, width, 9);
  ctx.strokeStyle = "#52616a";
  ctx.lineWidth = 0.8;
  ctx.strokeRect(x, y, width, 9);
  ctx.fillStyle = "#43515a";
  ctx.font = "9px system-ui";
  ctx.fillText(lowLabel, x, y + 21);
  ctx.textAlign = "right";
  ctx.fillText(highLabel, x + width, y + 21);
}

function marginalDensityLabel(coordinate, edgeLimited = false) {
  if (edgeLimited) return "Compatibility";
  return "Relative density";
}

function drawMarginalLegend(ctx, width, coordinate, edgeLimited = false) {
  const y = 17;
  const starts = [58, Math.max(215, width * 0.34), Math.max(390, width * 0.67)];
  ctx.font = "10px system-ui";
  ctx.textAlign = "left";
  ctx.fillStyle = "rgba(33, 102, 172, 0.18)";
  ctx.fillRect(starts[0], y - 7, 22, 10);
  ctx.strokeStyle = "#2166ac";
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(starts[0], y - 2); ctx.lineTo(starts[0] + 22, y - 2); ctx.stroke();
  ctx.fillStyle = "#43515a";
  ctx.fillText(marginalDensityLabel(coordinate, edgeLimited), starts[0] + 28, y + 2);
  ctx.fillStyle = "rgba(193, 139, 40, 0.24)";
  ctx.fillRect(starts[1], y - 7, 22, 10);
  ctx.strokeStyle = "rgba(193, 139, 40, 0.7)";
  ctx.lineWidth = 1;
  ctx.strokeRect(starts[1], y - 7, 22, 10);
  ctx.fillStyle = "#43515a";
  ctx.fillText(edgeLimited ? "Domain-truncated 95% interval" : "Central 95% credible interval", starts[1] + 28, y + 2);
  ctx.strokeStyle = "#17212b";
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(starts[2] + 10, y - 9); ctx.lineTo(starts[2] + 10, y + 5); ctx.stroke();
  ctx.fillStyle = "#43515a";
  ctx.fillText(edgeLimited ? "Domain-truncated median" : "Posterior median", starts[2] + 20, y + 2);
}

function axisDisplayValues(coordinate, values) {
  return coordinate === "GPP" ? values.map((value) => 100 * value / MODERN_GPP) : values;
}

function axisLabel(coordinate, logarithmic = false) {
  if (coordinate === "pCO2") return `pCO₂ (ppm${logarithmic ? ", logarithmic display axis" : ""})`;
  if (coordinate === "GPP") return "GPP (% modern)";
  return "pO₂ (PAL)";
}

function axisTickValues(coordinate, values, pixelWidth = Infinity) {
  const minimum = values[0];
  const maximum = values[values.length - 1];
  if (coordinate === "pCO2") {
    const preferred = [50, 100, 150, 200, 250, 300, 350, 400, 500, 750, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 60000]
      .filter((value) => value >= minimum && value <= maximum);
    const candidates = preferred.length >= 3 ? preferred : [...new Set([0, 0.25, 0.5, 0.75, 1].map((fraction) => {
      const value = 10 ** (Math.log10(minimum) + fraction * (Math.log10(maximum) - Math.log10(minimum)));
      return Math.round(value);
    }))];
    if (!Number.isFinite(pixelWidth)) return candidates;
    const transformedMinimum = Math.log10(minimum);
    const transformedSpan = Math.log10(maximum) - transformedMinimum;
    const selected = [];
    let lastPixel = -Infinity;
    for (const value of candidates) {
      const pixel = (Math.log10(value) - transformedMinimum) / transformedSpan * pixelWidth;
      if (pixel - lastPixel >= 34) {
        selected.push(value);
        lastPixel = pixel;
      }
    }
    return selected;
  }
  const scale = niceAxisScale(minimum, maximum, 5);
  return scale.ticks.filter((value) => value >= minimum && value <= maximum);
}

function axisTickText(coordinate, value) {
  if (coordinate === "pCO2") return value >= 1000 ? `${format(value / 1000, 1)}k` : format(value, 0);
  if (coordinate === "GPP") return format(value, 0);
  return format(value, 2);
}

function posteriorQuantile(axis, probabilityMass, probability) {
  const total = probabilityMass.reduce((sum, value) => sum + value, 0);
  const target = probability * total;
  let cumulative = 0;
  for (let index = 0; index < probabilityMass.length; index += 1) {
    const previous = cumulative;
    cumulative += probabilityMass[index];
    if (cumulative >= target) {
      if (index === 0 || cumulative === previous) return axis[index];
      const fraction = (target - previous) / (cumulative - previous);
      return axis[index - 1] + fraction * (axis[index] - axis[index - 1]);
    }
  }
  return axis[axis.length - 1];
}

function drawMarginalPosterior(axis, density, probabilityMass, coordinate, low, high, median, edgeLimited = false) {
  const { ctx, width, height } = canvasContext("solver-marginal-canvas", 3.0);
  const margin = { left: 58, right: 20, top: 42, bottom: 52 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const xs = axisDisplayValues(coordinate, axis);
  const displayLow = coordinate === "GPP" ? 100 * low / MODERN_GPP : low;
  const displayHigh = coordinate === "GPP" ? 100 * high / MODERN_GPP : high;
  const displayMedian = coordinate === "GPP" ? 100 * median / MODERN_GPP : median;
  const logarithmic = coordinate === "pCO2";
  const displayDensity = density.map((value, index) => {
    if (coordinate === "pCO2") {
      // Transform density per ppm to density per log10(ppm).
      return value * Math.LN10 * axis[index];
    }
    if (coordinate === "GPP") {
      // Transform density per PgC/yr to density per percent modern.
      return value * MODERN_GPP / 100;
    }
    return value;
  });
  const transform = (value) => logarithmic ? Math.log10(value) : value;
  const displayValue = (value) => coordinate === "GPP" ? 100 * value / MODERN_GPP : value;
  const supportLow = displayValue(posteriorQuantile(axis, probabilityMass, 0.001));
  const supportHigh = displayValue(posteriorQuantile(axis, probabilityMass, 0.999));
  const domainMin = transform(xs[0]);
  const domainMax = transform(xs[xs.length - 1]);
  const supportMin = transform(supportLow);
  const supportMax = transform(supportHigh);
  const supportSpan = Math.max(supportMax - supportMin, (domainMax - domainMin) * 0.01);
  const xmin = edgeLimited ? domainMin : Math.max(domainMin, supportMin - 0.12 * supportSpan);
  const xmax = edgeLimited ? domainMax : Math.min(domainMax, supportMax + 0.12 * supportSpan);
  const visibleDensity = displayDensity.filter((value, index) => {
    const transformed = transform(xs[index]);
    return transformed >= xmin && transformed <= xmax;
  });
  const maximum = (Math.max(...visibleDensity, 0) || 1) * 1.08;
  const xPixel = (value) => margin.left + (transform(value) - xmin) / (xmax - xmin) * plotW;
  const yPixel = (value) => margin.top + plotH - value / maximum * plotH;

  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, width, height);
  drawMarginalLegend(ctx, width, coordinate, edgeLimited);
  ctx.save();
  ctx.beginPath();
  ctx.rect(margin.left, margin.top, plotW, plotH);
  ctx.clip();
  ctx.fillStyle = "rgba(193, 139, 40, 0.16)";
  ctx.fillRect(xPixel(displayLow), margin.top, xPixel(displayHigh) - xPixel(displayLow), plotH);
  ctx.beginPath();
  ctx.moveTo(xPixel(xs[0]), margin.top + plotH);
  xs.forEach((value, index) => ctx.lineTo(xPixel(value), yPixel(displayDensity[index])));
  ctx.lineTo(xPixel(xs[xs.length - 1]), margin.top + plotH);
  ctx.closePath();
  ctx.fillStyle = "rgba(33, 102, 172, 0.18)";
  ctx.fill();
  ctx.beginPath();
  xs.forEach((value, index) => {
    const x = xPixel(value);
    const y = yPixel(displayDensity[index]);
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#2166ac";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.strokeStyle = "#17212b";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(xPixel(displayMedian), margin.top);
  ctx.lineTo(xPixel(displayMedian), margin.top + plotH);
  ctx.stroke();
  ctx.restore();
  ctx.strokeStyle = "#26343d";
  ctx.lineWidth = 1;
  ctx.strokeRect(margin.left, margin.top, plotW, plotH);
  ctx.fillStyle = "#43515a";
  ctx.font = "11px system-ui";
  ctx.textAlign = "center";
  const visibleMinimum = logarithmic ? 10 ** xmin : xmin;
  const visibleMaximum = logarithmic ? 10 ** xmax : xmax;
  axisTickValues(coordinate, [visibleMinimum, visibleMaximum], plotW).forEach((tick) => {
    ctx.fillText(axisTickText(coordinate, tick), xPixel(tick), margin.top + plotH + 18);
  });
  ctx.fillText(axisLabel(coordinate, logarithmic), margin.left + plotW / 2, height - 12);
  ctx.save();
  ctx.translate(16, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(marginalDensityLabel(coordinate, edgeLimited), 0, 0);
  ctx.restore();
  enablePlotExport("solver-marginal-canvas");
}

function posteriorIndexBounds(probabilityMass, minimumIntervals = 6) {
  const total = probabilityMass.reduce((sum, value) => sum + value, 0);
  if (!(total > 0)) return [0, probabilityMass.length - 1];
  const lowerTarget = 0.001 * total;
  const upperTarget = 0.999 * total;
  let cumulative = 0;
  let lower = 0;
  let upper = probabilityMass.length - 1;
  for (let index = 0; index < probabilityMass.length; index += 1) {
    cumulative += probabilityMass[index];
    if (cumulative >= lowerTarget) { lower = index; break; }
  }
  cumulative = 0;
  for (let index = 0; index < probabilityMass.length; index += 1) {
    cumulative += probabilityMass[index];
    if (cumulative >= upperTarget) { upper = index; break; }
  }
  lower = Math.max(0, lower - 2);
  upper = Math.min(probabilityMass.length - 1, upper + 2);
  while (upper - lower < minimumIntervals && (lower > 0 || upper < probabilityMass.length - 1)) {
    if (lower > 0) lower -= 1;
    if (upper < probabilityMass.length - 1 && upper - lower < minimumIntervals) upper += 1;
  }
  return [lower, upper];
}

function includeHpdSupport(bounds, mask, nx, ny, coordinateIndex) {
  let minimum = coordinateIndex === 0 ? nx : ny;
  let maximum = -1;
  for (let ix = 0; ix < nx; ix += 1) {
    for (let iy = 0; iy < ny; iy += 1) {
      if (!mask[ix * ny + iy]) continue;
      const index = coordinateIndex === 0 ? ix : iy;
      minimum = Math.min(minimum, index);
      maximum = Math.max(maximum, index);
    }
  }
  if (maximum < 0) return bounds;
  const axisLength = coordinateIndex === 0 ? nx : ny;
  return [
    Math.min(bounds[0], Math.max(0, minimum - 2)),
    Math.max(bounds[1], Math.min(axisLength - 1, maximum + 2)),
  ];
}

function drawThresholdContour(ctx, level, xs, ys, values, ny, xPixel, yPixel) {
  const segment = (pointA, pointB) => {
    if (!pointA || !pointB) return;
    ctx.moveTo(pointA.x, pointA.y);
    ctx.lineTo(pointB.x, pointB.y);
  };
  ctx.beginPath();
  for (let ix = 0; ix < xs.length - 1; ix += 1) {
    for (let iy = 0; iy < ys.length - 1; iy += 1) {
      const p00 = { x: xPixel(xs[ix]), y: yPixel(ys[iy]) };
      const p10 = { x: xPixel(xs[ix + 1]), y: yPixel(ys[iy]) };
      const p11 = { x: xPixel(xs[ix + 1]), y: yPixel(ys[iy + 1]) };
      const p01 = { x: xPixel(xs[ix]), y: yPixel(ys[iy + 1]) };
      const v00 = values[ix * ny + iy];
      const v10 = values[(ix + 1) * ny + iy];
      const v11 = values[(ix + 1) * ny + iy + 1];
      const v01 = values[ix * ny + iy + 1];
      const intersections = [
        contourIntersection(level, v00, v10, p00, p10),
        contourIntersection(level, v10, v11, p10, p11),
        contourIntersection(level, v11, v01, p11, p01),
        contourIntersection(level, v01, v00, p01, p00),
      ];
      const present = intersections.filter(Boolean);
      if (present.length === 2) {
        segment(present[0], present[1]);
      } else if (present.length === 4) {
        const caseIndex =
          (v00 >= level ? 1 : 0)
          | (v10 >= level ? 2 : 0)
          | (v11 >= level ? 4 : 0)
          | (v01 >= level ? 8 : 0);
        const centerInside = 0.25 * (v00 + v10 + v11 + v01) >= level;
        if ((caseIndex === 5 && centerInside) || (caseIndex === 10 && !centerInside)) {
          segment(intersections[0], intersections[1]);
          segment(intersections[2], intersections[3]);
        } else {
          segment(intersections[0], intersections[3]);
          segment(intersections[1], intersections[2]);
        }
      }
    }
  }
  ctx.stroke();
}

function drawThresholdDomainBoundary(ctx, level, xs, ys, values, ny, xPixel, yPixel) {
  const drawInsidePart = (valueA, valueB, pointA, pointB) => {
    const insideA = valueA >= level;
    const insideB = valueB >= level;
    if (!insideA && !insideB) return;
    let start = pointA;
    let end = pointB;
    if (insideA !== insideB) {
      const crossing = contourIntersection(level, valueA, valueB, pointA, pointB);
      if (insideA) end = crossing;
      else start = crossing;
    }
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
  };
  ctx.beginPath();
  for (let ix = 0; ix < xs.length - 1; ix += 1) {
    drawInsidePart(
      values[ix * ny],
      values[(ix + 1) * ny],
      { x: xPixel(xs[ix]), y: yPixel(ys[0]) },
      { x: xPixel(xs[ix + 1]), y: yPixel(ys[0]) },
    );
    drawInsidePart(
      values[ix * ny + ny - 1],
      values[(ix + 1) * ny + ny - 1],
      { x: xPixel(xs[ix]), y: yPixel(ys[ny - 1]) },
      { x: xPixel(xs[ix + 1]), y: yPixel(ys[ny - 1]) },
    );
  }
  for (let iy = 0; iy < ys.length - 1; iy += 1) {
    drawInsidePart(
      values[iy],
      values[iy + 1],
      { x: xPixel(xs[0]), y: yPixel(ys[iy]) },
      { x: xPixel(xs[0]), y: yPixel(ys[iy + 1]) },
    );
    drawInsidePart(
      values[(xs.length - 1) * ny + iy],
      values[(xs.length - 1) * ny + iy + 1],
      { x: xPixel(xs[xs.length - 1]), y: yPixel(ys[iy]) },
      { x: xPixel(xs[xs.length - 1]), y: yPixel(ys[iy + 1]) },
    );
  }
  ctx.stroke();
}

function interpolationPosition(axis, value) {
  if (value <= axis[0]) return [0, 0];
  if (value >= axis[axis.length - 1]) return [axis.length - 2, 1];
  let lower = 0;
  let upper = axis.length - 1;
  while (upper - lower > 1) {
    const middle = Math.floor((lower + upper) / 2);
    if (axis[middle] <= value) lower = middle;
    else upper = middle;
  }
  return [lower, (value - axis[lower]) / (axis[upper] - axis[lower])];
}

function drawInterpolatedDensity(
  ctx, margin, plotW, plotH, xs, ys, density, ny, xTransform, xmin, xmax, ymin, ymax, maximum,
) {
  const rasterWidth = Math.max(2, Math.min(640, Math.round(plotW)));
  const rasterHeight = Math.max(2, Math.min(360, Math.round(plotH)));
  const xPositions = xs.map(xTransform);
  const xInterpolation = Array.from({ length: rasterWidth }, (_, pixel) => (
    interpolationPosition(xPositions, xmin + (pixel + 0.5) / rasterWidth * (xmax - xmin))
  ));
  const yInterpolation = Array.from({ length: rasterHeight }, (_, pixel) => (
    interpolationPosition(ys, ymax - (pixel + 0.5) / rasterHeight * (ymax - ymin))
  ));
  const raster = document.createElement("canvas");
  raster.width = rasterWidth;
  raster.height = rasterHeight;
  const rasterContext = raster.getContext("2d");
  const image = rasterContext.createImageData(rasterWidth, rasterHeight);
  for (let py = 0; py < rasterHeight; py += 1) {
    const [iy, ty] = yInterpolation[py];
    for (let px = 0; px < rasterWidth; px += 1) {
      const [ix, tx] = xInterpolation[px];
      const v00 = density[ix * ny + iy];
      const v10 = density[(ix + 1) * ny + iy];
      const v11 = density[(ix + 1) * ny + iy + 1];
      const v01 = density[ix * ny + iy + 1];
      const lower = v00 + tx * (v10 - v00);
      const upper = v01 + tx * (v11 - v01);
      const interpolated = lower + ty * (upper - lower);
      const intensity = maximum > 0 ? Math.pow(Math.max(0, interpolated) / maximum, 0.32) : 0;
      const rgb = heatRgb(intensity);
      const offset = 4 * (py * rasterWidth + px);
      image.data[offset] = rgb[0];
      image.data[offset + 1] = rgb[1];
      image.data[offset + 2] = rgb[2];
      image.data[offset + 3] = 255;
    }
  }
  rasterContext.putImageData(image, 0, 0);
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(raster, margin.left, margin.top, plotW, plotH);
  ctx.restore();
}

function drawProbabilityField(canvasId, xCoordinate, yCoordinate, xAxis, yAxis, density, probabilityMass, mask, hpdThreshold) {
  const { ctx, width, height } = canvasContext(canvasId);
  const margin = { left: 74, right: 24, top: 62, bottom: 58 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const xs = axisDisplayValues(xCoordinate, xAxis);
  const ys = axisDisplayValues(yCoordinate, yAxis);
  const nx = xs.length;
  const ny = ys.length;
  const maximum = density.reduce((highest, value) => Math.max(highest, value), 0);
  const xMarginalMass = Array.from({ length: nx }, (_, ix) => {
    let total = 0;
    for (let iy = 0; iy < ny; iy += 1) total += probabilityMass[ix * ny + iy];
    return total;
  });
  const yMarginalMass = Array.from({ length: ny }, (_, iy) => {
    let total = 0;
    for (let ix = 0; ix < nx; ix += 1) total += probabilityMass[ix * ny + iy];
    return total;
  });
  const [xStart, xEnd] = includeHpdSupport(
    posteriorIndexBounds(xMarginalMass), mask, nx, ny, 0,
  );
  const [yStart, yEnd] = includeHpdSupport(
    posteriorIndexBounds(yMarginalMass), mask, nx, ny, 1,
  );
  const visibleXs = xs.slice(xStart, xEnd + 1);
  const visibleYs = ys.slice(yStart, yEnd + 1);
  const xLogarithmic = xCoordinate === "pCO2";
  const xTransform = (value) => xLogarithmic ? Math.log10(value) : value;
  const xmin = xTransform(visibleXs[0]);
  const xmax = xTransform(visibleXs[visibleXs.length - 1]);
  const ymin = visibleYs[0];
  const ymax = visibleYs[visibleYs.length - 1];
  const xPixel = (value) => margin.left + (xTransform(value) - xmin) / (xmax - xmin) * plotW;
  const yPixel = (value) => margin.top + plotH - (value - ymin) / (ymax - ymin) * plotH;

  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, width, height);
  drawColorLegend(
    ctx,
    margin.left,
    21,
    Math.min(220, plotW * 0.38),
    posteriorColorStops,
    "Relative probability density",
    "Low",
    "High",
  );
  const regionX = margin.left + Math.min(220, plotW * 0.38) + 42;
  ctx.strokeStyle = "#17212b";
  ctx.lineWidth = 2;
  ctx.strokeRect(regionX, 22, 22, 10);
  ctx.fillStyle = "#43515a";
  ctx.font = "10px system-ui";
  ctx.textAlign = "left";
  ctx.fillText("95% credible region", regionX + 30, 31);
  ctx.save();
  ctx.beginPath();
  ctx.rect(margin.left, margin.top, plotW, plotH);
  ctx.clip();
  drawInterpolatedDensity(
    ctx, margin, plotW, plotH, xs, ys, density, ny,
    xTransform, xmin, xmax, ymin, ymax, maximum,
  );
  if (mask.some(Boolean)) {
    ctx.strokeStyle = "rgba(15, 20, 24, 0.92)";
    ctx.lineWidth = 1.7;
    const level = Number.isFinite(hpdThreshold) ? hpdThreshold
      : density.reduce((lowest, value, i) => mask[i] ? Math.min(lowest, value) : lowest, Infinity);
    drawThresholdContour(ctx, level, xs, ys, density, ny, xPixel, yPixel);
    drawThresholdDomainBoundary(ctx, level, xs, ys, density, ny, xPixel, yPixel);
  }
  ctx.restore();

  ctx.strokeStyle = "#26343d";
  ctx.lineWidth = 1;
  ctx.strokeRect(margin.left, margin.top, plotW, plotH);
  ctx.fillStyle = "#43515a";
  ctx.font = "11px system-ui";
  ctx.textAlign = "center";
  const xTicks = axisTickValues(xCoordinate, visibleXs, plotW);
  xTicks.forEach((tick) => {
    const x = xPixel(tick);
    ctx.strokeStyle = "rgba(255,255,255,0.45)";
    ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, margin.top + plotH); ctx.stroke();
    ctx.fillStyle = "#43515a";
    ctx.fillText(axisTickText(xCoordinate, tick), x, margin.top + plotH + 19);
  });
  ctx.fillText(axisLabel(xCoordinate, xLogarithmic), margin.left + plotW / 2, height - 14);
  ctx.textAlign = "right";
  for (const tick of axisTickValues(yCoordinate, visibleYs, plotH)) {
    const y = yPixel(tick);
    ctx.strokeStyle = "rgba(255,255,255,0.45)";
    ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(margin.left + plotW, y); ctx.stroke();
    ctx.fillStyle = "#43515a";
    ctx.fillText(axisTickText(yCoordinate, tick), margin.left - 9, y + 4);
  }
  ctx.save();
  ctx.translate(18, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText(axisLabel(yCoordinate), 0, 0);
  ctx.restore();
  enablePlotExport(canvasId);
}

function contourIntersection(level, valueA, valueB, pointA, pointB) {
  const crosses = (valueA < level && valueB >= level) || (valueB < level && valueA >= level);
  if (!crosses || valueA === valueB) return null;
  const fraction = (level - valueA) / (valueB - valueA);
  return {
    x: pointA.x + fraction * (pointB.x - pointA.x),
    y: pointA.y + fraction * (pointB.y - pointA.y),
  };
}

function contourLabel(level, decimals) {
  const fixed = Math.abs(level).toFixed(decimals);
  const absolute = fixed.includes(".") ? fixed.replace(/0+$/, "").replace(/\.$/, "") : fixed;
  return `${level < 0 ? "−" : ""}${absolute}`;
}

function drawContour(ctx, level, xs, ys, values, ny, xPixel, yPixel, labelTargetY, labelXBounds, showLabel, labelDecimals) {
  const labelCandidates = [];
  ctx.strokeStyle = "rgba(22, 31, 38, 0.78)";
  ctx.lineWidth = 1.15;
  for (let ix = 0; ix < xs.length - 1; ix += 1) {
    for (let iy = 0; iy < ys.length - 1; iy += 1) {
      const p00 = { x: xPixel(xs[ix]), y: yPixel(ys[iy]) };
      const p10 = { x: xPixel(xs[ix + 1]), y: yPixel(ys[iy]) };
      const p11 = { x: xPixel(xs[ix + 1]), y: yPixel(ys[iy + 1]) };
      const p01 = { x: xPixel(xs[ix]), y: yPixel(ys[iy + 1]) };
      const v00 = values[ix * ny + iy];
      const v10 = values[(ix + 1) * ny + iy];
      const v11 = values[(ix + 1) * ny + iy + 1];
      const v01 = values[ix * ny + iy + 1];
      const points = [
        contourIntersection(level, v00, v10, p00, p10),
        contourIntersection(level, v10, v11, p10, p11),
        contourIntersection(level, v11, v01, p11, p01),
        contourIntersection(level, v01, v00, p01, p00),
      ].filter(Boolean);
      for (let index = 0; index + 1 < points.length; index += 2) {
        ctx.beginPath();
        ctx.moveTo(points[index].x, points[index].y);
        ctx.lineTo(points[index + 1].x, points[index + 1].y);
        ctx.stroke();
        const midpoint = {
          x: 0.5 * (points[index].x + points[index + 1].x),
          y: 0.5 * (points[index].y + points[index + 1].y),
        };
        labelCandidates.push(midpoint);
      }
    }
  }
  if (!showLabel) return;
  const interior = labelCandidates.filter((point) => point.x >= labelXBounds[0] && point.x <= labelXBounds[1]);
  const candidates = interior.length ? interior : labelCandidates;
  const labelPoint = candidates.reduce(
    (best, point) => !best || Math.abs(point.y - labelTargetY) < Math.abs(best.y - labelTargetY) ? point : best,
    null,
  );
  if (!labelPoint) return;
  const label = contourLabel(level, labelDecimals);
  ctx.font = "600 10px system-ui";
  const width = ctx.measureText(label).width + 7;
  const x = labelPoint.x - width / 2;
  const y = labelPoint.y;
  ctx.fillStyle = "rgba(255,255,255,0.84)";
  ctx.fillRect(x, y - 7, width, 14);
  ctx.fillStyle = "#17212b";
  ctx.textAlign = "center";
  ctx.fillText(label, x + width / 2, y + 4);
}

function drawIsotopeField(result) {
  const { ctx, width, height } = canvasContext("isotope-canvas");
  const margin = { left: 74, right: 24, top: 62, bottom: 58 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const xs = result.axes.pCO2;
  const ys = result.axes.GPP.map((value) => 100 * value / MODERN_GPP);
  const values = result.central_cap_delta17_permil;
  const nx = xs.length;
  const ny = ys.length;
  const minimum = result.minimum_cap_delta17_permil;
  const maximum = result.maximum_cap_delta17_permil;
  const xmin = Math.log10(xs[0]);
  const xmax = Math.log10(xs[nx - 1]);
  const ymin = ys[0];
  const ymax = ys[ny - 1];
  const xPixel = (value) => margin.left + (Math.log10(value) - xmin) / (xmax - xmin) * plotW;
  const yPixel = (value) => margin.top + plotH - (value - ymin) / (ymax - ymin) * plotH;

  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, width, height);
  const legendWidth = Math.min(260, plotW * 0.46);
  drawColorLegend(
    ctx,
    margin.left,
    21,
    legendWidth,
    isotopeColorStops,
    "Atmospheric O₂ Δ′¹⁷O₀.₅₂₈ (‰)",
    format(result.minimum_cap_delta17_permil, result.contour_label_decimals),
    format(result.maximum_cap_delta17_permil, result.contour_label_decimals),
    12,
  );
  ctx.fillStyle = "#43515a";
  ctx.font = "600 10px system-ui";
  ctx.textAlign = "left";
  ctx.fillText(
    `Fixed pO₂ = ${format(result.fixed_p_o2_pal, 2)} PAL`,
    margin.left + legendWidth + 34,
    28,
  );
  for (let ix = 0; ix < nx - 1; ix += 1) {
    for (let iy = 0; iy < ny - 1; iy += 1) {
      const index = ix * ny + iy;
      const normalized = (values[index] - minimum) / (maximum - minimum || 1);
      const x0 = xPixel(xs[ix]);
      const x1 = xPixel(xs[ix + 1]);
      const y0 = yPixel(ys[iy]);
      const y1 = yPixel(ys[iy + 1]);
      ctx.fillStyle = heatColor(normalized, isotopeColorStops);
      ctx.fillRect(x0, y1, Math.max(1, x1 - x0 + 0.4), Math.max(1, y0 - y1 + 0.4));
    }
  }

  const levels = result.contour_levels_permil;
  const labelDecimals = result.contour_label_decimals;
  const canvas = $("isotope-canvas");
  canvas.dataset.contourCount = String(levels.length);
  canvas.dataset.contourValues = levels.join(",");
  canvas.setAttribute(
    "aria-label",
    `Atmospheric oxygen Δ′17O0.528 isotope field at ${format(result.fixed_p_o2_pal, 2)} PAL with ${levels.length} labeled contours balanced across the plotted field`,
  );
  levels.forEach((level, index) => {
    const fraction = levels.length === 1 ? 0.5 : 0.18 + 0.64 * index / (levels.length - 1);
    drawContour(
      ctx,
      level,
      xs,
      ys,
      values,
      ny,
      xPixel,
      yPixel,
      margin.top + fraction * plotH,
      [margin.left + 0.12 * plotW, margin.left + 0.88 * plotW],
      true,
      labelDecimals,
    );
  });

  ctx.strokeStyle = "#26343d";
  ctx.lineWidth = 1;
  ctx.strokeRect(margin.left, margin.top, plotW, plotH);
  ctx.fillStyle = "#43515a";
  ctx.font = "11px system-ui";
  ctx.textAlign = "center";
  const xTicks = [50, 100, 300, 1000, 3000, 10000, 30000, 60000].filter((x) => x >= xs[0] && x <= xs[nx - 1]);
  xTicks.forEach((tick) => {
    const x = xPixel(tick);
    ctx.fillText(tick >= 1000 ? `${tick / 1000}k` : String(tick), x, margin.top + plotH + 19);
  });
  ctx.fillText("pCO₂ (ppm, logarithmic axis)", margin.left + plotW / 2, height - 14);
  ctx.textAlign = "right";
  const yStep = ymax - ymin > 120 ? 50 : 25;
  for (let tick = Math.ceil(ymin / yStep) * yStep; tick <= ymax; tick += yStep) {
    ctx.fillText(String(tick), margin.left - 9, yPixel(tick) + 4);
  }
  ctx.save();
  ctx.translate(18, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("GPP (% modern)", 0, 0);
  ctx.restore();
  enablePlotExport("isotope-canvas");
}

async function runIsotopeField() {
  const button = $("run-surface");
  $("surface-error").textContent = "";
  setBusy(button, true, "Calculating…");
  try {
    const xmin = finite(number("surface-xmin"), "Minimum pCO2");
    const xmax = finite(number("surface-xmax"), "Maximum pCO2");
    const ymin = finite(number("surface-ymin"), "Minimum GPP");
    const ymax = finite(number("surface-ymax"), "Maximum GPP");
    if (xmin >= xmax || ymin >= ymax) throw new Error("Each surface minimum must be lower than its maximum.");
    validateSurfaceDomain(xmin, xmax, ymin, ymax);
    const payload = await api("/api/v1/field/isotope", {
      method: "POST",
      onComputeState: (status) => setBusy(button, true, status === "waiting" ? "Waiting…" : "Calculating…"),
      body: JSON.stringify({
        p_o2_pal: finite(number("surface-po2"), "Isotope-field pO2"),
        pco2_bounds_ppm: [xmin, xmax],
        gpp_bounds_pgC_per_year: [ymin * MODERN_GPP / 100, ymax * MODERN_GPP / 100],
        pco2_grid_size: 241,
        gpp_grid_size: 201,
      }),
    });
    drawIsotopeField(payload.result);
  } catch (error) {
    $("surface-error").textContent = publicErrorMessage(error);
  } finally {
    setBusy(button, false);
  }
}

function runSurface() {
  return runIsotopeField();
}

function updateSurfaceMode() {
  $("run-surface").dataset.label = $("run-surface").textContent;
}

function niceTickStep(rawStep) {
  if (!Number.isFinite(rawStep) || rawStep <= 0) return 1;
  const exponent = Math.floor(Math.log10(rawStep));
  const magnitude = 10 ** exponent;
  const fraction = rawStep / magnitude;
  const niceFraction = fraction <= 1.5 ? 1 : fraction <= 3 ? 2 : fraction <= 7 ? 5 : 10;
  return niceFraction * magnitude;
}

function niceAxisScale(minimum, maximum, targetIntervals = 5) {
  let lower = minimum;
  let upper = maximum;
  if (lower === upper) {
    const halfSpan = Math.max(Math.abs(lower) * 0.05, 0.0005);
    lower -= halfSpan;
    upper += halfSpan;
  }
  const step = niceTickStep((upper - lower) / targetIntervals);
  lower = Math.floor(lower / step) * step;
  upper = Math.ceil(upper / step) * step;
  const ticks = [];
  for (let value = lower; value <= upper + step * 1e-8; value += step) {
    ticks.push(Math.abs(value) < step * 1e-10 ? 0 : value);
  }
  return { lower, upper, step, ticks };
}

function timeTickLabel(value, step) {
  const digits = step >= 1 ? 0 : Math.max(0, Math.ceil(-Math.log10(step)));
  return formatFixed(value, digits);
}

function drawLine(id, times, values, title, unit, color, axisDigits = 3, fixedAxis = false) {
  const { ctx, width, height } = canvasContext(id);
  const margin = { left: 70, right: 22, top: 40, bottom: 50 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const xmin = times[0];
  const xmax = times[times.length - 1] || 1;
  const rawYmin = Math.min(...values);
  const rawYmax = Math.max(...values);
  const padding = Math.max((rawYmax - rawYmin) * 0.12, 0.0005);
  const yScale = niceAxisScale(rawYmin - padding, rawYmax + padding);
  const ymin = yScale.lower;
  const ymax = yScale.upper;
  const xTickStep = niceTickStep((xmax - xmin) / 7);
  const xTicks = [];
  const firstXTick = Math.ceil(xmin / xTickStep) * xTickStep;
  for (let value = firstXTick; value <= xmax + xTickStep * 1e-8; value += xTickStep) {
    xTicks.push(Math.abs(value) < xTickStep * 1e-10 ? 0 : value);
  }
  const xp = (x) => margin.left + (x - xmin) / (xmax - xmin || 1) * plotW;
  const yp = (y) => margin.top + plotH - (y - ymin) / (ymax - ymin || 1) * plotH;
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#17212b"; ctx.font = "600 13px system-ui"; ctx.textAlign = "left"; ctx.fillText(title, margin.left, 20);
  ctx.strokeStyle = "#d6dde1"; ctx.lineWidth = 1;
  yScale.ticks.forEach((value) => {
    const y = yp(value);
    ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(margin.left + plotW, y); ctx.stroke();
    const tickLabel = fixedAxis ? formatFixed(value, axisDigits) : format(value, axisDigits);
    ctx.fillStyle = "#61707c"; ctx.font = "10px system-ui"; ctx.textAlign = "right"; ctx.fillText(tickLabel, margin.left - 8, y + 3);
  });
  ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.beginPath();
  values.forEach((value, index) => { const x = xp(times[index]); const y = yp(value); index ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
  ctx.strokeStyle = "#26343d"; ctx.lineWidth = 1; ctx.strokeRect(margin.left, margin.top, plotW, plotH);
  ctx.fillStyle = "#43515a"; ctx.font = "10px system-ui"; ctx.textAlign = "center";
  xTicks.forEach((value) => ctx.fillText(timeTickLabel(value, xTickStep), xp(value), margin.top + plotH + 18));
  ctx.fillText("Time (years)", margin.left + plotW / 2, height - 10);
  ctx.save(); ctx.translate(16, margin.top + plotH / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(unit, 0, 0); ctx.restore();
}

function updateTransientControls(applyDefaults = true) {
  const type = $("transient-type").value;
  const caption = $("transient-final-caption");
  const input = $("transient-final");
  const durationInput = $("transient-duration");
  const trajectory = type === "pCO2_trajectory";
  $("transient-final-label").classList.toggle("hidden", trajectory);
  $("trajectory-controls").classList.toggle("hidden", !trajectory);
  const settings = {
    pCO2: ["Final pCO<sub>2</sub>", "ppm", "1000", "1", "12000"],
    pCO2_trajectory: ["", "", "", "", "12000"],
    pO2: ["Final pO<sub>2</sub>", "PAL", "0.50", "0.01", "12000"],
    GPP: ["Final GPP", "% modern", "50.0", "0.1", "18000"],
    photosynthesis: ["Final photosynthesis", "% initial", "50.0", "0.1", "22000"],
  }[type];
  if (!trajectory) {
    caption.innerHTML = `${settings[0]} <small>${settings[1]}</small>`;
    input.step = settings[3];
    if (applyDefaults) input.value = settings[2];
  }
  if (applyDefaults) durationInput.value = settings[4];
}

function applyTrajectoryPreset() {
  if ($("trajectory-preset").value !== "historical") return;
  $("trajectory-start").value = "285.5";
  $("trajectory-end").value = "422.8";
  $("trajectory-duration").value = "174";
  $("trajectory-interpolation").value = "smoothstep";
}

async function runTransient() {
  const button = $("run-transient");
  const exportButton = $("download-transient-xlsx");
  const type = $("transient-type").value;
  $("transient-error").textContent = "";
  exportButton.disabled = true;
  state.lastTransient = null;
  setBusy(button, true, "Solving…");
  const endProgress = beginTransientProgress(type);
  try {
    const initial = currentForwardState();
    const duration = finite(number("transient-duration"), "Duration");
    const finalValue = type === "pCO2_trajectory"
      ? finite(number("trajectory-end"), "Final pCO2")
      : finite(number("transient-final"), "Final value");
    let path;
    let request;
    if (type === "pCO2_trajectory") {
      initial.p_co2_ppm = finite(number("trajectory-start"), "Initial pCO2");
      const transitionDuration = finite(number("trajectory-duration"), "Transition duration");
      if (transitionDuration <= 0 || transitionDuration > 100000) {
        throw new Error("Transition duration must be between 0 and 100,000 years.");
      }
      path = "/api/v1/transients/pco2-trajectory";
      request = {
        initial,
        final_pco2_ppm: finalValue,
        transition_duration_years: transitionDuration,
        interpolation: $("trajectory-interpolation").value,
        duration_years: duration,
        sample_count: 241,
        equilibrium_search_max_years: Math.max(100000, duration, transitionDuration),
      };
    } else if (type === "photosynthesis") {
      path = "/api/v1/transients/photosynthesis-step";
      request = {
        initial,
        photosynthesis_fraction: finalValue / 100,
        duration_years: duration,
        sample_count: 181,
        equilibrium_search_max_years: Math.max(100000, duration),
      };
    } else {
      const final = { ...initial };
      if (type === "pCO2") final.p_co2_ppm = finalValue;
      if (type === "pO2") final.p_o2_pal = finalValue;
      if (type === "GPP") final.gpp_pgC_per_year = finalValue * MODERN_GPP / 100;
      path = "/api/v1/transients/state-step";
      request = {
        initial,
        final,
        duration_years: duration,
        sample_count: 181,
        equilibrium_search_max_years: Math.max(100000, duration),
      };
    }
    const payload = await api(path, {
      method: "POST", body: JSON.stringify(request), onComputeState: endProgress.onState,
    });
    const result = payload.result;
    const times = result.time_years;
    const d17 = result.states.map((item) => item.cap_delta17_prime_permil);
    const d18 = result.states.map((item) => item.delta18_prime_permil);
    const preStepDuration = Math.min(duration, niceTickStep(duration / 6));
    const displayTimes = [-preStepDuration, ...times];
    const displayD17 = [d17[0], ...d17];
    const displayD18 = [d18[0], ...d18];
    let forcingTitle;
    let forcingUnit;
    let forcingInitial;
    let forcingFinal;
    let forcingDigits;
    if (type === "pCO2_trajectory") {
      forcingTitle = "Prescribed pCO₂ trajectory";
      forcingUnit = "pCO₂ (ppm)";
      forcingInitial = initial.p_co2_ppm;
      forcingFinal = finalValue;
      forcingDigits = 0;
    } else if (type === "pCO2") {
      forcingTitle = "Imposed pCO₂ step";
      forcingUnit = "pCO₂ (ppm)";
      forcingInitial = initial.p_co2_ppm;
      forcingFinal = finalValue;
      forcingDigits = 0;
    } else if (type === "pO2") {
      forcingTitle = "Imposed pO₂ step";
      forcingUnit = "pO₂ (PAL)";
      forcingInitial = initial.p_o2_pal;
      forcingFinal = finalValue;
      forcingDigits = 2;
    } else if (type === "GPP") {
      forcingTitle = "Imposed GPP step";
      forcingUnit = "GPP (% modern)";
      forcingInitial = 100 * initial.gpp_pgC_per_year / MODERN_GPP;
      forcingFinal = finalValue;
      forcingDigits = 1;
    } else {
      forcingTitle = "Imposed photosynthesis step";
      forcingUnit = "Photosynthesis (% initial)";
      forcingInitial = 100;
      forcingFinal = finalValue;
      forcingDigits = 1;
    }
    const trajectoryPlotEnd = type === "pCO2_trajectory"
      ? Math.min(duration, request.transition_duration_years * 1.5)
      : duration;
    const trajectoryIndices = type === "pCO2_trajectory"
      ? times.map((time, index) => ({ time, index })).filter((item) => item.time <= trajectoryPlotEnd)
      : [];
    const trajectoryPreDuration = type === "pCO2_trajectory"
      ? Math.min(request.transition_duration_years * 0.25, 50)
      : preStepDuration;
    const forcingTimes = type === "pCO2_trajectory"
      ? [-trajectoryPreDuration, ...trajectoryIndices.map((item) => item.time)]
      : [-preStepDuration, 0, 0, duration];
    const forcingValues = type === "pCO2_trajectory"
      ? [result.pco2_ppm[0], ...trajectoryIndices.map((item) => result.pco2_ppm[item.index])]
      : [forcingInitial, forcingInitial, forcingFinal, forcingFinal];
    drawLine(
      "transient-forcing",
      forcingTimes,
      forcingValues,
      forcingTitle,
      forcingUnit,
      "#c18b28",
      forcingDigits,
      false,
    );
    enablePlotExport("transient-forcing");
    const showPco2Response = type === "photosynthesis";
    $("transient-pco2-panel").classList.toggle("hidden", !showPco2Response);
    if (showPco2Response) {
      drawLine(
        "transient-pco2",
        displayTimes,
        [result.pco2_ppm[0], ...result.pco2_ppm],
        "Atmospheric pCO₂ response",
        "pCO₂ (ppm)",
        "#7a5ca5",
        0,
      );
      enablePlotExport("transient-pco2");
    }
    drawLine("transient-d17", displayTimes, displayD17, "Atmospheric O₂ Δ′¹⁷O", "Δ′¹⁷O (‰)", "#006f71", 3, true);
    drawLine("transient-d18", displayTimes, displayD18, "Atmospheric O₂ δ′¹⁸O", "δ′¹⁸O (‰)", "#a83d45", 3, true);
    enablePlotExport("transient-d17");
    enablePlotExport("transient-d18");
    const equilibrium = result.operational_equilibrium?.time_years ?? result.equilibrium_time_years;
    let transitionD17Card = "";
    let transitionD18Card = "";
    if (type === "pCO2_trajectory") {
      const transitionState = result.transition_end_state;
      transitionD17Card = `<div><span>At transition end Δ′<sup>17</sup>O</span><strong>${formatFixed(transitionState.cap_delta17_prime_permil, 3)}‰</strong></div>`;
      transitionD18Card = `<div><span>At transition end δ′<sup>18</sup>O</span><strong>${formatFixed(transitionState.delta18_prime_permil, 3)}‰</strong></div>`;
    }
    $("transient-summary").innerHTML = `
      <div><span>Initial Δ′<sup>17</sup>O</span><strong>${formatFixed(d17[0], 3)}‰</strong></div>
      ${transitionD17Card}
      <div><span>Displayed final Δ′<sup>17</sup>O</span><strong>${formatFixed(d17[d17.length - 1], 3)}‰</strong></div>
      <div><span>Initial δ′<sup>18</sup>O</span><strong>${formatFixed(d18[0], 3)}‰</strong></div>
      ${transitionD18Card}
      <div><span>Displayed final δ′<sup>18</sup>O</span><strong>${formatFixed(d18[d18.length - 1], 3)}‰</strong></div>
      <div><span>Operational equilibrium time</span><strong>${equilibrium == null ? "Beyond search horizon" : `${format(equilibrium, 0)} years`}</strong></div>`;
    state.lastTransient = { type, request };
    exportButton.disabled = false;
  } catch (error) {
    $("transient-error").textContent = publicErrorMessage(error);
  } finally {
    endProgress();
    setBusy(button, false);
  }
}

async function downloadTransientWorkbook() {
  if (!state.lastTransient) return;
  const button = $("download-transient-xlsx");
  const { type, request } = state.lastTransient;
  $("transient-error").textContent = "";
  setBusy(button, true, "Preparing…");
  try {
    const body = {
      experiment_type: type,
      ...(type === "photosynthesis"
        ? { photosynthesis_step: request }
        : type === "pCO2_trajectory"
          ? { pco2_trajectory: request }
          : { state_step: request }),
    };
    const response = await modelFetch("/api/v1/export/transient.xlsx", {
      method: "POST",
      onComputeState: (status) => setBusy(button, true, status === "waiting" ? "Waiting…" : "Preparing…"),
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(detailMessage(payload));
    }
    const blob = await response.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `oxytib_${type.toLowerCase()}_time_response.xlsx`;
    link.click();
    URL.revokeObjectURL(link.href);
  } catch (error) {
    $("transient-error").textContent = publicErrorMessage(error);
  } finally {
    setBusy(button, false);
  }
}

function resetInputs() {
  state.source = "air";
  state.solveFor = "pCO2";
  document.querySelectorAll("#source-selector button").forEach((button) => {
    button.classList.toggle("active", button.dataset.source === state.source);
  });
  document.querySelectorAll("#solve-selector button").forEach((button) => {
    button.classList.toggle("active", button.dataset.coordinate === state.solveFor);
  });
  $("air-inputs").classList.remove("hidden");
  $("spherule-inputs").classList.add("hidden");
  $("sulfate-inputs").classList.add("hidden");
  document.querySelectorAll("#sulfate-inputs input").forEach((input) => {
    input.value = input.defaultValue;
  });
  $("sulfate-f-mode").value = "fixed";
  $("sulfate-b-mode").value = "fixed";
  $("sulfate-fractionation").value = "none";
  $("sulfate-advanced").open = false;
  updateSulfateControls();
  $("air-d17").value = "-0.432";
  $("air-sigma").value = "0.015";
  $("air-d18").value = "23.900";
  $("air-d18-sigma").value = "0.300";
  $("spherule-d17").value = "-0.660";
  $("spherule-d18").value = "43.269";
  $("spherule-d17-sigma").value = "0.060";
  $("spherule-d18-sigma").value = "0.500";
  $("pco2").value = "294";
  $("gpp").value = "100.0";
  $("po2").value = "1.00";
  $("surface-po2").value = "1.00";
  $("pco2-sigma").value = "";
  $("pco2-lower").value = "";
  $("pco2-upper").value = "";
  $("gpp-sigma").value = "";
  $("gpp-lower").value = "";
  $("gpp-upper").value = "";
  $("po2-sigma").value = "";
  $("po2-lower").value = "";
  $("po2-upper").value = "";
  setConstraintMode("pCO2", "fixed");
  setConstraintMode("GPP", "fixed");
  setConstraintMode("pO2", "fixed");
  updateCoordinateControls();
  $("solver-error").textContent = "";
  $("solver-result").classList.add("hidden");
  $("solver-empty").classList.remove("hidden");
  $("download-result-xlsx").disabled = true;
  state.lastResult = null;
}

function resetSurfaceInputs() {
  $("surface-xmin").value = "50";
  $("surface-xmax").value = "60000";
  $("surface-ymin").value = "6.3";
  $("surface-ymax").value = "150";
  $("surface-po2").value = "1.00";
  $("surface-error").textContent = "";
  clearPlot("isotope-canvas");
}

function resetTransientInputs() {
  $("transient-type").value = "pCO2";
  $("trajectory-preset").value = "historical";
  updateTransientControls(true);
  applyTrajectoryPreset();
  $("transient-error").textContent = "";
  $("transient-summary").innerHTML = "";
  $("transient-pco2-panel").classList.add("hidden");
  ["transient-forcing", "transient-pco2", "transient-d17", "transient-d18"].forEach(clearPlot);
  $("download-transient-xlsx").disabled = true;
  state.lastTransient = null;
}

function bindInterface() {
  $("theme-toggle").addEventListener("change", (event) => {
    applyTheme(event.target.checked ? "dark" : "light");
  });
  document.querySelectorAll(".tab").forEach((button) => button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => item.classList.toggle("active", item === button));
    document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
    $(`view-${button.dataset.tab}`).classList.add("active");
  }));
  document.querySelectorAll("#source-selector button").forEach((button) => button.addEventListener("click", () => {
    state.source = button.dataset.source;
    document.querySelectorAll("#source-selector button").forEach((item) => item.classList.toggle("active", item === button));
    $("air-inputs").classList.toggle("hidden", state.source !== "air");
    $("spherule-inputs").classList.toggle("hidden", state.source !== "spherule");
    $("sulfate-inputs").classList.toggle("hidden", state.source !== "sulfate");
    if (state.source === "spherule") isotopeTarget().catch(() => {});
  }));
  document.querySelectorAll("#solve-selector button").forEach((button) => button.addEventListener("click", () => {
    state.solveFor = button.dataset.coordinate;
    $("solver-result").classList.add("hidden");
    $("solver-empty").classList.remove("hidden");
    $("download-result-xlsx").disabled = true;
    $("solver-error").textContent = "";
    state.lastResult = null;
    document.querySelectorAll("#solve-selector button").forEach((item) => item.classList.toggle("active", item === button));
    updateCoordinateControls();
  }));
  for (const [selector, coordinate] of [["pco2-constraint-mode", "pCO2"], ["gpp-constraint-mode", "GPP"], ["po2-constraint-mode", "pO2"]]) {
    document.querySelectorAll(`#${selector} button`).forEach((button) => button.addEventListener("click", () => {
      setConstraintMode(coordinate, button.dataset.mode);
    }));
  }
  $("run-solver").addEventListener("click", runSolver);
  ["sulfate-f-mode", "sulfate-b-mode", "sulfate-fractionation"].forEach((id) => {
    $(id).addEventListener("change", updateSulfateControls);
  });
  $("sulfate-fractionation").addEventListener("change", () => {
    $("sulfate-advanced").open = $("sulfate-fractionation").value === "specified";
  });
  $("run-surface").addEventListener("click", runSurface);
  $("run-transient").addEventListener("click", runTransient);
  $("download-transient-xlsx").addEventListener("click", downloadTransientWorkbook);
  $("transient-type").addEventListener("change", () => updateTransientControls(true));
  $("trajectory-preset").addEventListener("change", applyTrajectoryPreset);
  ["trajectory-start", "trajectory-end", "trajectory-duration", "trajectory-interpolation"].forEach((id) => {
    $(id).addEventListener("change", () => { $("trajectory-preset").value = "custom"; });
  });
  $("download-result-xlsx").addEventListener("click", downloadWorkbook);
  document.querySelectorAll(".plot-export").forEach((button) => {
    button.addEventListener("click", () => downloadPlot(button));
  });
  $("reset-inputs").addEventListener("click", resetInputs);
  $("reset-surface").addEventListener("click", resetSurfaceInputs);
  $("reset-transient").addEventListener("click", resetTransientInputs);
  window.addEventListener("pageshow", () => {
    updateTransientControls(false);
    updateSulfateControls();
  });
}

async function initialize() {
  applyTheme(preferredTheme(), false);
  bindInterface();
  updateSulfateControls();
  setConstraintMode("pCO2", "fixed");
  setConstraintMode("GPP", "fixed");
  setConstraintMode("pO2", "fixed");
  updateCoordinateControls();
  updateSurfaceMode();
  updateTransientControls(false);
  try {
    state.metadata = await api("/api/v1/model");
    $("model-state").classList.add("ready");
    $("model-state").innerHTML = "<span></span> Model ready";
    $("footer-model").textContent = `OXYTIB ${state.metadata.citation.version}`;
  } catch (error) {
    $("model-state").classList.add("error");
    $("model-state").innerHTML = "<span></span> Model unavailable";
  }
}

initialize();
