/**
 * app.js
 * High-Performance Frontend Engine for SDG Predictive Monitoring Dashboard
 * BCSE497J: Project I - Vellore Institute of Technology (VIT) Chennai Campus
 */

// Application State
const state = {
  data: null,
  selectedState: "Tamil Nadu",
  selectedYear: 2026,
  selectedSdgFilter: "ALL",
  compareState: "Kerala",
  compareSdg: "SDG3_Health",
  geoMetric: "Composite_Score",
  healthIndicator: "",
  charts: {}
};

const COLORS = {
  sdg3: "#4ade80",       /* Green  — SDG 3 Health   */
  sdg4: "#facc15",       /* Yellow — SDG 4 Education */
  sdg13: "#f87171",      /* Red    — SDG 13 Climate  */
  composite: "#a78bfa",
  highRisk: "#f87171",
  medRisk: "#facc15",
  lowRisk: "#4ade80",
  accent: "#6b8fad",
  grid: "rgba(232, 237, 242, 0.06)",
  tick: "#9aa5b1",
  legend: "#c5ced6",
  tooltipBg: "#161b22",
  tooltipTitle: "#e8edf2",
  tooltipBody: "#9aa5b1"
};

const CHART_FONT = "Inter, Roboto, system-ui, sans-serif";

function refreshIcons() {
  if (window.lucide) lucide.createIcons();
}

function applyChartDefaults() {
  if (!window.Chart) return;
  Chart.defaults.font.family = CHART_FONT;
  Chart.defaults.color = COLORS.tick;
  Chart.defaults.borderColor = COLORS.grid;
}

function setupAboutModal() {
  const modal = document.getElementById("aboutModal");
  const openers = [document.getElementById("aboutOpenBtn"), document.getElementById("aboutOpenBtnFooter")];
  const closer = document.getElementById("aboutCloseBtn");
  const open = () => modal.classList.add("open");
  const close = () => modal.classList.remove("open");
  openers.forEach(btn => btn && btn.addEventListener("click", open));
  if (closer) closer.addEventListener("click", close);
  modal.addEventListener("click", (e) => {
    if (e.target === modal) close();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") close();
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  applyChartDefaults();
  setupAboutModal();
  setupTabs();
  setupGlobalControls();
  refreshIcons();
  await loadDashboardData();
  refreshIcons();
});

// Setup Navigation Tabs
function setupTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(tc => tc.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add("active");
        handleTabSwitch(targetId);
      }
    });
  });
}

function handleTabSwitch(tabId) {
  if (tabId === "tab-forecaster") {
    renderTrendChart();
    renderCompareChart();
  } else if (tabId === "tab-climate") {
    renderClimateCharts();
  } else if (tabId === "tab-health") {
    renderHealthChart();
  }
}

// Setup Global Controls (State dropdown, Year buttons, SDG selector)
function setupGlobalControls() {
  // Year toggle buttons
  const yearBtns = document.querySelectorAll(".year-btn");
  yearBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      yearBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.selectedYear = parseInt(btn.getAttribute("data-year"));
      
      // Update UI year labels
      document.querySelectorAll(".selected-year-text").forEach(el => {
        el.textContent = state.selectedYear;
      });

      updateNationalKPIs();
      renderGoalScorecards();
      renderRiskMatrixTable();
      renderGeoAnalytics();
    });
  });

  // State selector
  const stateSelect = document.getElementById("stateSelect");
  stateSelect.addEventListener("change", (e) => {
    state.selectedState = e.target.value;
    document.getElementById("forecaster-state-title").textContent = state.selectedState;
    document.getElementById("policy-state-name").textContent = state.selectedState;
    
    renderTrendChart();
    renderGoalScorecards();
    loadPolicyAdvisory();
  });

  // SDG Focus Filter
  const sdgFilterSelect = document.getElementById("sdgFilterSelect");
  sdgFilterSelect.addEventListener("change", (e) => {
    state.selectedSdgFilter = e.target.value;
    renderTrendChart();
  });

  // Compare State / SDG
  const compareStateSelect = document.getElementById("compareStateSelect");
  compareStateSelect.addEventListener("change", (e) => {
    state.compareState = e.target.value;
    renderCompareChart();
  });

  const compareSdgSelect = document.getElementById("compareSdgSelect");
  compareSdgSelect.addEventListener("change", (e) => {
    state.compareSdg = e.target.value;
    renderCompareChart();
  });

  // Table Filters
  const tableSearchInput = document.getElementById("tableSearchInput");
  tableSearchInput.addEventListener("input", () => renderRiskMatrixTable());

  const tableRiskFilter = document.getElementById("tableRiskFilter");
  tableRiskFilter.addEventListener("change", () => renderRiskMatrixTable());

  const tableSdgFilter = document.getElementById("tableSdgFilter");
  tableSdgFilter.addEventListener("change", () => renderRiskMatrixTable());

  // Geo Metric Select
  const geoMetricSelect = document.getElementById("geoMetricSelect");
  geoMetricSelect.addEventListener("change", (e) => {
    state.geoMetric = e.target.value;
    renderGeoAnalytics();
  });

  // Health Sub-indicator select
  const healthIndicatorSelect = document.getElementById("healthIndicatorSelect");
  healthIndicatorSelect.addEventListener("change", (e) => {
    state.healthIndicator = e.target.value;
    renderHealthChart();
  });
}

// Fetch Initial Data from Server API
async function loadDashboardData() {
  try {
    const res = await fetch("/api/initial-data");
    if (!res.ok) throw new Error("Failed to load initial data");
    state.data = await res.json();

    populateDropdowns();
    updateNationalKPIs();
    renderTrendChart();
    renderGoalScorecards();
    renderCompareChart();
    renderPriorityAlerts();
    renderRiskMatrixTable();
    renderGeoAnalytics();
    renderClimateCharts();
    renderHealthIndicators();
    renderValidationSection();
    loadPolicyAdvisory();

  } catch (err) {
    console.error("Error loading dashboard data:", err);
  }
}

// Populate State & Indicator Dropdowns
function populateDropdowns() {
  const stateSelect = document.getElementById("stateSelect");
  const compareStateSelect = document.getElementById("compareStateSelect");
  
  stateSelect.innerHTML = "";
  compareStateSelect.innerHTML = "";

  state.data.states.forEach(st => {
    const opt = document.createElement("option");
    opt.value = st;
    opt.textContent = st;
    if (st === state.selectedState) opt.selected = true;
    stateSelect.appendChild(opt);

    const compOpt = document.createElement("option");
    compOpt.value = st;
    compOpt.textContent = st;
    if (st === state.compareState) compOpt.selected = true;
    compareStateSelect.appendChild(compOpt);
  });

  // Health Indicator columns
  const healthIndicatorSelect = document.getElementById("healthIndicatorSelect");
  healthIndicatorSelect.innerHTML = "";
  if (state.data.health_indicators.length > 0) {
    const sample = state.data.health_indicators[0];
    const numericKeys = Object.keys(sample).filter(k => k !== "SNo" && k !== "Area");
    numericKeys.forEach((key, idx) => {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = key;
      if (idx === 0) {
        opt.selected = true;
        state.healthIndicator = key;
      }
      healthIndicatorSelect.appendChild(opt);
    });
  }
}

// Update Top Executive KPI Cards
function updateNationalKPIs() {
  const kpis = state.data.kpis[state.selectedYear];
  if (!kpis) return;

  document.getElementById("kpi-sdg3-val").textContent = kpis.sdg3_avg.toFixed(1);
  const s3DeltaEl = document.getElementById("kpi-sdg3-delta");
  s3DeltaEl.textContent = `${kpis.sdg3_delta >= 0 ? '+' : ''}${kpis.sdg3_delta.toFixed(1)} pts vs 2023`;
  s3DeltaEl.className = `kpi-delta ${kpis.sdg3_delta >= 0 ? 'pos' : 'neg'}`;

  document.getElementById("kpi-sdg4-val").textContent = kpis.sdg4_avg.toFixed(1);
  const s4DeltaEl = document.getElementById("kpi-sdg4-delta");
  s4DeltaEl.textContent = `${kpis.sdg4_delta >= 0 ? '+' : ''}${kpis.sdg4_delta.toFixed(1)} pts vs 2023`;
  s4DeltaEl.className = `kpi-delta ${kpis.sdg4_delta >= 0 ? 'pos' : 'neg'}`;

  document.getElementById("kpi-sdg13-val").textContent = kpis.sdg13_avg.toFixed(1);
  const s13DeltaEl = document.getElementById("kpi-sdg13-delta");
  s13DeltaEl.textContent = `${kpis.sdg13_delta >= 0 ? '+' : ''}${kpis.sdg13_delta.toFixed(1)} pts vs 2023`;
  s13DeltaEl.className = `kpi-delta ${kpis.sdg13_delta >= 0 ? 'pos' : 'neg'}`;

  document.getElementById("kpi-aspirant-val").textContent = kpis.high_risk_count;
  document.getElementById("kpi-aspirant-desc").textContent = `States <50 in ${state.selectedYear}`;

  const critCount = state.data.early_warnings.filter(a => a.Severity === "CRITICAL").length;
  document.getElementById("kpi-alert-val").textContent = critCount;
}

// Render Main Trajectory Forecaster Line Chart (Chart.js)
function renderTrendChart() {
  const ctx = document.getElementById("trendChart").getContext("2d");
  if (state.charts.trendChart) state.charts.trendChart.destroy();

  const years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026];
  const stData = state.data.timeseries.filter(d => d.State === state.selectedState);

  const targets = (state.selectedSdgFilter === "ALL") 
    ? ["SDG3_Health", "SDG4_Education", "SDG13_Climate"]
    : [state.selectedSdgFilter];

  const datasets = [];

  targets.forEach(sdg => {
    const color = (sdg === "SDG3_Health") ? COLORS.sdg3 : ((sdg === "SDG4_Education") ? COLORS.sdg4 : COLORS.sdg13);
    const label = (sdg === "SDG3_Health") ? "SDG 3: Health" : ((sdg === "SDG4_Education") ? "SDG 4: Education" : "SDG 13: Climate");

    const histScores = [];
    const foreScores = [];
    const ciUpper = [];
    const ciLower = [];

    years.forEach(yr => {
      const rec = stData.find(d => d.SDG === sdg && d.Year === yr);
      if (yr <= 2023) {
        histScores.push(rec ? rec.Score : null);
        foreScores.push(yr === 2023 ? (rec ? rec.Score : null) : null);
      } else {
        histScores.push(null);
        foreScores.push(rec ? rec.Score : null);
      }
      ciUpper.push(rec ? rec.CI_Upper_95 : null);
      ciLower.push(rec ? rec.CI_Lower_95 : null);
    });

    // 1. Shaded Confidence Interval Upper Band
    datasets.push({
      label: `${label} (CI Upper)`,
      data: ciUpper,
      borderColor: "transparent",
      backgroundColor: "transparent",
      pointRadius: 0,
      fill: false
    });

    // 2. Shaded Confidence Interval Lower Band (fills to upper)
    datasets.push({
      label: `${label} (95% CI)`,
      data: ciLower,
      borderColor: "transparent",
      backgroundColor: color + "20",
      fill: "-1",
      pointRadius: 0
    });

    // 3. Historical Solid Line
    datasets.push({
      label: `${label} (Historical)`,
      data: histScores,
      borderColor: color,
      backgroundColor: color,
      borderWidth: 3,
      pointRadius: 5,
      pointHoverRadius: 7,
      tension: 0.2
    });

    // 4. Forecast Dashed Line
    datasets.push({
      label: `${label} (Forecast 2024-26)`,
      data: foreScores,
      borderColor: color,
      backgroundColor: color,
      borderWidth: 3,
      borderDash: [6, 4],
      pointRadius: 6,
      pointStyle: 'rectRot',
      pointHoverRadius: 8,
      tension: 0.1
    });
  });

  state.charts.trendChart = new Chart(ctx, {
    type: "line",
    data: { labels: years, datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: {
            color: COLORS.legend,
            font: { family: CHART_FONT, size: 11, weight: "500" },
            filter: item => !item.text.includes("CI Upper") && !item.text.includes("95% CI")
          }
        },
        tooltip: {
          backgroundColor: COLORS.tooltipBg,
          titleColor: COLORS.tooltipTitle,
          bodyColor: COLORS.tooltipBody,
          borderColor: "rgba(107, 143, 173, 0.35)",
          borderWidth: 1,
          padding: 10,
          callbacks: {
            label: context => {
              if (!context.raw) return null;
              return `${context.dataset.label}: ${context.raw.toFixed(1)} pts`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: COLORS.grid },
          ticks: { color: COLORS.tick, font: { family: CHART_FONT, weight: "500" } }
        },
        y: {
          min: 20,
          max: 100,
          grid: { color: COLORS.grid },
          ticks: { color: COLORS.tick },
          title: { display: true, text: "SDG Index Score (0-100)", color: "#6b7580" }
        }
      }
    }
  });
}

// Render Goal Scorecards in Sidebar
function renderGoalScorecards() {
  const container = document.getElementById("state-goal-cards");
  container.innerHTML = "";

  const stHistLatest = state.data.historical
    .filter(d => d.State === state.selectedState && d.Year === 2023)[0] || {};

  const sdgKeys = ["SDG3_Health", "SDG4_Education", "SDG13_Climate"];

  sdgKeys.forEach(sdg => {
    const foreRec = state.data.forecast.find(
      d => d.State === state.selectedState && d.SDG === sdg && d.Year === state.selectedYear
    ) || {};

    const slopeInfo = state.data.slopes[`${state.selectedState}__${sdg}`] || { slope: 0, r2: 0.8 };
    const currVal = stHistLatest[sdg] || 0;
    const projVal = foreRec.Forecast_Score || 0;
    const ciLow = foreRec.CI_Lower_95 || 0;
    const ciHi = foreRec.CI_Upper_95 || 0;

    let tierClass = "badge-low";
    let tierText = "Front Runner";
    let borderColor = COLORS.lowRisk;

    if (projVal < 50) {
      tierClass = "badge-high";
      tierText = "Aspirant (<50)";
      borderColor = COLORS.highRisk;
    } else if (projVal < 75) {
      tierClass = "badge-med";
      tierText = "Performer (50–74)";
      borderColor = COLORS.medRisk;
    }

    const card = document.createElement("div");
    card.className = "score-stat-card";
    card.style.borderLeftColor = borderColor;

    const labelName = state.data.sdg_labels[sdg].split(":")[1].trim();

    card.innerHTML = `
      <div class="stat-header">
        <span class="stat-title">${labelName}</span>
        <span class="badge ${tierClass}">${tierText}</span>
      </div>
      <div class="stat-header" style="margin-top: 6px;">
        <span class="stat-score" style="color: ${borderColor};">${projVal.toFixed(1)}</span>
        <span style="font-size: 11px; color: var(--text-muted);">2023: <b>${currVal.toFixed(1)}</b> (${slopeInfo.slope >= 0 ? '+' : ''}${slopeInfo.slope.toFixed(2)}/yr)</span>
      </div>
      <div class="stat-meta">
        <span>95% CI: [${ciLow.toFixed(1)} – ${ciHi.toFixed(1)}]</span>
        <span>Fit R²: <b>${slopeInfo.r2.toFixed(2)}</b></span>
      </div>
    `;
    container.appendChild(card);
  });
}

// Render Cross-State Comparative Trajectory Chart
function renderCompareChart() {
  const ctx = document.getElementById("compareChart").getContext("2d");
  if (state.charts.compareChart) state.charts.compareChart.destroy();

  const years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026];
  const statesToComp = [state.selectedState, state.compareState];
  const colorMap = [COLORS.sdg3, COLORS.composite];

  const datasets = [];

  statesToComp.forEach((stName, idx) => {
    const col = colorMap[idx % colorMap.length];
    const stData = state.data.timeseries.filter(d => d.State === stName && d.SDG === state.compareSdg);

    const histScores = [];
    const foreScores = [];

    years.forEach(yr => {
      const rec = stData.find(d => d.Year === yr);
      if (yr <= 2023) {
        histScores.push(rec ? rec.Score : null);
        foreScores.push(yr === 2023 ? (rec ? rec.Score : null) : null);
      } else {
        histScores.push(null);
        foreScores.push(rec ? rec.Score : null);
      }
    });

    datasets.push({
      label: `${stName} (Historical)`,
      data: histScores,
      borderColor: col,
      backgroundColor: col,
      borderWidth: 2.5,
      pointRadius: 4
    });

    datasets.push({
      label: `${stName} (Forecast)`,
      data: foreScores,
      borderColor: col,
      backgroundColor: col,
      borderWidth: 2.5,
      borderDash: [5, 4],
      pointRadius: 5,
      pointStyle: "rectRot"
    });
  });

  state.charts.compareChart = new Chart(ctx, {
    type: "line",
    data: { labels: years, datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: COLORS.legend, font: { family: CHART_FONT, size: 11, weight: "500" } } }
      },
      scales: {
        x: { grid: { color: COLORS.grid }, ticks: { color: COLORS.tick } },
        y: { min: 20, max: 100, grid: { color: COLORS.grid }, ticks: { color: COLORS.tick } }
      }
    }
  });
}

// Render Priority Early Warning Alerts
function renderPriorityAlerts() {
  const container = document.getElementById("priority-alerts-container");
  container.innerHTML = "";

  if (!state.data.early_warnings || state.data.early_warnings.length === 0) {
    container.innerHTML = `<div class="empty-state"><i data-lucide="circle-check" class="icon-sm"></i> No critical threshold drop alerts detected.</div>`;
    refreshIcons();
    return;
  }

  state.data.early_warnings.slice(0, 6).forEach(alert => {
    const card = document.createElement("div");
    card.className = `alert-card ${alert.Severity === 'CRITICAL' ? '' : (alert.Severity === 'WARNING' ? 'warning' : 'advisory')}`;

    let badgeClass = (alert.Severity === 'CRITICAL') ? 'badge-high' : ((alert.Severity === 'WARNING') ? 'badge-med' : 'badge-low');

    card.innerHTML = `
      <div>
        <span class="badge ${badgeClass}">${alert.Severity}</span>
        <b style="color: var(--text-primary); margin-left: 10px; font-size: 14px;">${alert.State} — ${alert.SDG}</b>
        <div style="color: var(--text-secondary); font-size: 13px; margin-top: 4px;">${alert.Description}</div>
      </div>
      <div style="text-align: right; min-width: 140px;">
        <span style="font-size: 11px; color: var(--text-muted);">2023 Baseline: <b>${alert['2023_Baseline'].toFixed(1)}</b></span><br>
        <span style="font-size: 13px; font-weight: 600; color: var(--status-negative);">2026 Projected: <b>${alert['2026_Projected'].toFixed(1)}</b></span>
      </div>
    `;
    container.appendChild(card);
  });
}

// Render Risk Matrix Data Table
function renderRiskMatrixTable() {
  const tbody = document.querySelector("#riskDataTable tbody");
  tbody.innerHTML = "";

  const searchVal = document.getElementById("tableSearchInput").value.toLowerCase();
  const riskFilter = document.getElementById("tableRiskFilter").value;
  const sdgFilter = document.getElementById("tableSdgFilter").value;

  const yearForecasts = state.data.forecast.filter(d => d.Year === state.selectedYear);

  let filtered = yearForecasts.filter(row => {
    if (searchVal && !row.State.toLowerCase().includes(searchVal)) return false;
    if (riskFilter !== "ALL" && row.Risk_Category !== riskFilter) return false;
    if (sdgFilter !== "ALL" && row.SDG !== sdgFilter) return false;
    return true;
  });

  filtered.sort((a, b) => b.Forecast_Score - a.Forecast_Score);

  filtered.forEach(row => {
    const histRec = state.data.historical.find(h => h.State === row.State && h.Year === 2023) || {};
    const baseVal = histRec[row.SDG] || 0;

    const tr = document.createElement("tr");

    let badgeClass = (row.Risk_Category === "Low Risk") ? "badge-low" : ((row.Risk_Category === "Medium Risk") ? "badge-med" : "badge-high");
    let sdgLabel = (row.SDG === "SDG3_Health") ? "SDG 3: Health" : ((row.SDG === "SDG4_Education") ? "SDG 4: Education" : "SDG 13: Climate");

    tr.innerHTML = `
      <td><b>${row.State}</b></td>
      <td>${sdgLabel}</td>
      <td>${baseVal.toFixed(1)}</td>
      <td style="font-weight: 600; font-size: 15px;">${row.Forecast_Score.toFixed(1)}</td>
      <td style="color: var(--text-muted);">[${row.CI_Lower_95.toFixed(1)} – ${row.CI_Upper_95.toFixed(1)}]</td>
      <td style="color: ${row.Annual_Growth_Rate >= 0 ? 'var(--status-positive)' : 'var(--status-negative)'}; font-weight: 600;">
        ${row.Annual_Growth_Rate >= 0 ? '+' : ''}${row.Annual_Growth_Rate.toFixed(2)}
      </td>
      <td><span class="badge ${badgeClass}">${row.NITI_Tier}</span></td>
      <td><span class="badge ${badgeClass}">${row.Risk_Category}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

// Render Spatial Geo Heatmap and Leaderboard
function renderGeoAnalytics() {
  const cardsGrid = document.getElementById("state-cards-grid");
  const leaderboard = document.getElementById("state-leaderboard-container");
  cardsGrid.innerHTML = "";
  leaderboard.innerHTML = "";

  const yearData = [];

  state.data.states.forEach(st => {
    let score = 50;
    if (state.selectedYear <= 2023) {
      const rec = state.data.historical.find(h => h.State === st && h.Year === state.selectedYear);
      if (rec) score = rec[state.geoMetric] || rec.Composite_Score || 50;
    } else {
      if (state.geoMetric === "Composite_Score") {
        const foreList = state.data.forecast.filter(f => f.State === st && f.Year === state.selectedYear);
        if (foreList.length > 0) {
          const sum = foreList.reduce((acc, c) => acc + c.Forecast_Score, 0);
          score = sum / foreList.length;
        }
      } else {
        const rec = state.data.forecast.find(f => f.State === st && f.SDG === state.geoMetric && f.Year === state.selectedYear);
        if (rec) score = rec.Forecast_Score;
      }
    }
    yearData.push({ state: st, score: score });
  });

  yearData.sort((a, b) => b.score - a.score);

  // Cards Grid
  yearData.forEach(item => {
    let col = (item.score >= 75) ? COLORS.lowRisk : ((item.score >= 50) ? COLORS.medRisk : COLORS.highRisk);
    let tier = (item.score >= 75) ? "Front Runner" : ((item.score >= 50) ? "Performer" : "Aspirant");

    const card = document.createElement("div");
    card.className = "score-stat-card";
    card.style.borderLeftColor = col;
    card.style.cursor = "pointer";
    card.addEventListener("click", () => {
      document.getElementById("stateSelect").value = item.state;
      state.selectedState = item.state;
      document.getElementById("forecaster-state-title").textContent = item.state;
      renderTrendChart();
      renderGoalScorecards();
      loadPolicyAdvisory();
    });

    card.innerHTML = `
      <div style="font-weight: 700; font-size: 13px;">${item.state}</div>
      <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;">
        <span style="font-size: 18px; font-weight: 600; color: ${col};">${item.score.toFixed(1)}</span>
        <span style="font-size: 11px; color: var(--text-muted);">${tier}</span>
      </div>
    `;
    cardsGrid.appendChild(card);
  });

  // Leaderboard Progress Bars
  yearData.forEach((item, idx) => {
    let col = (item.score >= 75) ? COLORS.lowRisk : ((item.score >= 50) ? COLORS.medRisk : COLORS.highRisk);

    const row = document.createElement("div");
    row.className = "leaderboard-row";
    row.innerHTML = `
      <div class="leaderboard-meta">
        <span>#${idx + 1} ${item.state}</span>
        <span style="color: ${col}; font-weight: 600;">${item.score.toFixed(1)}</span>
      </div>
      <div class="leaderboard-bar">
        <div class="leaderboard-fill" style="width: ${Math.min(100, Math.max(0, item.score))}%; background: ${col};"></div>
      </div>
    `;
    leaderboard.appendChild(row);
  });
}

// Render SDG 13 Climate Action Visuals
function renderClimateCharts() {
  // 1. Decarbonization Velocity Chart
  const ctxVel = document.getElementById("climateVelocityChart").getContext("2d");
  if (state.charts.climateVelChart) state.charts.climateVelChart.destroy();

  const climFore = state.data.forecast.filter(d => d.SDG === "SDG13_Climate" && d.Year === 2026);
  climFore.sort((a, b) => b.Annual_Growth_Rate - a.Annual_Growth_Rate);

  const topStates = climFore.slice(0, 15);

  state.charts.climateVelChart = new Chart(ctxVel, {
    type: "bar",
    data: {
      labels: topStates.map(d => d.State),
      datasets: [{
        label: "Annual Decarbonization Pace (pts/year)",
        data: topStates.map(d => d.Annual_Growth_Rate),
        backgroundColor: topStates.map(d => d.Annual_Growth_Rate >= 0 ? "rgba(138, 168, 154, 0.72)" : "rgba(176, 137, 137, 0.72)"),
        borderColor: topStates.map(d => d.Annual_Growth_Rate >= 0 ? COLORS.lowRisk : COLORS.highRisk),
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: COLORS.legend, font: { family: CHART_FONT, size: 11 } } } },
      scales: {
        x: { ticks: { color: COLORS.tick, font: { size: 10 } }, grid: { color: COLORS.grid } },
        y: { ticks: { color: COLORS.tick }, grid: { color: COLORS.grid } }
      }
    }
  });

  // 2. Renewable Energy Share Breakdown Chart
  const ctxRen = document.getElementById("renewableEnergyChart").getContext("2d");
  if (state.charts.renChart) state.charts.renChart.destroy();

  if (state.data.climate_indicators.length > 0) {
    const renCol = Object.keys(state.data.climate_indicators[0]).find(k => k.toLowerCase().includes("renewable"));
    const sortedClim = [...state.data.climate_indicators]
      .filter(d => d.Area !== "Target" && typeof d[renCol] === "number")
      .sort((a, b) => b[renCol] - a[renCol])
      .slice(0, 15);

    state.charts.renChart = new Chart(ctxRen, {
      type: "bar",
      data: {
        labels: sortedClim.map(d => d.Area),
        datasets: [{
          label: "Renewable Energy Share (%)",
          data: sortedClim.map(d => d[renCol]),
          backgroundColor: "rgba(123, 163, 201, 0.65)",
          borderColor: COLORS.sdg4,
          borderWidth: 1,
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: COLORS.legend, font: { family: CHART_FONT, size: 11 } } } },
        scales: {
          x: { ticks: { color: COLORS.tick, font: { size: 10 } }, grid: { color: COLORS.grid } },
          y: { max: 100, ticks: { color: COLORS.tick }, grid: { color: COLORS.grid } }
        }
      }
    });
  }
}

// Render Health Indicators Tab
function renderHealthIndicators() {
  const tbody = document.querySelector("#healthIndTable tbody");
  tbody.innerHTML = "";

  state.data.health_indicators.forEach(row => {
    if (row.Area === "Target") return;
    const tr = document.createElement("tr");

    const mmr = row["Maternal Mortality Ratio (per 100000 live births)"] || "-";
    const u5 = row["Under-five Mortality Rate (per 1000 live births)"] || "-";
    const tb = row["Total case notification rate of tuberculosis"] || "-";
    const fp = row["Women in the age group of 15-49 years using modern methods of family planning"] || "-";
    const doc = row["Number of governmental physicians nurses and midwives"] || "-";
    const imm = row["Children aged 12-23 months fully immunized (BCG Measles and three doses of Pentavalent vaccine)"] || "-";

    tr.innerHTML = `
      <td><b>${row.Area}</b></td>
      <td>${typeof u5 === 'number' ? u5.toFixed(1) : u5}</td>
      <td>${typeof tb === 'number' ? tb.toFixed(1) : tb}</td>
      <td>${typeof fp === 'number' ? fp.toFixed(1) + '%' : fp}</td>
      <td>${typeof doc === 'number' ? doc.toFixed(1) : doc}</td>
      <td>${typeof imm === 'number' ? imm.toFixed(1) + '%' : imm}</td>
      <td style="font-weight: 600; color: ${typeof mmr === 'number' && mmr > 100 ? 'var(--status-negative)' : 'var(--status-positive)'};">
        ${typeof mmr === 'number' ? mmr.toFixed(0) : mmr}
      </td>
    `;
    tbody.appendChild(tr);
  });

  renderHealthChart();
}

function renderHealthChart() {
  const ctx = document.getElementById("healthIndChart").getContext("2d");
  if (state.charts.healthChart) state.charts.healthChart.destroy();
  if (!state.healthIndicator) return;

  const validRows = state.data.health_indicators
    .filter(d => d.Area !== "Target" && typeof d[state.healthIndicator] === "number")
    .sort((a, b) => b[state.healthIndicator] - a[state.healthIndicator])
    .slice(0, 18);

  state.charts.healthChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: validRows.map(d => d.Area),
      datasets: [{
        label: state.healthIndicator,
        data: validRows.map(d => d[state.healthIndicator]),
        backgroundColor: "rgba(143, 184, 168, 0.65)",
        borderColor: COLORS.sdg3,
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: COLORS.legend, font: { family: CHART_FONT, size: 11 } } } },
      scales: {
        x: { ticks: { color: COLORS.tick, font: { size: 10 } }, grid: { color: COLORS.grid } },
        y: { ticks: { color: COLORS.tick }, grid: { color: COLORS.grid } }
      }
    }
  });
}

// Load Policy & Governance Advisory from Server API
async function loadPolicyAdvisory() {
  const summaryEl = document.getElementById("policy-executive-summary");
  const recContainer = document.getElementById("policy-recommendations-container");
  const natContainer = document.getElementById("national-priorities-container");

  try {
    const res = await fetch(`/api/policy/${encodeURIComponent(state.selectedState)}`);
    if (!res.ok) return;
    const advisory = await res.json();

    summaryEl.innerHTML = `
      <div class="policy-brief">
        <h4 style="color: var(--text-primary); margin-bottom: 8px; font-weight: 600;">Executive governance brief for state leadership</h4>
        <p style="color: var(--text-secondary); font-size: 14px;">
          Empirical projections indicate that <b>${state.selectedState}</b> requires the most critical developmental acceleration in
          <span style="color: var(--status-negative); font-weight: 600;">${advisory.priority_sdg}</span>.
          Fast-tracking targeted resource allocation and scheme enforcement will mitigate risk of target slippage by 2026.
        </p>
      </div>
    `;

    recContainer.innerHTML = "";
    advisory.recommendations.forEach(rec => {
      const box = document.createElement("div");
      box.className = "policy-item";

      let urgColor = (rec.urgency.includes("CRITICAL")) ? "var(--status-negative)" : ((rec.urgency.includes("ACCELERATION")) ? "var(--status-caution)" : "var(--status-positive)");
      box.style.borderLeftColor = urgColor;

      box.innerHTML = `
        <h4>
          <span>${rec.sdg_title}</span>
          <span class="badge" style="background: rgba(255,255,255,0.05); color: ${urgColor}; border: 1px solid ${urgColor};">${rec.urgency}</span>
        </h4>
        <div style="font-size: 12px; color: var(--text-muted); margin: 6px 0 10px 0;">
          2023 Current: <b>${rec['2023_score'].toFixed(1)}</b> → 2026 Projected: <b>${rec['2026_proj'].toFixed(1)}</b> (Velocity: ${rec.annual_velocity >= 0 ? '+' : ''}${rec.annual_velocity.toFixed(2)} pts/yr)
        </div>
        <div>
          <b style="font-size: 13px; color: var(--text-primary);">Recommended priority interventions:</b>
          <ul>
            ${rec.action_items.map(act => `<li>${act}</li>`).join('')}
          </ul>
        </div>
      `;
      recContainer.appendChild(box);
    });

    // National priorities
    const highRiskStates = state.data.forecast.filter(d => d.Year === 2026 && d.Forecast_Score < 50);
    if (highRiskStates.length > 0) {
      const grouped = {};
      highRiskStates.forEach(h => {
        if (!grouped[h.SDG]) grouped[h.SDG] = [];
        grouped[h.SDG].push(h.State);
      });

      let html = `<div style="font-size: 13px; line-height: 1.8;">`;
      Object.keys(grouped).forEach(k => {
        html += `<div style="margin-bottom: 8px;"><b>${state.data.sdg_labels[k]}:</b> <span style="color: var(--status-negative);">${grouped[k].join(', ')}</span></div>`;
      });
      html += `</div>`;
      natContainer.innerHTML = html;
    } else {
      natContainer.innerHTML = `<div class="empty-state"><i data-lucide="circle-check" class="icon-sm"></i> No states projected in Aspirant tier (&lt;50) by 2026.</div>`;
    }

    refreshIcons();
  } catch (err) {
    console.error("Error loading policy:", err);
  }
}

// Render Validation KPIs and Scorecard
function renderValidationSection() {
  const kpisGrid = document.getElementById("validation-kpis-grid");
  const tbody = document.querySelector("#valScorecardTable tbody");
  kpisGrid.innerHTML = "";
  tbody.innerHTML = "";

  state.data.validation_summary.forEach(summary => {
    let col = (summary.SDG === "SDG3_Health") ? COLORS.sdg3 : ((summary.SDG === "SDG4_Education") ? COLORS.sdg4 : COLORS.sdg13);
    let title = (summary.SDG === "SDG3_Health") ? "SDG 3 (Health)" : ((summary.SDG === "SDG4_Education") ? "SDG 4 (Education)" : "SDG 13 (Climate)");

    const card = document.createElement("div");
    card.className = "glass-panel kpi-card";
    card.innerHTML = `
      <div class="kpi-title">${title} Holdout R²</div>
      <div class="kpi-value" style="color: ${col};">${summary.mean_r2.toFixed(2)}</div>
      <div class="kpi-delta pos">Mean holdout RMSE: <b>${summary.mean_rmse.toFixed(2)}</b> pts</div>
      <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">Naive baseline beaten: <b>${summary.naive_beaten_pct}%</b> of states</div>
    `;
    kpisGrid.appendChild(card);
  });

  state.data.validation_records.slice(0, 36).forEach(row => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><b>${row.State}</b></td>
      <td>${row.SDG.replace('_', ' ')}</td>
      <td>${row.Holdout_Actual_2022.toFixed(1)}</td>
      <td>${row.Holdout_Pred_2022.toFixed(1)}</td>
      <td>${row.Holdout_Actual_2023.toFixed(1)}</td>
      <td>${row.Holdout_Pred_2023.toFixed(1)}</td>
      <td style="font-weight: 700;">${row.RMSE.toFixed(2)}</td>
      <td>${row.MAE.toFixed(2)}</td>
      <td style="font-weight: 600; color: var(--accent);">${row.R2.toFixed(2)}</td>
      <td><span class="badge ${row.Naive_Beaten ? 'badge-low' : 'badge-med'}">${row.Naive_Beaten ? 'Yes' : 'Equal'}</span></td>
    `;
    tbody.appendChild(tr);
  });
}
