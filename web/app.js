import { apiGet, apiPost, closeModal, toast } from "./api.js";
import {
  renderMcpPage,
  renderTaxonomyPage,
  renderWireLabPage,
} from "./pages_wire.js";
import {
  renderConfigureRunPage,
  renderHamsterWheelPage,
  renderOverviewPage,
  renderRelayLabPage,
} from "./pages_eval.js";
import {
  renderOwaspPage,
  renderRegressionDiffPage,
  renderSusceptibilityPage,
} from "./pages_analysis.js";

const ROUTES = {
  overview: {
    title: "Overview & Telemetry",
    subtitle: "Unified backend, native binary, live endpoints, and evaluation telemetry",
    render: renderOverviewPage,
  },
  "wire-lab": {
    title: "Wire Framing & Hex Lab",
    subtitle: "Raw/Length-Prefixed/NDJSON byte streams, hex dissection, and Python + Rust gate parity",
    render: renderWireLabPage,
  },
  "configure-run": {
    title: "Eval Suite & Runner",
    subtitle: "Parameterized benign edge-case templates, indirect suites, and live WebSocket execution",
    render: renderConfigureRunPage,
  },
  taxonomy: {
    title: "Indirect Taxonomy & Wire",
    subtitle: "9 cross-architecture probes across 5 platform wire protocols and handoff matrix",
    render: renderTaxonomyPage,
  },
  "mcp-multimodal": {
    title: "MCP Tools & Multimodal",
    subtitle: "Live 9-tool JSON-RPC 2.0 loops, Pillow PNG probes, and disk BM25 corpus",
    render: renderMcpPage,
  },
  "relay-lab": {
    title: "2-Hop Relay Lab",
    subtitle: "Agent-A to Gateway to Agent-B protocol handoff ablations and trace inspector",
    render: renderRelayLabPage,
  },
  "hamster-wheel": {
    title: "Hamster Wheel Gauntlet",
    subtitle: "Continuous multi-cycle defensive evaluation and ADMIT/BLOCK gating",
    render: renderHamsterWheelPage,
  },
  susceptibility: {
    title: "Susceptibility & Stats",
    subtitle: "Outcome heatmaps, Wilson 95% CIs, exact Fisher/McNemar significance, and exports",
    render: renderSusceptibilityPage,
  },
  "regression-diff": {
    title: "Version Regression Diff",
    subtitle: "SHA-256 template-pinned comparison across saved runs or uploaded JSONL artifacts",
    render: renderRegressionDiffPage,
  },
  owasp: {
    title: "OWASP Audit & Controls",
    subtitle: "Automated OWASP LLM Top 10 remediation plans and interactive boundary validator",
    render: renderOwaspPage,
  },
};

function currentRouteFromHash() {
  const raw = (location.hash || "#/overview").replace(/^#\/?/, "").split("?")[0].trim();
  return ROUTES[raw] ? raw : "overview";
}

const appState = {
  queuedBenign: [],
  queuedIndirect: [],
  lastOverview: null,
  navigate(route) {
    if (!ROUTES[route]) route = "overview";
    location.hash = `#/${route}`;
  },
  async refreshStatus() {
    const apiEl = document.getElementById("status-api");
    const binEl = document.getElementById("status-bin");
    const agentsEl = document.getElementById("status-agents");
    try {
      const ov = await apiGet("/api/overview");
      appState.lastOverview = ov;
      if (apiEl) {
        apiEl.className = "badge badge-ok";
        apiEl.textContent = "ONLINE";
      }
      if (binEl) {
        binEl.className = ov.native_binary ? "badge badge-ok" : "badge badge-muted";
        binEl.textContent = ov.native_binary ? "READY" : "NOT BUILT";
        binEl.title = ov.native_binary || "Compile native/ to enable";
      }
      if (agentsEl) {
        agentsEl.textContent = `${ov.counts?.live_agents ?? 0} live / ${ov.counts?.registered_agents ?? 0}`;
      }
      return ov;
    } catch (err) {
      if (apiEl) {
        apiEl.className = "badge badge-bad";
        apiEl.textContent = "OFFLINE";
      }
      return null;
    }
  },
};

let renderSeq = 0;

async function renderCurrentRoute() {
  const seq = ++renderSeq;
  const route = currentRouteFromHash();
  const cfg = ROUTES[route];

  document.getElementById("page-title").textContent = cfg.title;
  document.getElementById("page-subtitle").textContent = cfg.subtitle;
  document.querySelectorAll("#nav-list .nav-item").forEach((a) => {
    a.classList.toggle("active", a.dataset.route === route);
  });

  const root = document.getElementById("page-root");
  root.innerHTML = `<div class="loading-box">Loading ${cfg.title}...</div>`;

  try {
    await cfg.render(root, appState);
  } catch (err) {
    if (seq !== renderSeq) return;
    root.innerHTML = `<div class="card"><h3 class="card-title">Failed to load ${cfg.title}</h3><p class="card-sub mt-12">${err.message}</p></div>`;
    toast(err.message, "err");
  }
}

function initShell() {
  document.getElementById("modal-close")?.addEventListener("click", closeModal);
  document.getElementById("modal-backdrop")?.addEventListener("click", (ev) => {
    if (ev.target.id === "modal-backdrop") closeModal();
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") closeModal();
  });

  document.getElementById("btn-refresh-discovery")?.addEventListener("click", async () => {
    try {
      const res = await apiPost("/api/agents/discover", {});
      await appState.refreshStatus();
      toast(`Discovered ${res.discovered.length} live endpoint(s).`, "ok");
      if (currentRouteFromHash() === "overview") {
        await renderCurrentRoute();
      }
    } catch (err) {
      toast(err.message, "err");
    }
  });

  window.addEventListener("hashchange", renderCurrentRoute);
  appState.refreshStatus();
  renderCurrentRoute();
}

initShell();
