import { apiGet, apiPost, closeModal, esc, openModal, toast } from "./api.js";
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
  setActiveJob(jobId, text = "") {
    const pill = document.getElementById("active-job-pill");
    const label = document.getElementById("active-job-label");
    if (!pill || !label) return;
    if (jobId) {
      pill.classList.remove("hidden");
      label.textContent = text || `Job ${jobId}`;
    } else {
      pill.classList.add("hidden");
      label.textContent = "";
    }
  },
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

  const sidebar = document.getElementById("sidebar");
  const navToggle = document.getElementById("btn-nav-toggle");
  if (sidebar?.classList.contains("nav-open")) {
    sidebar.classList.remove("nav-open");
    navToggle?.setAttribute("aria-expanded", "false");
  }

  const pageRoot = document.getElementById("page-root");
  const slot = document.createElement("div");
  slot.dataset.routeSeq = String(seq);
  slot.innerHTML = `<div class="loading-box">Loading ${esc(cfg.title)}...</div>`;
  pageRoot.replaceChildren(slot);

  try {
    await cfg.render(slot, appState);
  } catch (err) {
    if (seq !== renderSeq || !slot.isConnected) return;
    slot.innerHTML = `<div class="card"><h3 class="card-title">Failed to load ${esc(cfg.title)}</h3><p class="card-sub mt-12">${esc(err.message)}</p></div>`;
    toast(err.message, "err");
  }
}

function openQuickGuideModal() {
  openModal(
    "Quick Start Guide & Plain-English Glossary",
    `
      <p class="card-sub mb-12">
        <strong>work121</strong> tests whether AI agents and multi-agent tool pipelines resist prompt injection and framing smuggling — backed by deterministic statistics and wire-level checks.
      </p>
      <div class="guide-grid">
        <div class="guide-step">
          <div class="guide-step-title">1. Connect or Auto-Discover a Model</div>
          <p>Register a live local <strong>Ollama</strong> model with one click (Auto-Discover) or connect OpenAI-compatible, Anthropic, or Gemini endpoints. Simulated baseline agents work immediately without an API key.</p>
          <div class="btn-row"><button class="btn btn-secondary btn-sm" type="button" data-guide-route="overview">Go to Overview</button></div>
        </div>
        <div class="guide-step">
          <div class="guide-step-title">2. Run One-Tap or Custom Evals</div>
          <p>Use <strong>One-Tap Full Suite Launch</strong> on Overview for the 4-stage pipeline, or configure benign + indirect injection sweeps in <strong>Eval Suite &amp; Runner</strong>.</p>
          <div class="btn-row"><button class="btn btn-secondary btn-sm" type="button" data-guide-route="configure-run">Go to Eval Runner</button></div>
        </div>
        <div class="guide-step">
          <div class="guide-step-title">3. Inspect Heatmaps &amp; Significance</div>
          <p>Measure exact failure rates with <strong>Wilson 95% confidence intervals</strong> and compare two versions via exact <strong>Fisher</strong> and paired <strong>McNemar</strong> p-values.</p>
          <div class="btn-row"><button class="btn btn-secondary btn-sm" type="button" data-guide-route="susceptibility">Go to Stats &amp; Heatmaps</button></div>
        </div>
        <div class="guide-step">
          <div class="guide-step-title">4. Release Gate &amp; OWASP Audit</div>
          <p>Enforce automated <code>ADMIT</code> / <code>CONDITIONAL_REVIEW</code> / <code>BLOCK</code> release decisions in the <strong>Hamster Wheel</strong> and inspect <strong>OWASP LLM01 / LLM07 / LLM08</strong> controls.</p>
          <div class="btn-row"><button class="btn btn-secondary btn-sm" type="button" data-guide-route="owasp">Go to OWASP Audit</button></div>
        </div>
      </div>
      <div class="card mt-16">
        <h4 class="card-title mb-12">Key Terms Explained</h4>
        <ul class="card-sub" style="margin:0;padding-left:18px;display:grid;gap:6px;">
          <li><strong>Wire Gate (Python + Rust):</strong> Verifies framed messages (<code>raw</code>, <code>length_prefixed</code>, <code>ndjson</code>) with HMAC-SHA256 replay protection before prompts reach an agent.</li>
          <li><strong>2-Hop Relay Ablation:</strong> Runs the same task with and without gateway boundary defenses to quantify injection reduction between Agent A and Agent B.</li>
          <li><strong>Hamster Wheel Gauntlet:</strong> Multi-cycle endurance gate that blocks unstable or drifting agent builds from shipping.</li>
          <li><strong>Regression Diff:</strong> Compares two saved JSONL runs pinned to the same SHA-256 prompt template hash.</li>
        </ul>
      </div>
    `
  );
  document.querySelectorAll("#modal-body [data-guide-route]").forEach((btn) => {
    btn.addEventListener("click", () => {
      closeModal();
      appState.navigate(btn.dataset.guideRoute);
    });
  });
}

function initShell() {
  document.getElementById("modal-close")?.addEventListener("click", closeModal);
  document.getElementById("modal-backdrop")?.addEventListener("click", (ev) => {
    if (ev.target.id === "modal-backdrop") closeModal();
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") closeModal();
  });

  document.getElementById("btn-quick-help")?.addEventListener("click", openQuickGuideModal);

  const navToggle = document.getElementById("btn-nav-toggle");
  navToggle?.addEventListener("click", () => {
    const sidebar = document.getElementById("sidebar");
    if (!sidebar) return;
    const isOpen = sidebar.classList.toggle("nav-open");
    navToggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
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
