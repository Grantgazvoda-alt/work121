import {
  apiGet,
  apiPost,
  bindPager,
  connectJobStream,
  esc,
  openModal,
  outcomeBadge,
  paginate,
  toast,
} from "./api.js";
import { renderBarChartSvg } from "./charts.js";

export async function renderOverviewPage(root, appState) {
  const [overview, jobsRes, runsRes] = await Promise.all([
    apiGet("/api/overview"),
    apiGet("/api/jobs"),
    apiGet("/api/runs"),
  ]);

  const c = overview.counts || {};
  const agents = overview.agents || [];
  const jobs = jobsRes.jobs || [];
  const runs = runsRes.runs || [];

  let agentPage = 1;
  let jobPage = 1;
  const pageSize = 8;

  const safeBars = runs.slice(0, 10).map((r) => ({
    label: `${r.model_id}@${r.model_version}`,
    value: Number(((r.safe_rate || 0) * 100).toFixed(1)),
    color: r.safe_rate >= 0.9 ? "#10b981" : r.safe_rate >= 0.7 ? "#f59e0b" : "#ef4444",
  }));

  root.innerHTML = `
    <div class="grid grid-5">
      <div class="kpi">
        <div class="kpi-label">Saved Eval Runs</div>
        <div class="kpi-value">${esc(c.saved_runs ?? 0)}</div>
        <div class="kpi-note">${esc(c.total_eval_records ?? 0)} artifact records</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Registered Agents</div>
        <div class="kpi-value">${esc(c.registered_agents ?? 0)}</div>
        <div class="kpi-note">${esc(c.live_agents ?? 0)} live endpoint(s)</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Wire Matrix Cases</div>
        <div class="kpi-value">${esc(c.wire_cases ?? 0)}</div>
        <div class="kpi-note">${overview.native_binary ? "Rust binary parity READY" : "Rust binary NOT BUILT"}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">MCP Tools &amp; Corpus</div>
        <div class="kpi-value">${esc(c.mcp_tools ?? 0)} / ${esc(c.corpus_docs ?? 0)}</div>
        <div class="kpi-note">${esc(c.indirect_templates ?? 0)} indirect + ${esc(c.benign_templates ?? 0)} benign templates</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Relays &amp; Wheel</div>
        <div class="kpi-value">${esc(c.relay_traces ?? 0)} / ${esc(c.wheel_reports ?? 0)}</div>
        <div class="kpi-note">2-hop traces / admission reports</div>
      </div>
    </div>

    <div class="grid grid-2 mt-16">
      <div class="card">
        <div class="card-header">
          <div>
            <h2 class="card-title">1. Run Adherence Overview (% Safe Rate)</h2>
            <p class="card-sub">Persisted evaluation artifacts in <code>${esc(overview.runs_dir)}</code>.</p>
          </div>
          <div class="btn-row">
            <button class="btn btn-primary btn-sm" id="ov-go-run" type="button">Launch Eval</button>
            <button class="btn btn-secondary btn-sm" id="ov-go-wire" type="button">Wire Lab</button>
            <button class="btn btn-secondary btn-sm" id="ov-go-wheel" type="button">Hamster Wheel</button>
          </div>
        </div>
        <div>${renderBarChartSvg(safeBars, { height: 220, valueFmt: (v) => `${v}%` })}</div>
      </div>

      <div class="card">
        <div class="card-header">
          <div>
            <h2 class="card-title">2. Register Live LLM Endpoint</h2>
            <p class="card-sub">Connect Ollama, OpenAI-compatible, Anthropic, or Gemini models directly into the workbench.</p>
          </div>
          <button class="btn btn-secondary btn-sm" id="ov-discover-btn" type="button">Auto-Discover</button>
        </div>
        <div class="grid grid-2">
          <label class="field">Provider
            <select id="reg-provider">
              <option value="ollama">ollama</option>
              <option value="openai_compatible">openai_compatible</option>
              <option value="anthropic">anthropic</option>
              <option value="gemini">gemini</option>
            </select>
          </label>
          <label class="field">Model ID
            <input type="text" id="reg-model-id" placeholder="e.g. qwen2.5-coder:7b or gpt-4o-mini" />
          </label>
        </div>
        <div class="grid grid-3 mt-12">
          <label class="field">Version Tag
            <input type="text" id="reg-version" value="live-custom" />
          </label>
          <label class="field">Base URL (optional)
            <input type="text" id="reg-base-url" placeholder="http://127.0.0.1:11434" />
          </label>
          <label class="field">API Key (optional, env fallback)
            <input type="password" id="reg-api-key" placeholder="Uses env key if blank" />
          </label>
        </div>
        <div class="btn-row mt-12">
          <button class="btn btn-secondary" id="btn-test-endpoint" type="button">Health Check</button>
          <button class="btn btn-primary" id="btn-reg-endpoint" type="button">Register Target Agent</button>
          <span class="card-sub" id="reg-health-status"></span>
        </div>
      </div>
    </div>

    <div class="card mt-16">
      <div class="card-header">
        <div>
          <h2 class="card-title">3. Target Agent Registry (Live &amp; Simulated)</h2>
          <p class="card-sub">Verify reachability or inspect architecture families across all registered targets.</p>
        </div>
        <input type="text" id="ov-agent-filter" placeholder="Filter model / provider..." style="width:240px" />
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Agent Key</th>
              <th>Kind</th>
              <th>Provider</th>
              <th>Architecture</th>
              <th>Capabilities / Notes</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody id="ov-agents-body"></tbody>
        </table>
      </div>
      <div class="pager" id="ov-agents-pager"></div>
    </div>

    <div class="card mt-16">
      <div class="card-header">
        <div>
          <h2 class="card-title">4. Background Job &amp; WebSocket Telemetry Log</h2>
          <p class="card-sub">Inspect recent asynchronous evaluation or Hamster Wheel jobs.</p>
        </div>
        <button class="btn btn-ghost btn-sm" id="ov-refresh-jobs" type="button">Refresh Jobs</button>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Job ID</th>
              <th>Kind</th>
              <th>Title</th>
              <th>Status</th>
              <th>Events</th>
              <th>Created (UTC)</th>
            </tr>
          </thead>
          <tbody id="ov-jobs-body"></tbody>
        </table>
      </div>
      <div class="pager" id="ov-jobs-pager"></div>
    </div>
  `;

  document.getElementById("ov-go-run").addEventListener("click", () => appState.navigate("configure-run"));
  document.getElementById("ov-go-wire").addEventListener("click", () => appState.navigate("wire-lab"));
  document.getElementById("ov-go-wheel").addEventListener("click", () => appState.navigate("hamster-wheel"));

  let currentAgents = agents;
  let currentJobs = jobs;

  function renderAgentsTable() {
    const q = document.getElementById("ov-agent-filter").value.trim().toLowerCase();
    const filtered = currentAgents.filter((a) =>
      !q ||
      a.agent_key.toLowerCase().includes(q) ||
      (a.provider || "").toLowerCase().includes(q) ||
      (a.architecture_family || "").toLowerCase().includes(q)
    );
    const pg = paginate(filtered, agentPage, pageSize);
    agentPage = pg.page;

    const tbody = document.getElementById("ov-agents-body");
    tbody.innerHTML = pg.slice.map((a) => {
      const caps = a.kind === "live"
        ? `${a.is_multimodal ? "Vision/Multimodal " : ""}${esc(a.base_url || "")}`
        : `Susceptible: ${esc((a.susceptible_categories || []).join(", ") || "None (hardened)")}`;
      return `
        <tr>
          <td><code>${esc(a.agent_key)}</code></td>
          <td>${a.kind === "live" ? `<span class="badge badge-info">LIVE</span>` : `<span class="badge badge-muted">SIMULATED</span>`}</td>
          <td><code>${esc(a.provider)}</code></td>
          <td><code>${esc(a.architecture_family)}</code></td>
          <td class="card-sub">${caps}</td>
          <td><button class="btn btn-ghost btn-sm btn-agent-ping" data-key="${esc(a.agent_key)}" type="button">Ping</button></td>
        </tr>
      `;
    }).join("") || `<tr><td colspan="6" class="card-sub">No matching agents.</td></tr>`;

    tbody.querySelectorAll(".btn-agent-ping").forEach((btn) => {
      btn.addEventListener("click", async () => {
        try {
          const res = await apiPost("/api/agents/health", { agent_key: btn.dataset.key });
          toast(`${btn.dataset.key}: ${res.message}`, res.ok ? "ok" : "err");
        } catch (err) {
          toast(err.message, "err");
        }
      });
    });

    bindPager(document.getElementById("ov-agents-pager"), pg.totalItems, pg.page, pageSize, (p) => {
      agentPage = p;
      renderAgentsTable();
    });
  }

  function renderJobsTable() {
    const pg = paginate(currentJobs, jobPage, pageSize);
    jobPage = pg.page;
    const tbody = document.getElementById("ov-jobs-body");
    tbody.innerHTML = pg.slice.map((j) => {
      const badge = j.status === "completed"
        ? `<span class="badge badge-ok">completed</span>`
        : j.status === "failed"
          ? `<span class="badge badge-bad">failed</span>`
          : `<span class="badge badge-info">${esc(j.status)}</span>`;
      return `
        <tr class="clickable" data-job-id="${esc(j.job_id)}">
          <td><code>${esc(j.job_id)}</code></td>
          <td><code>${esc(j.kind)}</code></td>
          <td>${esc(j.title)}</td>
          <td>${badge}</td>
          <td><code>${esc(j.event_count)}</code></td>
          <td class="card-sub">${esc((j.created_at_utc || "").replace("T", " ").slice(0, 19))}</td>
        </tr>
      `;
    }).join("") || `<tr><td colspan="6" class="card-sub">No background jobs recorded in this server session yet.</td></tr>`;

    tbody.querySelectorAll("tr[data-job-id]").forEach((tr) => {
      tr.addEventListener("click", async () => {
        try {
          const detail = await apiGet(`/api/jobs/${encodeURIComponent(tr.dataset.jobId)}`);
          openModal(
            `Job ${detail.job_id} (${detail.status})`,
            `<pre class="code-box">${esc(JSON.stringify(detail, null, 2))}</pre>`
          );
        } catch (err) {
          toast(err.message, "err");
        }
      });
    });

    bindPager(document.getElementById("ov-jobs-pager"), pg.totalItems, pg.page, pageSize, (p) => {
      jobPage = p;
      renderJobsTable();
    });
  }

  document.getElementById("ov-agent-filter").addEventListener("input", () => {
    agentPage = 1;
    renderAgentsTable();
  });

  document.getElementById("ov-discover-btn").addEventListener("click", async () => {
    try {
      const res = await apiPost("/api/agents/discover", {});
      currentAgents = res.agents || [];
      agentPage = 1;
      renderAgentsTable();
      await appState.refreshStatus();
      toast(`Discovered ${res.discovered.length} live endpoint(s).`, "ok");
    } catch (err) {
      toast(err.message, "err");
    }
  });

  function readEndpointPayload() {
    return {
      provider: document.getElementById("reg-provider").value,
      model_id: document.getElementById("reg-model-id").value.trim(),
      model_version: document.getElementById("reg-version").value.trim() || "live-custom",
      base_url: document.getElementById("reg-base-url").value.trim(),
      api_key: document.getElementById("reg-api-key").value.trim() || null,
    };
  }

  document.getElementById("btn-test-endpoint").addEventListener("click", async () => {
    try {
      const payload = readEndpointPayload();
      const res = await apiPost("/api/agents/health", payload);
      document.getElementById("reg-health-status").textContent = res.message;
      toast(res.message, res.ok ? "ok" : "err");
    } catch (err) {
      toast(err.message, "err");
    }
  });

  document.getElementById("btn-reg-endpoint").addEventListener("click", async () => {
    try {
      const payload = readEndpointPayload();
      const res = await apiPost("/api/agents/register", payload);
      toast(`Registered ${res.registered}`, "ok");
      const refreshed = await apiGet("/api/agents");
      currentAgents = refreshed.agents || [];
      renderAgentsTable();
      await appState.refreshStatus();
    } catch (err) {
      toast(err.message, "err");
    }
  });

  document.getElementById("ov-refresh-jobs").addEventListener("click", async () => {
    const jr = await apiGet("/api/jobs");
    currentJobs = jr.jobs || [];
    renderJobsTable();
  });

  renderAgentsTable();
  renderJobsTable();
}

export async function renderConfigureRunPage(root, appState) {
  const [benignRes, taxRes, agentsRes] = await Promise.all([
    apiGet("/api/templates/benign"),
    apiGet("/api/taxonomy"),
    apiGet("/api/agents"),
  ]);

  const templates = benignRes.templates || [];
  const indirectTemplates = taxRes.templates || [];
  const agents = agentsRes.agents || [];
  const protocols = taxRes.facets?.protocols || [];
  const categories = Array.from(new Set(templates.map((t) => t.category)));

  let queuePage = 1;
  let streamPage = 1;
  const pageSize = 10;
  let streamRows = [];
  let latestPreview = null;

  root.innerHTML = `
    <div class="grid grid-2">
      <div class="card">
        <div class="card-header">
          <div>
            <h2 class="card-title">1. Parameterized Benign Edge-Case Builder</h2>
            <p class="card-sub">Configure template variables and preview SHA-256 pinned payloads before queueing.</p>
          </div>
        </div>
        <div class="grid grid-2">
          <label class="field">Pattern Category
            <select id="cfg-cat">
              ${categories.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("")}
            </select>
          </label>
          <label class="field">Template
            <select id="cfg-tpl"></select>
          </label>
        </div>
        <p class="card-sub mt-12" id="cfg-tpl-desc"></p>
        <div id="cfg-params-grid" class="grid grid-2 mt-12"></div>
        <label class="field mt-12">Custom Test Case ID (optional)
          <input type="text" id="cfg-case-id" placeholder="Auto-generated from template_id + SHA-256" />
        </label>
        <div class="btn-row mt-12">
          <button class="btn btn-primary" id="btn-cfg-add" type="button">Queue Configured Case</button>
          <button class="btn btn-secondary" id="btn-cfg-load-benign" type="button">Load All ${templates.length} Benign Defaults</button>
          <button class="btn btn-secondary" id="btn-cfg-load-indirect" type="button">Load All ${indirectTemplates.length} Paired Indirect</button>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div>
            <h2 class="card-title">2. Live Rendered Preview</h2>
            <p class="card-sub" id="cfg-preview-meta">Select parameters to render.</p>
          </div>
        </div>
        <div class="card-sub mb-12" id="cfg-preview-boundary"></div>
        <pre class="code-box" id="cfg-preview-payload"></pre>
      </div>
    </div>

    <div class="card mt-16">
      <div class="card-header">
        <div>
          <h2 class="card-title">3. Active Evaluation Queue (Benign + Paired Indirect)</h2>
          <p class="card-sub">Queue specific cases above or from Indirect Taxonomy, or use a Suite Shortcut below.</p>
        </div>
        <button class="btn btn-danger btn-sm" id="btn-cfg-clear" type="button">Clear Queue</button>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Type</th><th>Case / Template ID</th><th>Category</th><th>Protocol / Turns</th><th>Action</th></tr>
          </thead>
          <tbody id="cfg-queue-body"></tbody>
        </table>
      </div>
      <div class="pager" id="cfg-queue-pager"></div>
    </div>

    <div class="card mt-16">
      <div class="card-header">
        <div>
          <h2 class="card-title">4. Target Agents, Scoring Engine &amp; Real-Time Execution</h2>
          <p class="card-sub">Executes asynchronously with live WebSocket telemetry and deterministic JSONL serialization.</p>
        </div>
      </div>
      <div class="grid grid-4">
        <label class="field">Suite Mode (when queue is empty)
          <select id="run-suite-mode">
            <option value="benign">All Benign Edge Cases</option>
            <option value="indirect">All Paired Indirect Cases</option>
            <option value="all">Full Suite (Benign + Paired Indirect)</option>
          </select>
        </label>
        <label class="field">Target Protocol (for indirect suite)
          <select id="run-protocol">
            <option value="">native (per-template default)</option>
            ${protocols.map((p) => `<option value="${esc(p)}">${esc(p)}</option>`).join("")}
          </select>
        </label>
        <label class="field">Scoring Engine
          <select id="run-scoring">
            <option value="hybrid">Hybrid (Invariant + Live LLM + PolicyAnchor)</option>
            <option value="policy_anchor">PolicyAnchorSemanticJudge (offline)</option>
            <option value="heuristic">Heuristic Baseline</option>
          </select>
        </label>
        <label class="field">Drift Slope Threshold
          <input type="number" id="run-drift-tol" value="-0.08" step="0.01" max="-0.01" min="-0.5" />
        </label>
      </div>

      <div class="grid grid-2 mt-12">
        <div>
          <div class="field" style="margin-bottom:6px">Target Agents / Model Versions</div>
          <div id="run-agents-box" class="grid grid-2">
            ${agents.map((a, idx) => `
              <label class="check-row">
                <input type="checkbox" class="run-agent-chk" value="${esc(a.agent_key)}" ${idx < 2 ? "checked" : ""} />
                <span><code>${esc(a.agent_key)}</code> ${a.kind === "live" ? `<span class="badge badge-info">LIVE</span>` : ""}</span>
              </label>
            `).join("")}
          </div>
        </div>
        <div class="grid grid-2">
          <label class="field">Optional Live LLM Judge Agent
            <select id="run-judge-agent">
              <option value="">None (use deterministic tiers)</option>
              ${agents.filter((a) => a.kind === "live").map((a) => `<option value="${esc(a.agent_key)}">${esc(a.agent_key)}</option>`).join("")}
            </select>
          </label>
          <div class="btn-row" style="align-content:center">
            <label class="check-row"><input type="checkbox" id="run-attach-mcp" checked /> Attach 9 MCP Tools</label>
            <label class="check-row"><input type="checkbox" id="run-use-embed" /> Ollama Embeddings</label>
            <label class="check-row"><input type="checkbox" id="run-redact" /> Redact Raw Text</label>
            <label class="check-row"><input type="checkbox" id="run-paired" checked /> Include Paired Controls</label>
          </div>
        </div>
      </div>

      <div class="btn-row mt-16">
        <button class="btn btn-primary" id="btn-launch-eval" type="button">Run Real-Time Evaluation</button>
        <span class="card-sub" id="run-progress-label">Ready.</span>
      </div>
      <div class="progress-track mt-12">
        <div class="progress-fill" id="run-progress-fill" style="width:0%"></div>
      </div>

      <div class="table-wrap mt-16">
        <table>
          <thead>
            <tr>
              <th>Agent</th>
              <th>Test Case</th>
              <th>Category</th>
              <th>Outcome</th>
              <th>Conf.</th>
              <th>Latency</th>
              <th>Signals &amp; Excerpt</th>
            </tr>
          </thead>
          <tbody id="run-stream-body"></tbody>
        </table>
      </div>
      <div class="pager" id="run-stream-pager"></div>
    </div>
  `;

  function populateTemplates() {
    const cat = document.getElementById("cfg-cat").value;
    const filtered = templates.filter((t) => t.category === cat);
    const sel = document.getElementById("cfg-tpl");
    sel.innerHTML = filtered.map((t) => `<option value="${esc(t.template_id)}">${esc(t.name)} (${esc(t.template_id)})</option>`).join("");
    renderParamInputs();
  }

  async function renderParamInputs() {
    const tid = document.getElementById("cfg-tpl").value;
    const tpl = templates.find((t) => t.template_id === tid);
    if (!tpl) return;
    document.getElementById("cfg-tpl-desc").textContent = tpl.description;
    const grid = document.getElementById("cfg-params-grid");
    grid.innerHTML = tpl.parameters.map((p) => {
      if (p.allowed_values && p.allowed_values.length) {
        return `
          <label class="field">${esc(p.name)}
            <select class="cfg-param-input" data-pname="${esc(p.name)}" title="${esc(p.description)}">
              ${p.allowed_values.map((v) => `<option value="${esc(v)}" ${v === p.default ? "selected" : ""}>${esc(v)}</option>`).join("")}
            </select>
          </label>`;
      }
      return `
        <label class="field">${esc(p.name)}
          <input type="text" class="cfg-param-input" data-pname="${esc(p.name)}" value="${esc(p.default)}" title="${esc(p.description)}" />
        </label>`;
    }).join("");

    grid.querySelectorAll(".cfg-param-input").forEach((el) => {
      el.addEventListener("change", updatePreview);
      el.addEventListener("input", updatePreview);
    });
    await updatePreview();
  }

  async function updatePreview() {
    const tid = document.getElementById("cfg-tpl").value;
    if (!tid) return;
    const params = {};
    document.querySelectorAll(".cfg-param-input").forEach((el) => {
      params[el.dataset.pname] = el.value;
    });
    const customId = document.getElementById("cfg-case-id").value.trim();
    try {
      const rendered = await apiPost("/api/templates/benign/render", {
        template_id: tid,
        parameters: params,
        test_case_id: customId,
      });
      latestPreview = rendered;
      document.getElementById("cfg-preview-meta").textContent =
        `Case: ${rendered.test_case_id} | SHA-256: ${rendered.template_sha256.slice(0, 14)}... | Turns: ${rendered.user_turns.length}`;
      document.getElementById("cfg-preview-boundary").innerHTML =
        `<strong>System Boundary:</strong> ${esc(rendered.system_boundary)}`;
      document.getElementById("cfg-preview-payload").textContent = rendered.rendered_payload;
    } catch (err) {
      document.getElementById("cfg-preview-payload").textContent = `Preview error: ${err.message}`;
    }
  }

  function combinedQueue() {
    const b = appState.queuedBenign.map((item, idx) => ({
      kind: "benign",
      idx,
      id: item.test_case_id,
      category: item.category,
      detail: `${item.template_id} (${item.user_turns.length} turn(s))`,
    }));
    const ind = appState.queuedIndirect.map((item, idx) => ({
      kind: "indirect",
      idx,
      id: item.template_id,
      category: "indirect_protocol",
      detail: `${item.protocol || "native"} (${item.paired !== false ? "paired" : "single"})`,
    }));
    return [...b, ...ind];
  }

  function renderQueue() {
    const items = combinedQueue();
    const pg = paginate(items, queuePage, pageSize);
    queuePage = pg.page;
    const tbody = document.getElementById("cfg-queue-body");
    tbody.innerHTML = pg.slice.map((row) => `
      <tr>
        <td><span class="badge ${row.kind === "benign" ? "badge-info" : "badge-warn"}">${esc(row.kind)}</span></td>
        <td><code>${esc(row.id)}</code></td>
        <td><code>${esc(row.category)}</code></td>
        <td class="card-sub">${esc(row.detail)}</td>
        <td><button class="btn btn-ghost btn-sm cfg-q-del" data-kind="${esc(row.kind)}" data-idx="${row.idx}" type="button">Remove</button></td>
      </tr>
    `).join("") || `<tr><td colspan="5" class="card-sub">Queue is empty — clicking Run will execute the Suite Mode shortcut below.</td></tr>`;

    tbody.querySelectorAll(".cfg-q-del").forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = Number(btn.dataset.idx);
        if (btn.dataset.kind === "benign") appState.queuedBenign.splice(idx, 1);
        else appState.queuedIndirect.splice(idx, 1);
        renderQueue();
      });
    });

    bindPager(document.getElementById("cfg-queue-pager"), pg.totalItems, pg.page, pageSize, (p) => {
      queuePage = p;
      renderQueue();
    });
  }

  function renderStream() {
    const pg = paginate(streamRows, streamPage, pageSize);
    streamPage = pg.page;
    const tbody = document.getElementById("run-stream-body");
    tbody.innerHTML = pg.slice.map((r) => `
      <tr>
        <td><code>${esc(r.agent_key)}</code></td>
        <td><code>${esc(r.test_case_id)}</code></td>
        <td><code>${esc(r.category)}</code></td>
        <td>${outcomeBadge(r.outcome)}</td>
        <td><code>${Number(r.confidence || 0).toFixed(2)}</code></td>
        <td><code>${r.latency_ms != null ? `${r.latency_ms} ms` : "—"}</code></td>
        <td>
          <div><code>${esc((r.signals || []).join(", ") || "none")}</code></div>
          <div class="card-sub">${esc(r.response_preview || "")}</div>
        </td>
      </tr>
    `).join("") || `<tr><td colspan="7" class="card-sub">No evaluation results streamed yet.</td></tr>`;

    bindPager(document.getElementById("run-stream-pager"), pg.totalItems, pg.page, pageSize, (p) => {
      streamPage = p;
      renderStream();
    });
  }

  document.getElementById("cfg-cat").addEventListener("change", populateTemplates);
  document.getElementById("cfg-tpl").addEventListener("change", renderParamInputs);
  document.getElementById("cfg-case-id").addEventListener("input", updatePreview);

  document.getElementById("btn-cfg-add").addEventListener("click", () => {
    if (!latestPreview) return;
    appState.queuedBenign.push(latestPreview);
    renderQueue();
    toast(`Queued ${latestPreview.test_case_id}.`, "ok");
  });

  document.getElementById("btn-cfg-load-benign").addEventListener("click", async () => {
    try {
      const rendered = await Promise.all(
        templates.map((t) =>
          apiPost("/api/templates/benign/render", {
            template_id: t.template_id,
            test_case_id: `default-${t.template_id}`,
          })
        )
      );
      appState.queuedBenign = rendered;
      queuePage = 1;
      renderQueue();
      toast(`Loaded ${rendered.length} default benign validation cases.`, "ok");
    } catch (err) {
      toast(err.message, "err");
    }
  });

  document.getElementById("btn-cfg-load-indirect").addEventListener("click", () => {
    const proto = document.getElementById("run-protocol").value || null;
    appState.queuedIndirect = indirectTemplates.map((t) => ({
      template_id: t.template_id,
      protocol: proto,
      paired: true,
    }));
    queuePage = 1;
    renderQueue();
    toast(`Loaded ${appState.queuedIndirect.length} paired indirect templates.`, "ok");
  });

  document.getElementById("btn-cfg-clear").addEventListener("click", () => {
    appState.queuedBenign = [];
    appState.queuedIndirect = [];
    renderQueue();
  });

  document.getElementById("btn-launch-eval").addEventListener("click", async () => {
    const agentKeys = Array.from(document.querySelectorAll(".run-agent-chk:checked")).map((c) => c.value);
    if (!agentKeys.length) {
      toast("Select at least one target agent.", "err");
      return;
    }

    const body = {
      agent_keys: agentKeys,
      benign_cases: appState.queuedBenign.map((c) => ({
        template_id: c.template_id,
        parameters: c.parameters,
        test_case_id: c.test_case_id,
      })),
      indirect_cases: appState.queuedIndirect,
      suite_mode: document.getElementById("run-suite-mode").value,
      protocol: document.getElementById("run-protocol").value || null,
      paired: document.getElementById("run-paired").checked,
      scoring_engine: document.getElementById("run-scoring").value,
      drift_threshold: Number(document.getElementById("run-drift-tol").value),
      judge_agent_key: document.getElementById("run-judge-agent").value,
      attach_mcp: document.getElementById("run-attach-mcp").checked,
      use_embeddings: document.getElementById("run-use-embed").checked,
      redact: document.getElementById("run-redact").checked,
    };

    try {
      const btn = document.getElementById("btn-launch-eval");
      btn.disabled = true;
      streamRows = [];
      streamPage = 1;
      renderStream();
      document.getElementById("run-progress-fill").style.width = "4%";
      document.getElementById("run-progress-label").textContent = "Starting evaluation job...";

      const start = await apiPost("/api/eval/run", body);
      toast(`Started evaluation job ${start.job_id} (${start.total_cases} case(s)).`, "info");

      connectJobStream(start.job_id, async (ev) => {
        if (ev.event === "case_complete") {
          streamRows.unshift(ev);
          const pct = Math.min(100, Math.round((ev.completed / Math.max(1, ev.total)) * 100));
          document.getElementById("run-progress-fill").style.width = `${pct}%`;
          document.getElementById("run-progress-label").textContent =
            `[${ev.agent_key}] ${ev.completed}/${ev.total} (${pct}%): ${ev.test_case_id}`;
          renderStream();
        } else if (ev.event === "job_complete") {
          btn.disabled = false;
          document.getElementById("run-progress-fill").style.width = "100%";
          const runCount = ev.result?.runs?.length || agentKeys.length;
          document.getElementById("run-progress-label").textContent =
            `Completed ${ev.result?.total_completed || streamRows.length} evaluations across ${runCount} run(s). Saved to eval_runs/.`;
          toast(`Evaluation job complete! ${runCount} JSONL file(s) saved.`, "ok");
          await appState.refreshStatus();
        } else if (ev.event === "job_failed") {
          btn.disabled = false;
          document.getElementById("run-progress-label").textContent = `Job failed: ${ev.error}`;
          toast(`Evaluation failed: ${ev.error}`, "err");
        }
      });
    } catch (err) {
      document.getElementById("btn-launch-eval").disabled = false;
      toast(err.message, "err");
    }
  });

  await populateTemplates();
  renderQueue();
  renderStream();
}

export async function renderRelayLabPage(root) {
  const [scenRes, tracesRes, agentsRes] = await Promise.all([
    apiGet("/api/relay/scenarios"),
    apiGet("/api/relay/traces"),
    apiGet("/api/agents"),
  ]);

  const scenarios = scenRes.scenarios || [];
  const protocols = scenRes.protocols || [];
  const agents = agentsRes.agents || [];
  let traces = tracesRes.traces || [];
  let tracePage = 1;
  const pageSize = 10;

  root.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div>
          <h2 class="card-title">1. Live 2-Hop Cross-Model Relay Runner (Agent A &rarr; Gateway &rarr; Agent B)</h2>
          <p class="card-sub">Measure origin fact retention, downstream schema fidelity, and perturbation containment across protocol boundaries.</p>
        </div>
      </div>
      <div class="grid grid-3">
        <label class="field">Relay Scenario
          <select id="rel-scenario">
            ${scenarios.map((s) => `<option value="${esc(s.scenario_id)}">${esc(s.scenario_id)} &mdash; ${esc(s.name)}</option>`).join("")}
          </select>
        </label>
        <label class="field">Upstream Agent A (Hop 1)
          <select id="rel-agent-a">
            ${agents.map((a, i) => `<option value="${esc(a.agent_key)}" ${i === 0 ? "selected" : ""}>${esc(a.agent_key)}</option>`).join("")}
          </select>
        </label>
        <label class="field">Downstream Agent B (Hop 2)
          <select id="rel-agent-b">
            ${agents.map((a, i) => `<option value="${esc(a.agent_key)}" ${i === Math.min(1, agents.length - 1) ? "selected" : ""}>${esc(a.agent_key)}</option>`).join("")}
          </select>
        </label>
      </div>
      <div class="grid grid-2 mt-12">
        <label class="field">Hop-1 Source Protocol
          <select id="rel-src-proto">
            ${protocols.map((p, i) => `<option value="${esc(p)}" ${i === 0 ? "selected" : ""}>${esc(p)}</option>`).join("")}
          </select>
        </label>
        <label class="field">Hop-2 Target Protocol
          <select id="rel-tgt-proto">
            ${protocols.map((p, i) => `<option value="${esc(p)}" ${i === 1 ? "selected" : ""}>${esc(p)}</option>`).join("")}
          </select>
        </label>
      </div>
      <div class="btn-row mt-12">
        <label class="check-row"><input type="checkbox" id="rel-ablation" checked /> Run A/B Ablation (Raw Passthrough vs. Hardened Gateway)</label>
        <label class="check-row"><input type="checkbox" id="rel-san-ctrl" checked /> Sanitize Control Tokens</label>
        <label class="check-row"><input type="checkbox" id="rel-esc-xml" checked /> Escape XML Delimiters</label>
        <label class="check-row"><input type="checkbox" id="rel-wrap-json" checked /> Wrap Provenance JSON Envelope</label>
      </div>
      <div class="card-sub mt-12" id="rel-scenario-desc"></div>
      <div class="btn-row mt-12">
        <button class="btn btn-primary" id="btn-rel-run" type="button">Execute 2-Hop Relay</button>
      </div>
      <div id="rel-live-out" class="mt-16"></div>
    </div>

    <div class="card mt-16">
      <div class="card-header">
        <div>
          <h2 class="card-title">2. Persisted Relay Traces (eval_runs/relays/*.json)</h2>
          <p class="card-sub">Click any persisted trace to dissect Hop 1, Gateway wire transform, and Hop 2 outputs.</p>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Relay ID</th>
              <th>Scenario</th>
              <th>Handoff</th>
              <th>Policy</th>
              <th>Status</th>
              <th>Fact Ret.</th>
              <th>Schema</th>
              <th>Latency</th>
            </tr>
          </thead>
          <tbody id="rel-traces-body"></tbody>
        </table>
      </div>
      <div class="pager" id="rel-traces-pager"></div>
    </div>
  `;

  function updateScenarioDesc() {
    const sid = document.getElementById("rel-scenario").value;
    const sc = scenarios.find((s) => s.scenario_id === sid);
    if (!sc) return;
    document.getElementById("rel-scenario-desc").innerHTML =
      `${esc(sc.description)} &middot; Required origin facts: <code>${esc(sc.required_origin_facts.join(", "))}</code> &middot; Expected format: <code>${esc(sc.expected_downstream_format)}</code>`;
  }

  function renderRelayCard(r) {
    return `
      <div class="card mt-12">
        <div class="card-header">
          <div>
            <h3 class="card-title">${esc(r.scenario_id)} &mdash; <code>${esc(r.defense_policy_label || (r.defense_policy?.envelope_wrap_json ? "hardened" : "raw_passthrough"))}</code></h3>
            <p class="card-sub">${esc(r.hop_1.agent_key)} (${esc(r.source_protocol)}) &rarr; ${esc(r.hop_2.agent_key)} (${esc(r.target_protocol)})</p>
          </div>
          ${outcomeBadge(r.relay_status)}
        </div>
        <div class="grid grid-4">
          <div class="kpi"><div class="kpi-label">Fact Retention</div><div class="kpi-value">${(Number(r.origin_fact_retention) * 100).toFixed(0)}%</div></div>
          <div class="kpi"><div class="kpi-label">Downstream Schema</div><div class="kpi-value">${r.downstream_schema_valid ? `<span class="badge badge-ok">VALID</span>` : `<span class="badge badge-bad">INVALID</span>`}</div></div>
          <div class="kpi"><div class="kpi-label">Perturbation Propagated</div><div class="kpi-value">${r.perturbation_propagated ? `<span class="badge badge-bad">YES</span>` : `<span class="badge badge-ok">NO</span>`}</div></div>
          <div class="kpi"><div class="kpi-label">Total Latency</div><div class="kpi-value">${esc(r.total_latency_ms)} ms</div></div>
        </div>
        <div class="grid grid-3 mt-12">
          <div><div class="card-sub mb-12"><strong>Hop 1 Output</strong></div><pre class="code-box">${esc(r.hop_1.output_text)}</pre></div>
          <div><div class="card-sub mb-12"><strong>Gateway Wire Payload</strong></div><pre class="code-box">${esc(r.gateway_payload)}</pre></div>
          <div><div class="card-sub mb-12"><strong>Hop 2 Output</strong></div><pre class="code-box">${esc(r.hop_2.output_text)}</pre></div>
        </div>
      </div>
    `;
  }

  function renderTracesTable() {
    const pg = paginate(traces, tracePage, pageSize);
    tracePage = pg.page;
    const tbody = document.getElementById("rel-traces-body");
    tbody.innerHTML = pg.slice.map((t, i) => {
      const pol = t.defense_policy || {};
      const polLabel = (pol.sanitize_control_tokens || pol.escape_xml_delimiters || pol.envelope_wrap_json)
        ? "hardened_gateway"
        : "raw_passthrough";
      return `
        <tr class="clickable" data-ridx="${(pg.page - 1) * pageSize + i}">
          <td><code>${esc(t.relay_id)}</code></td>
          <td><code>${esc(t.scenario_id)}</code></td>
          <td class="card-sub">${esc(t.hop_1?.agent_key)} &rarr; ${esc(t.hop_2?.agent_key)}</td>
          <td><code>${esc(polLabel)}</code></td>
          <td>${outcomeBadge(t.relay_status)}</td>
          <td><code>${(Number(t.origin_fact_retention) * 100).toFixed(0)}%</code></td>
          <td>${t.downstream_schema_valid ? `<span class="badge badge-ok">YES</span>` : `<span class="badge badge-bad">NO</span>`}</td>
          <td><code>${esc(t.total_latency_ms)} ms</code></td>
        </tr>
      `;
    }).join("") || `<tr><td colspan="8" class="card-sub">No persisted relay traces found.</td></tr>`;

    tbody.querySelectorAll("tr[data-ridx]").forEach((tr) => {
      tr.addEventListener("click", () => {
        const item = traces[Number(tr.dataset.ridx)];
        if (item) openModal(`Relay Trace ${item.relay_id}`, renderRelayCard(item));
      });
    });

    bindPager(document.getElementById("rel-traces-pager"), pg.totalItems, pg.page, pageSize, (p) => {
      tracePage = p;
      renderTracesTable();
    });
  }

  document.getElementById("rel-scenario").addEventListener("change", updateScenarioDesc);
  document.getElementById("btn-rel-run").addEventListener("click", async () => {
    try {
      const res = await apiPost("/api/relay/run", {
        scenario_id: document.getElementById("rel-scenario").value,
        agent_a_key: document.getElementById("rel-agent-a").value,
        agent_b_key: document.getElementById("rel-agent-b").value,
        source_protocol: document.getElementById("rel-src-proto").value,
        target_protocol: document.getElementById("rel-tgt-proto").value,
        ablation: document.getElementById("rel-ablation").checked,
        defense_policy: {
          sanitize_control_tokens: document.getElementById("rel-san-ctrl").checked,
          escape_xml_delimiters: document.getElementById("rel-esc-xml").checked,
          envelope_wrap_json: document.getElementById("rel-wrap-json").checked,
        },
      });
      const results = res.results || [];
      document.getElementById("rel-live-out").innerHTML = results.map(renderRelayCard).join("");
      const updated = await apiGet("/api/relay/traces");
      traces = updated.traces || [];
      tracePage = 1;
      renderTracesTable();
      toast(`Executed ${results.length} relay trace(s).`, "ok");
    } catch (err) {
      toast(err.message, "err");
    }
  });

  updateScenarioDesc();
  renderTracesTable();
}

export async function renderHamsterWheelPage(root, appState) {
  const [reportsRes, agentsRes] = await Promise.all([
    apiGet("/api/wheel/reports"),
    apiGet("/api/agents"),
  ]);

  const agents = agentsRes.agents || [];
  let reports = reportsRes.reports || [];
  let reportPage = 1;
  let streamPage = 1;
  const pageSize = 8;
  let streamEvents = [];
  let selectedReport = reports[0] || null;

  root.innerHTML = `
    <div class="card">
      <div class="card-header">
        <div>
          <h2 class="card-title">1. Model Hamster Wheel Continuous Admission Gauntlet</h2>
          <p class="card-sub">Continuously cycle target models through paired indirect probes, benign validation templates, protocol translations, relay ablations, and admission gates.</p>
        </div>
      </div>
      <div class="grid grid-4">
        <label class="field">Target Agent
          <select id="wh-agent">
            ${agents.map((a) => `<option value="${esc(a.agent_key)}">${esc(a.agent_key)} (${esc(a.kind)})</option>`).join("")}
          </select>
        </label>
        <label class="field">Wheel Cycles
          <input type="number" id="wh-cycles" value="1" min="1" max="10" />
        </label>
        <label class="field">Max Failure Rate (ADMIT)
          <input type="number" id="wh-max-fail" value="0.05" step="0.01" min="0" max="1" />
        </label>
        <label class="field">Max Paired Regression (ADMIT)
          <input type="number" id="wh-max-preg" value="0.05" step="0.01" min="0" max="1" />
        </label>
      </div>
      <div class="btn-row mt-12">
        <label class="check-row"><input type="checkbox" id="wh-inc-ind" checked /> Paired Indirect Suite</label>
        <label class="check-row"><input type="checkbox" id="wh-inc-ben" checked /> Benign Validation Suite</label>
        <label class="check-row"><input type="checkbox" id="wh-all-proto" /> Sweep All 5 Wire Protocols</label>
        <label class="check-row"><input type="checkbox" id="wh-relay" /> Include 2-Hop Relay Ablation</label>
        <label class="check-row"><input type="checkbox" id="wh-mcp" checked /> Attach MCP Tools</label>
        <label class="check-row"><input type="checkbox" id="wh-embed" /> Ollama Embeddings</label>
        <label class="check-row"><input type="checkbox" id="wh-stop" /> Stop on Gate Fail</label>
      </div>
      <div class="btn-row mt-12">
        <button class="btn btn-primary" id="btn-wh-start" type="button">Enter Hamster Wheel</button>
        <span class="card-sub" id="wh-status-text">Idle.</span>
      </div>

      <div class="table-wrap mt-12">
        <table>
          <thead>
            <tr><th>Cycle / Proto</th><th>Test Case</th><th>Category</th><th>Outcome</th><th>Tools</th><th>Latency</th><th>Signals</th></tr>
          </thead>
          <tbody id="wh-stream-body"></tbody>
        </table>
      </div>
      <div class="pager" id="wh-stream-pager"></div>
    </div>

    <div class="card mt-16">
      <div class="card-header">
        <div>
          <h2 class="card-title">2. Active Admission Report Inspector</h2>
          <p class="card-sub">Per-channel breakdown, cycle consistency, McNemar p-value, and OWASP remediation.</p>
        </div>
      </div>
      <div id="wh-report-detail"></div>
    </div>

    <div class="card mt-16">
      <div class="card-header">
        <div>
          <h2 class="card-title">3. Saved Wheel Admission Reports (eval_runs/wheel/*.json)</h2>
          <p class="card-sub">Click any row to inspect its cycle summaries and channel breakdown above.</p>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Wheel ID</th>
              <th>Agent</th>
              <th>Verdict</th>
              <th>Safe Rate</th>
              <th>Failure Rate</th>
              <th>Paired Reg.</th>
              <th>Cycles</th>
              <th>Cases</th>
            </tr>
          </thead>
          <tbody id="wh-reports-body"></tbody>
        </table>
      </div>
      <div class="pager" id="wh-reports-pager"></div>
    </div>
  `;

  function renderReportDetail(rep) {
    const el = document.getElementById("wh-report-detail");
    if (!rep) {
      el.innerHTML = `<div class="card-sub">No Hamster Wheel report selected yet. Launch one above.</div>`;
      return;
    }
    const chBars = (rep.channel_breakdown || []).map((b) => ({
      label: b.channel,
      value: Number(((b.safe_rate || 0) * 100).toFixed(1)),
      color: b.failure_rate === 0 ? "#10b981" : b.failure_rate <= 0.15 ? "#f59e0b" : "#ef4444",
    }));
    el.innerHTML = `
      <div class="grid grid-5">
        <div class="kpi"><div class="kpi-label">Admission Verdict</div><div class="kpi-value">${outcomeBadge(rep.admission_verdict)}</div><div class="kpi-note">${esc(rep.agent_key)}</div></div>
        <div class="kpi"><div class="kpi-label">Overall Safe Rate</div><div class="kpi-value">${(rep.overall_safe_rate * 100).toFixed(1)}%</div><div class="kpi-note">${rep.total_cases_executed} cases</div></div>
        <div class="kpi"><div class="kpi-label">Failure Rate</div><div class="kpi-value">${(rep.overall_failure_rate * 100).toFixed(1)}%</div><div class="kpi-note">McNemar p: ${rep.overall_mcnemar_p ?? "—"}</div></div>
        <div class="kpi"><div class="kpi-label">Paired Regression</div><div class="kpi-value">${rep.overall_paired_regression_rate != null ? `${(rep.overall_paired_regression_rate * 100).toFixed(1)}%` : "—"}</div><div class="kpi-note">Consistency: ${rep.consistency_rate ?? "—"}</div></div>
        <div class="kpi"><div class="kpi-label">MCP Calls / Latency</div><div class="kpi-value">${rep.total_mcp_tool_calls}</div><div class="kpi-note">${rep.total_latency_ms} ms total</div></div>
      </div>
      <div class="grid grid-2 mt-16">
        <div>
          <div class="card-sub mb-12"><strong>Channel Safe Rate (%)</strong></div>
          ${renderBarChartSvg(chBars, { height: 190, valueFmt: (v) => `${v}%` })}
        </div>
        <div>
          <div class="card-sub mb-12"><strong>Cycle Summaries</strong></div>
          <div class="table-wrap">
            <table>
              <thead><tr><th>Cycle</th><th>Protocol</th><th>Safe</th><th>Fail</th><th>Paired Reg.</th><th>Signals</th></tr></thead>
              <tbody>
                ${(rep.cycle_summaries || []).map((cs) => `
                  <tr>
                    <td><code>#${cs.cycle_index}</code></td>
                    <td><code>${esc(cs.protocol)}</code></td>
                    <td><code>${(cs.safe_rate * 100).toFixed(1)}%</code></td>
                    <td><code>${(cs.failure_rate * 100).toFixed(1)}%</code></td>
                    <td><code>${cs.paired_regression_rate ?? "—"}</code></td>
                    <td class="card-sub">${esc((cs.top_failure_signals || []).join(", ") || "None")}</td>
                  </tr>`).join("")}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    `;
  }

  function renderStream() {
    const pg = paginate(streamEvents, streamPage, pageSize);
    streamPage = pg.page;
    document.getElementById("wh-stream-body").innerHTML = pg.slice.map((ev) => `
      <tr>
        <td><code>C${ev.cycle_index}/${esc(ev.protocol)}</code></td>
        <td><code>${esc(ev.test_case_id)}</code></td>
        <td><code>${esc(ev.category)}</code></td>
        <td>${outcomeBadge(ev.outcome)}</td>
        <td><code>${ev.tool_calls ?? 0}</code></td>
        <td><code>${ev.latency_ms != null ? `${ev.latency_ms} ms` : "—"}</code></td>
        <td class="card-sub">${esc((ev.signals || []).join(", "))}</td>
      </tr>
    `).join("") || `<tr><td colspan="7" class="card-sub">No live wheel telemetry yet.</td></tr>`;

    bindPager(document.getElementById("wh-stream-pager"), pg.totalItems, pg.page, pageSize, (p) => {
      streamPage = p;
      renderStream();
    });
  }

  function renderReportsTable() {
    const pg = paginate(reports, reportPage, pageSize);
    reportPage = pg.page;
    const tbody = document.getElementById("wh-reports-body");
    tbody.innerHTML = pg.slice.map((r) => `
      <tr class="clickable ${selectedReport?.wheel_id === r.wheel_id ? "selected" : ""}" data-wid="${esc(r.wheel_id)}">
        <td><code>${esc(r.wheel_id)}</code></td>
        <td><code>${esc(r.agent_key)}</code></td>
        <td>${outcomeBadge(r.admission_verdict)}</td>
        <td><code>${(r.overall_safe_rate * 100).toFixed(1)}%</code></td>
        <td><code>${(r.overall_failure_rate * 100).toFixed(1)}%</code></td>
        <td><code>${r.overall_paired_regression_rate ?? "—"}</code></td>
        <td><code>${r.cycles_completed}</code></td>
        <td><code>${r.total_cases_executed}</code></td>
      </tr>
    `).join("") || `<tr><td colspan="8" class="card-sub">No saved wheel reports found.</td></tr>`;

    tbody.querySelectorAll("tr[data-wid]").forEach((tr) => {
      tr.addEventListener("click", () => {
        selectedReport = reports.find((x) => x.wheel_id === tr.dataset.wid) || null;
        renderReportsTable();
        renderReportDetail(selectedReport);
      });
    });

    bindPager(document.getElementById("wh-reports-pager"), pg.totalItems, pg.page, pageSize, (p) => {
      reportPage = p;
      renderReportsTable();
    });
  }

  document.getElementById("btn-wh-start").addEventListener("click", async () => {
    const btn = document.getElementById("btn-wh-start");
    try {
      btn.disabled = true;
      streamEvents = [];
      streamPage = 1;
      renderStream();
      document.getElementById("wh-status-text").textContent = "Launching Hamster Wheel job...";

      const start = await apiPost("/api/wheel/start", {
        agent_key: document.getElementById("wh-agent").value,
        cycles: Number(document.getElementById("wh-cycles").value),
        max_failure_rate_for_admit: Number(document.getElementById("wh-max-fail").value),
        max_paired_regression_for_admit: Number(document.getElementById("wh-max-preg").value),
        include_indirect_paired: document.getElementById("wh-inc-ind").checked,
        include_benign_suite: document.getElementById("wh-inc-ben").checked,
        all_protocols: document.getElementById("wh-all-proto").checked,
        include_relay_ablation: document.getElementById("wh-relay").checked,
        attach_mcp: document.getElementById("wh-mcp").checked,
        use_embeddings: document.getElementById("wh-embed").checked,
        stop_on_gate_fail: document.getElementById("wh-stop").checked,
      });

      connectJobStream(start.job_id, async (ev) => {
        if (ev.event === "cycle_start") {
          document.getElementById("wh-status-text").textContent =
            `Revolution ${ev.revolution}/${ev.total_revolutions} (${ev.protocol}) — ${ev.cases_in_cycle} cases`;
        } else if (ev.event === "case_complete") {
          streamEvents.unshift(ev);
          document.getElementById("wh-status-text").textContent =
            `Rev ${ev.revolution}/${ev.total_revolutions}: ${ev.case_index}/${ev.cases_in_cycle} (${ev.test_case_id})`;
          renderStream();
        } else if (ev.event === "job_complete") {
          btn.disabled = false;
          selectedReport = ev.result;
          document.getElementById("wh-status-text").textContent =
            `Finished: Verdict ${selectedReport.admission_verdict} (${selectedReport.total_cases_executed} cases)`;
          renderReportDetail(selectedReport);
          const refreshed = await apiGet("/api/wheel/reports");
          reports = refreshed.reports || [];
          reportPage = 1;
          renderReportsTable();
          await appState.refreshStatus();
          toast(`Hamster Wheel verdict: ${selectedReport.admission_verdict}`, selectedReport.admission_verdict === "ADMIT" ? "ok" : "err");
        } else if (ev.event === "job_failed") {
          btn.disabled = false;
          document.getElementById("wh-status-text").textContent = `Failed: ${ev.error}`;
          toast(ev.error, "err");
        }
      });
    } catch (err) {
      btn.disabled = false;
      toast(err.message, "err");
    }
  });

  renderReportDetail(selectedReport);
  renderReportsTable();
  renderStream();
}
