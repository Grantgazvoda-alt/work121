export function esc(val) {
  return String(val ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export async function apiGet(path) {
  const res = await fetch(path, { headers: { Accept: "application/json" } });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status} on ${path}`);
  return data;
}

export async function apiPost(path, body = {}) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status} on ${path}`);
  return data;
}

export async function apiDelete(path) {
  const res = await fetch(path, { method: "DELETE" });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status} on ${path}`);
  return data;
}

export function toast(message, kind = "info") {
  const stack = document.getElementById("toast-stack");
  if (!stack) return;
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = message;
  stack.appendChild(el);
  setTimeout(() => el.remove(), 4800);
}

export function openModal(title, html) {
  document.getElementById("modal-title").textContent = title;
  document.getElementById("modal-body").innerHTML = html;
  document.getElementById("modal-backdrop").classList.remove("hidden");
}

export function closeModal() {
  document.getElementById("modal-backdrop").classList.add("hidden");
}

export function outcomeBadge(outcome) {
  const v = String(outcome || "indeterminate");
  if (v === "safe_redirect" || v === "safe_refusal" || v === "ADMIT" || v === "PASS_CLEAN" || v === "native") {
    return `<span class="badge badge-ok">${esc(v)}</span>`;
  }
  if (v === "partial_erosion" || v === "CONDITIONAL_REVIEW" || v === "PARTIAL_DRIFT" || v === "preserved_with_adaptation" || v === "MODERATE") {
    return `<span class="badge badge-warn">${esc(v)}</span>`;
  }
  if (v === "boundary_failure" || v === "BLOCK" || v === "PERTURBATION_PROPAGATED" || v === "SCHEMA_OR_FACT_FAILURE" || v === "degraded_privilege_boundary" || v === "lossy_modality_flattening" || v === "HIGH" || v === "CRITICAL") {
    return `<span class="badge badge-bad">${esc(v)}</span>`;
  }
  return `<span class="badge badge-muted">${esc(v)}</span>`;
}

export function connectJobStream(jobId, handlerOrOpts) {
  const callbacks =
    typeof handlerOrOpts === "function"
      ? { onEvent: handlerOrOpts, onComplete: null }
      : {
          onEvent: typeof handlerOrOpts?.onEvent === "function" ? handlerOrOpts.onEvent : () => {},
          onComplete: typeof handlerOrOpts?.onComplete === "function" ? handlerOrOpts.onComplete : null,
        };

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${proto}//${location.host}/ws/jobs/${encodeURIComponent(jobId)}`;
  const pill = document.getElementById("active-job-pill");
  const label = document.getElementById("active-job-label");
  if (pill && label) {
    pill.classList.remove("hidden");
    label.textContent = `Job ${jobId}`;
  }

  let closedByTerminal = false;
  let seenCount = 0;
  let pollTimer = null;

  function emitEvent(ev) {
    callbacks.onEvent(ev);
    if (ev.event === "job_complete" || ev.event === "job_failed") {
      closedByTerminal = true;
      if (pill) pill.classList.add("hidden");
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
      if (callbacks.onComplete) callbacks.onComplete(ev);
    }
  }

  function startFallbackPolling() {
    if (closedByTerminal || pollTimer) return;
    const pollOnce = async () => {
      if (closedByTerminal) return;
      try {
        const job = await apiGet(`/api/jobs/${encodeURIComponent(jobId)}`);
        const evs = job.events || [];
        for (; seenCount < evs.length; seenCount++) {
          emitEvent(evs[seenCount]);
        }
        if (!closedByTerminal && (job.status === "completed" || job.status === "failed")) {
          emitEvent(
            job.status === "completed"
              ? { event: "job_complete", status: "completed", result: job.result }
              : { event: "job_failed", status: "failed", error: job.error }
          );
        }
      } catch (_) {}
    };
    pollOnce();
    pollTimer = setInterval(pollOnce, 600);
  }

  const ws = new WebSocket(wsUrl);
  ws.onmessage = (msg) => {
    try {
      const ev = JSON.parse(msg.data);
      if (ev.event === "snapshot") {
        const snapEvents = ev.events || [];
        seenCount = snapEvents.length;
        snapEvents.forEach((item) => callbacks.onEvent(item));
        if (ev.status === "completed") {
          emitEvent({ event: "job_complete", status: "completed", result: ev.result });
        } else if (ev.status === "failed") {
          emitEvent({ event: "job_failed", status: "failed", error: ev.error });
        }
      } else {
        seenCount += 1;
        emitEvent(ev);
      }
    } catch (_) {}
  };
  ws.onerror = () => startFallbackPolling();
  ws.onclose = () => {
    if (!closedByTerminal) {
      startFallbackPolling();
    } else if (pill) {
      pill.classList.add("hidden");
    }
  };
  return ws;
}

export function paginate(items, page, pageSize = 12) {
  const list = Array.isArray(items) ? items : [];
  const totalPages = Math.max(1, Math.ceil(list.length / pageSize));
  const safePage = Math.min(Math.max(1, page), totalPages);
  const slice = list.slice((safePage - 1) * pageSize, safePage * pageSize);
  return { slice, page: safePage, totalPages, totalItems: list.length };
}

export function bindPager(el, totalItems, page, pageSize, onChange) {
  if (!el) return;
  const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));
  const cur = Math.min(Math.max(1, page), totalPages);
  const start = totalItems === 0 ? 0 : (cur - 1) * pageSize + 1;
  const end = Math.min(totalItems, cur * pageSize);
  el.innerHTML = `
    <span>Showing ${start}&ndash;${end} of ${totalItems} (Page ${cur}/${totalPages})</span>
    <div class="btn-row">
      <button class="btn btn-ghost btn-sm" data-pager="first" ${cur <= 1 ? "disabled" : ""} type="button">&laquo;</button>
      <button class="btn btn-ghost btn-sm" data-pager="prev" ${cur <= 1 ? "disabled" : ""} type="button">Prev</button>
      <button class="btn btn-ghost btn-sm" data-pager="next" ${cur >= totalPages ? "disabled" : ""} type="button">Next</button>
      <button class="btn btn-ghost btn-sm" data-pager="last" ${cur >= totalPages ? "disabled" : ""} type="button">&raquo;</button>
    </div>
  `;
  el.querySelector('[data-pager="first"]')?.addEventListener("click", () => onChange(1));
  el.querySelector('[data-pager="prev"]')?.addEventListener("click", () => onChange(cur - 1));
  el.querySelector('[data-pager="next"]')?.addEventListener("click", () => onChange(cur + 1));
  el.querySelector('[data-pager="last"]')?.addEventListener("click", () => onChange(totalPages));
}
