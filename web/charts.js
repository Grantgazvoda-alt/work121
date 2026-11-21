import { esc } from "./api.js";

export function renderBarChartSvg(items, { height = 210, labelWidth = 170, barColor = "#38bdf8", valueFmt = (v) => String(v) } = {}) {
  if (!items || !items.length) return `<div class="card-sub">No data to plot.</div>`;
  const maxVal = Math.max(...items.map((d) => Math.abs(Number(d.value) || 0)), 1e-6);
  const rowH = Math.max(24, Math.floor((height - 20) / items.length));
  const svgH = items.length * rowH + 16;
  const svgW = 620;
  const plotW = svgW - labelWidth - 80;

  const rows = items.map((d, i) => {
    const val = Number(d.value) || 0;
    const w = Math.max(2, (Math.abs(val) / maxVal) * plotW);
    const y = 10 + i * rowH;
    const fill = d.color || barColor;
    return `
      <g>
        <text x="${labelWidth - 8}" y="${y + 15}" text-anchor="end" fill="#8fa3bc" font-size="11" font-family="monospace">${esc(String(d.label).slice(0, 26))}</text>
        <rect x="${labelWidth}" y="${y + 3}" width="${w.toFixed(1)}" height="${rowH - 8}" rx="4" fill="${fill}" opacity="0.88">
          <title>${esc(d.label)}: ${esc(valueFmt(val))}</title>
        </rect>
        <text x="${labelWidth + w + 8}" y="${y + 15}" fill="#e6edf5" font-size="11" font-family="monospace">${esc(valueFmt(val))}</text>
      </g>`;
  }).join("");

  return `<svg class="svg-chart" viewBox="0 0 ${svgW} ${svgH}" role="img">${rows}</svg>`;
}

export function renderAlignmentLineSvg(scores) {
  if (!scores || scores.length < 2) return "";
  const w = 540;
  const h = 160;
  const padL = 42;
  const padR = 16;
  const padT = 14;
  const padB = 28;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const pts = scores.map((s, i) => {
    const x = padL + (i / (scores.length - 1)) * plotW;
    const y = padT + (1.0 - Math.max(0, Math.min(1, Number(s)))) * plotH;
    return { x, y, v: Number(s), turn: i + 1 };
  });
  const poly = pts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  const grid = [0.0, 0.5, 1.0].map((g) => {
    const y = padT + (1 - g) * plotH;
    return `<line x1="${padL}" y1="${y}" x2="${w - padR}" y2="${y}" stroke="#223044" stroke-dasharray="3,3" />
            <text x="${padL - 6}" y="${y + 4}" text-anchor="end" fill="#8fa3bc" font-size="10" font-family="monospace">${g.toFixed(1)}</text>`;
  }).join("");
  const dots = pts.map((p) => `
    <circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="4.5" fill="#38bdf8" stroke="#080d14" stroke-width="1.5">
      <title>Turn ${p.turn}: ${p.v.toFixed(4)}</title>
    </circle>
    <text x="${p.x.toFixed(1)}" y="${h - 8}" text-anchor="middle" fill="#8fa3bc" font-size="10" font-family="monospace">T${p.turn}</text>
  `).join("");

  return `<svg class="svg-chart" viewBox="0 0 ${w} ${h}">
    ${grid}
    <polyline fill="none" stroke="#38bdf8" stroke-width="2.5" points="${poly}" />
    ${dots}
  </svg>`;
}

export function renderHandoffHeatmapSvg(protocols, matrix) {
  const n = protocols.length;
  const cell = 64;
  const padL = 175;
  const padT = 82;
  const w = padL + n * cell + 20;
  const h = padT + n * cell + 20;
  const lookup = new Map(matrix.map((c) => [`${c.source_protocol}|${c.target_protocol}`, c]));

  const short = (p) => p.replace("_chat_completions", "").replace("_generate_content", "").replace("_messages", "").replace("huggingface_", "hf_").replace("encoder_decoder_", "encdec_");

  const colHeaders = protocols.map((p, j) => {
    const x = padL + j * cell + cell / 2;
    return `<text x="${x}" y="${padT - 12}" text-anchor="start" transform="rotate(-28 ${x} ${padT - 12})" fill="#94a3b8" font-size="11" font-family="monospace">${esc(short(p))}</text>`;
  }).join("");

  let body = "";
  protocols.forEach((src, i) => {
    const y = padT + i * cell;
    body += `<text x="${padL - 10}" y="${y + cell / 2 + 4}" text-anchor="end" fill="#cbd5e1" font-size="11" font-family="monospace">${esc(short(src))}</text>`;
    protocols.forEach((tgt, j) => {
      const x = padL + j * cell;
      const c = lookup.get(`${src}|${tgt}`);
      const sc = c ? Number(c.compatibility_score) : 0;
      const bg = sc >= 0.99 ? "#059669" : sc >= 0.65 ? "#0284c7" : sc >= 0.35 ? "#d97706" : "#b91c1c";
      body += `
        <rect x="${x + 2}" y="${y + 2}" width="${cell - 4}" height="${cell - 4}" rx="6" fill="${bg}" opacity="0.88">
          <title>${esc(src)} -> ${esc(tgt)}: ${sc.toFixed(2)} (${esc(c?.fidelity || "")})</title>
        </rect>
        <text x="${x + cell / 2}" y="${y + cell / 2 + 4}" text-anchor="middle" fill="#ffffff" font-size="12" font-weight="700" font-family="monospace">${sc.toFixed(2)}</text>
      `;
    });
  });

  return `<svg class="svg-chart" viewBox="0 0 ${w} ${h}">${colHeaders}${body}</svg>`;
}

export function renderOutcomeHeatmapSvg(runs) {
  if (!runs || !runs.length) return `<div class="card-sub">No runs loaded.</div>`;
  const testIds = Array.from(new Set(runs.flatMap((r) => (r.outcomes || []).map((o) => o.test_case_id)))).slice(0, 22);
  if (!testIds.length) return `<div class="card-sub">No outcomes available.</div>`;

  const rowRuns = runs.slice(0, 14);
  const cellW = 28;
  const cellH = 26;
  const padL = 220;
  const padT = 110;
  const w = padL + testIds.length * cellW + 20;
  const h = padT + rowRuns.length * cellH + 20;

  const colorMap = {
    safe_redirect: "#10b981",
    safe_refusal: "#059669",
    partial_erosion: "#f59e0b",
    boundary_failure: "#ef4444",
    indeterminate: "#64748b",
  };

  const cols = testIds.map((tid, j) => {
    const x = padL + j * cellW + cellW / 2;
    return `<text x="${x}" y="${padT - 8}" text-anchor="start" transform="rotate(-48 ${x} ${padT - 8})" fill="#8fa3bc" font-size="10" font-family="monospace">${esc(tid.slice(0, 22))}</text>`;
  }).join("");

  const rows = rowRuns.map((run, i) => {
    const y = padT + i * cellH;
    const map = new Map((run.outcomes || []).map((o) => [o.test_case_id, o]));
    const label = `${run.model_id} (${run.model_version})`.slice(0, 30);
    const cells = testIds.map((tid, j) => {
      const x = padL + j * cellW;
      const o = map.get(tid);
      const fill = o ? (colorMap[o.outcome] || "#475569") : "#111827";
      const tip = o ? `${run.run_id} | ${tid}: ${o.outcome}` : `${tid}: not in run`;
      return `<rect x="${x + 2}" y="${y + 2}" width="${cellW - 4}" height="${cellH - 4}" rx="4" fill="${fill}" stroke="#080d14"><title>${esc(tip)}</title></rect>`;
    }).join("");
    return `<text x="${padL - 8}" y="${y + 17}" text-anchor="end" fill="#cbd5e1" font-size="11" font-family="monospace">${esc(label)}</text>${cells}`;
  }).join("");

  return `<svg class="svg-chart" viewBox="0 0 ${w} ${h}">${cols}${rows}</svg>`;
}
