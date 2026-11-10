"use strict";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : Number(x).toFixed(d);
let EXAMPLES = [], LAST = null, CUR_TRIAL = null;

async function api(path, body) {
  const opt = body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)};
  let r;
  try { r = await fetch(path, opt); } catch (e) { throw new Error("Cannot reach the local server. Is start_app running?"); }
  let data = null;
  try { data = await r.json(); } catch (e) { /* non-json */ }
  if (!r.ok) throw new Error((data && data.detail) ? data.detail : `HTTP ${r.status}`);
  return data;
}
function showErr(msg) { const e = $("err"); e.textContent = msg; e.classList.toggle("hidden", !msg); if (msg) e.scrollIntoView({behavior: "smooth", block: "center"}); }

// ---------------------------------------------------------------- tabs & status
document.querySelectorAll(".tab").forEach((b) => b.onclick = () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "exp") loadExperiments();
  if (b.dataset.tab === "hist") loadHistory();
  if (b.dataset.tab === "lab") fillLabCriteria();
});
async function poll() {
  try {
    const h = await api("/api/health");
    const s = $("status");
    s.className = "status " + h.status;
    if (h.status === "ready") s.textContent = `ready · ${h.target}${h.adapter ? " + LoRA" : " (base, no adapter found)"} · draft ${h.draft}`;
    else if (h.status === "error") s.textContent = "model error: " + h.error;
    else { s.textContent = "model: loading…"; setTimeout(poll, 2000); }
  } catch (e) { $("status").textContent = "server unreachable"; setTimeout(poll, 3000); }
}

// ---------------------------------------------------------------- inputs
async function loadExamples() {
  try {
    EXAMPLES = await api("/api/examples");
    $("exampleSel").innerHTML = '<option value="">— choose —</option>' + EXAMPLES.map((e, i) => `<option value="${i}">${esc(e.name)}</option>`).join("");
  } catch (e) { /* examples are optional */ }
}
$("exampleSel").onchange = () => {
  const e = EXAMPLES[$("exampleSel").value]; if (!e) return;
  $("note").value = e.note; CUR_TRIAL = e.trial_id; $("nct").value = e.trial_id || "";
  $("inc").value = e.criteria.filter((c) => c.type === "inclusion").map((c) => c.text.replace(/\n/g, " ")).join("\n");
  $("exc").value = e.criteria.filter((c) => c.type === "exclusion").map((c) => c.text.replace(/\n/g, " ")).join("\n");
  noteHint();
};
function noteHint() {
  const n = $("note").value.trim();
  const lines = n.split(/\n/).filter((l) => l.trim());
  const numbered = lines.length && lines.every((l) => /^\s*\d+\.\s+/.test(l));
  $("noteHint").textContent = n ? `${n.length} characters · ${numbered ? "pre-numbered sentences (ids kept)" : "sentences will be numbered automatically"}` : "";
}
$("note").oninput = noteHint;
function setCriteria(list) {
  $("inc").value = list.filter((c) => c.type === "inclusion").map((c) => c.text).join("\n");
  $("exc").value = list.filter((c) => c.type === "exclusion").map((c) => c.text).join("\n");
}
$("fetchTrial").onclick = async () => {
  const id = $("nct").value.trim(); showErr("");
  $("fetchTrial").disabled = true;
  try { const t = await api("/api/trial/" + encodeURIComponent(id)); setCriteria(t.criteria); CUR_TRIAL = t.nct_id;
        if (!t.criteria.length) showErr("Trial found but its eligibility text could not be split; paste it below."); }
  catch (e) { showErr(e.message); } finally { $("fetchTrial").disabled = false; }
};
$("splitElig").onclick = async () => {
  showErr("");
  try { const r = await api("/api/parse_criteria", {text: $("rawElig").value}); setCriteria(r.criteria); }
  catch (e) { showErr(e.message); }
};
function collectCriteria() {
  const lines = (id, type) => $(id).value.split("\n").map((s) => s.trim()).filter(Boolean).map((text) => ({type, text}));
  return [...lines("inc", "inclusion"), ...lines("exc", "exclusion")];
}

// ---------------------------------------------------------------- screening
const LABEL_CLASS = {"included": ["b-ok", "l-met-inc"], "not excluded": ["b-ok", "l-notexc"], "not included": ["b-bad", "l-blocking"],
  "excluded": ["b-bad", "l-blocking"], "not enough information": ["b-warn", "l-nei"], "not applicable": ["b-na", "l-na"], "invalid": ["b-warn", "l-invalid"]};
$("run").onclick = async () => {
  showErr("");
  const note = $("note").value.trim(), criteria = collectCriteria();
  if (note.length < 20) return showErr("Please enter a patient note (at least one full sentence).");
  if (!criteria.length) return showErr("Please enter at least one inclusion or exclusion criterion (one per line).");
  if (criteria.length > 40) return showErr("At most 40 criteria per run on CPU.");
  const btn = $("run"); btn.disabled = true;
  const t0 = Date.now();
  const timer = setInterval(() => $("runInfo").innerHTML = `<span class="spinner"></span> running ${criteria.length} criteria on CPU… ${((Date.now() - t0) / 1000).toFixed(0)} s (≈2–6 s per criterion)`, 500);
  try {
    LAST = await api("/api/screen", {note, criteria, mode: $("mode").value, k: +$("k").value, trial_id: CUR_TRIAL || ($("nct").value.trim() || null)});
    renderResults(LAST);
    $("runInfo").textContent = `done in ${LAST.total_s} s · saved as report ${LAST.id}`;
  } catch (e) { showErr(e.message); $("runInfo").textContent = ""; }
  finally { clearInterval(timer); btn.disabled = false; }
};
function renderResults(d) {
  $("results").classList.remove("hidden");
  const s = d.summary;
  const lst = (k, t) => s[k].length ? `<h3>${t} (${s[k].length})</h3><ul>${s[k].map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "";
  $("summary").innerHTML = `<div class="type">Preliminary patient–trial outcome (rule-based aggregation, not a ground-truth label)</div>
    <div class="verdict">${esc(s.verdict)}</div><div class="rule">${esc(s.rule)}</div>
    <div class="chips">${Object.entries(s.counts).map(([k, v]) => `<span class="chip">${esc(k)}: ${v}</span>`).join("")}</div>
    ${lst("blocking", "Blocking criteria")}${lst("missing_information", "Missing information")}${lst("needs_verification", "Decisions without cited evidence – verify")}
    <div class="row" style="margin-top:10px">
      <a href="/api/report/${d.id}.html?download=1"><button class="primary">Download report (HTML)</button></a>
      <a href="/api/report/${d.id}.html" target="_blank"><button class="secondary">Open printable report</button></a>
      <a href="/api/report/${d.id}.json"><button class="secondary">Download JSON</button></a>
    </div>
    <div class="meta">model ${esc(d.model.target)} · adapter ${esc(d.model.adapter || "none")} · ${esc(d.model.decoding)} · mode ${esc(d.mode)}${d.model.draft ? " · draft " + esc(d.model.draft) : ""} · prefix prefill ${d.prefix_prefill_s}s · total ${d.total_s}s</div>`;
  $("critList").innerHTML = d.results.map((r, i) => {
    const [b, l] = LABEL_CLASS[r.label] || ["b-na", "l-na"];
    const ev = r.evidence_text.length ? r.evidence_text.map((e) => `<div><b>[S${e.id}]</b> ${esc(e.text)}</div>`).join("") : "<i>no sentence cited</i>";
    const t = r.timing;
    const extra = t.acceptance_rate !== undefined ? ` · acceptance ${fmt(t.acceptance_rate, 2)} (${t.accepted}/${t.drafted})` : "";
    return `<div class="crit ${l}" data-i="${i}">
      <div class="top"><div><div class="type">${esc(r.criterion_type)} #${i + 1}</div>${esc(r.criterion)}</div><span class="badge ${b}">${esc(r.label)}</span></div>
      <div class="ev">${ev}</div>
      <div class="expl">${esc(r.explanation)}</div>
      ${r.hallucinated_ids.length ? `<div class="flag">Model cited non-existent sentence ids ${r.hallucinated_ids.join(", ")} – discarded.</div>` : ""}
      ${r.truncated_sentences.length ? `<div class="flag">Note truncated: sentences ${r.truncated_sentences.join(", ")} were not seen by the model.</div>` : ""}
      <div class="meta">label source: ${esc(r.label_strategy || "generated")}${r.generated_label && r.generated_label !== r.label ? ` (generated JSON said "${esc(r.generated_label)}")` : ""}<br>raw: <span class="raw">${esc(r.raw_output)}</span><br>${t.prompt_tokens} prompt tok · ${t.new_tokens} new tok · prefill ${fmt(t.prefill_s)}s · decode ${fmt(t.decode_s)}s · ${fmt(t.tokens_per_s, 1)} tok/s${extra}</div>
    </div>`;
  }).join("");
  const cited = new Set(d.results.flatMap((r) => r.evidence));
  $("sentView").innerHTML = d.sentences.map((s) => `<li value="${s.id}" id="sent-${s.id}" class="${cited.has(s.id) ? "cited" : ""}">${esc(s.text)}</li>`).join("");
  document.querySelectorAll(".crit").forEach((c) => c.onclick = () => {
    document.querySelectorAll(".crit").forEach((x) => x.classList.toggle("active", x === c));
    const r = d.results[+c.dataset.i];
    document.querySelectorAll("#sentView li").forEach((li) => li.classList.toggle("hl", r.evidence.includes(+li.value)));
    if (r.evidence.length) $("sent-" + r.evidence[0]).scrollIntoView({behavior: "smooth", block: "nearest"});
  });
  fillLabCriteria();
}

// ---------------------------------------------------------------- inference lab
function fillLabCriteria() {
  const c = collectCriteria();
  $("labCrit").innerHTML = c.length ? c.map((x, i) => `<option value="${i}">[${x.type}] ${esc(x.text.slice(0, 120))}</option>`).join("") : '<option value="">enter criteria in tab 1 first</option>';
}
$("labRun").onclick = async () => {
  const c = collectCriteria()[$("labCrit").value], note = $("note").value.trim();
  if (!c || note.length < 20) { $("labOut").innerHTML = '<div class="error">Enter a patient note and criteria in tab 1 first.</div>'; return; }
  $("labRun").disabled = true; $("labOut").innerHTML = '<p><span class="spinner"></span> running 4 decoding paths… (the uncached reference is the slow one)</p>';
  try {
    const r = await api("/api/compare", {note, criterion: c, k: +$("labK").value});
    const base = r.rows.find((x) => x.mode === "kv") || r.rows[0];
    $("labOut").innerHTML = `<table><tr><th>mode</th><th>output (generated text)</th><th>identical</th><th class="num">prompt tok</th><th class="num">new tok</th><th class="num">prefill s</th><th class="num">decode s</th><th class="num">total s</th><th class="num">tok/s</th><th class="num">speed vs KV</th><th>speculative stats</th></tr>
      ${r.rows.map((x) => `<tr><td><b>${esc(x.mode)}</b></td><td class="raw">${esc(x.output)}</td><td class="${x.same_as_first ? "yes" : "no"}">${x.same_as_first ? "yes" : "NO"}</td>
      <td class="num">${x.prompt_tokens}</td><td class="num">${x.new_tokens}</td><td class="num">${fmt(x.prefill_s)}</td><td class="num">${fmt(x.decode_s)}</td><td class="num">${fmt(x.total_s)}</td><td class="num">${fmt(x.tokens_per_s, 1)}</td>
      <td class="num">${fmt(base.total_s / x.total_s, 2)}×</td><td>${x.acceptance_rate !== undefined ? `accept ${fmt(x.acceptance_rate, 2)} (${x.accepted}/${x.drafted}), ${x.rounds} rounds, ${fmt(x.tokens_per_target_forward, 2)} tok/target-pass` : ""}</td></tr>`).join("")}</table>
      <p class="hint">${esc(r.note)}. "identical" compares each path's generated text with the first row (uncached reference). Timings are single runs on a laptop CPU; the Experiments tab has repeated, warmed-up benchmarks.</p>`;
  } catch (e) { $("labOut").innerHTML = `<div class="error">${esc(e.message)}</div>`; }
  finally { $("labRun").disabled = false; }
};
$("attRun").onclick = async () => {
  $("attRun").disabled = true; $("attOut").innerHTML = '<p><span class="spinner"></span> running…</p>';
  try {
    const r = await api("/api/attention_bench", {seq_len: +$("attS").value, query_len: +$("attT").value, block: +$("attB").value});
    const mx = Math.max(...r.rows.map((x) => x.median_ms));
    $("attOut").innerHTML = `<table><tr><th>implementation</th><th class="num">median ms (3 runs)</th><th style="width:30%"></th><th class="num">max |err|</th><th class="num">mean |err|</th></tr>
      ${r.rows.map((x) => `<tr><td>${esc(x.impl)}</td><td class="num">${fmt(x.median_ms, 2)}</td><td><div class="bar" style="width:${100 * x.median_ms / mx}%"></div></td><td class="num">${x.max_abs_err.toExponential(2)}</td><td class="num">${x.mean_abs_err.toExponential(2)}</td></tr>`).join("")}</table>
      <p class="hint">T=${r.T}, S=${r.S}. Full score matrix the reference materialises: ${fmt(r.score_matrix_mb, 2)} MB; one tile of the tiled kernel: ${fmt(r.tile_mb, 3)} MB. The Python tile loop is slower than the fused C++ SDPA kernel; its benefit is the bounded memory, not speed.</p>`;
  } catch (e) { $("attOut").innerHTML = `<div class="error">${esc(e.message)}</div>`; }
  finally { $("attRun").disabled = false; }
};

// ---------------------------------------------------------------- experiments (rendered from live log files)
function lineChart(series, w = 620, h = 220, ylab = "") {
  const pts = series.flatMap((s) => s.points);
  if (!pts.length) return "";
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = 0, y1 = Math.max(...ys) * 1.05;
  const X = (x) => 50 + (w - 70) * (x - x0) / ((x1 - x0) || 1), Y = (y) => h - 30 - (h - 50) * (y - y0) / ((y1 - y0) || 1);
  const ticks = [0, .25, .5, .75, 1].map((f) => y0 + f * (y1 - y0));
  return `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px">
    ${ticks.map((t) => `<line x1="50" x2="${w - 20}" y1="${Y(t)}" y2="${Y(t)}" stroke="#e5e7eb"/><text x="44" y="${Y(t) + 4}" text-anchor="end">${t.toFixed(2)}</text>`).join("")}
    <text x="${w / 2}" y="${h - 4}" text-anchor="middle">optimizer step</text><text x="12" y="14">${esc(ylab)}</text>
    ${series.map((s) => `<polyline fill="none" stroke="${s.color}" stroke-width="${s.width || 1.5}" points="${s.points.map((p) => `${X(p[0])},${Y(p[1])}`).join(" ")}"/>
      ${s.dots ? s.points.map((p) => `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="3.5" fill="${s.color}"/>`).join("") : ""}`).join("")}
    ${series.map((s, i) => `<rect x="${w - 190}" y="${12 + i * 16}" width="10" height="10" fill="${s.color}"/><text x="${w - 175}" y="${21 + i * 16}">${esc(s.name)}</text>`).join("")}
  </svg>`;
}
function smooth(a, k = 9) { return a.map((p, i) => { const s = a.slice(Math.max(0, i - k + 1), i + 1); return [p[0], s.reduce((t, q) => t + q[1], 0) / s.length]; }); }
async function loadExperiments() {
  const out = $("expOut");
  try {
    const d = await api("/api/experiments");
    let h = "";
    for (const [name, log] of Object.entries(d.train_curves)) {
      const tr = log.filter((r) => r.loss !== undefined).map((r) => [r.step, r.loss]);
      const va = log.filter((r) => r.val_loss !== undefined).map((r) => [r.step, r.val_loss]);
      h += `<div class="card"><h2>Training curve – ${esc(name)} (LoRA)</h2>${lineChart([{name: "train loss (step)", color: "#9db7e6", points: tr, width: 1},
            {name: "train loss (moving avg 9)", color: "#1f5fbf", points: smooth(tr), width: 2}, {name: "validation loss", color: "#c2410c", points: va, dots: true, width: 2}], 620, 220, "CE loss")}</div>`;
    }
    const ev = d.eval; const keys = Object.keys(ev);
    if (keys.length) {
      h += `<div class="card"><h2>Criterion-level evaluation (before vs after fine-tuning)</h2><table><tr><th>run</th><th>split</th><th class="num">n</th><th class="num">accuracy</th><th class="num">macro-F1</th><th class="num">incl. acc</th><th class="num">excl. acc</th><th class="num">evidence micro-F1</th><th class="num">evidence macro-F1 (gold≠∅)</th><th class="num">invalid</th></tr>
      ${keys.map((k) => { const m = ev[k]; return `<tr><td>${esc(k)}</td><td>${esc(m.meta.split)}</td><td class="num">${m.meta.n}</td><td class="num">${fmt(m.overall.classification.accuracy)}</td><td class="num">${fmt(m.overall.classification.macro_f1)}</td>
      <td class="num">${fmt(m.inclusion.classification.accuracy)}</td><td class="num">${fmt(m.exclusion.classification.accuracy)}</td><td class="num">${fmt(m.overall.evidence.micro.f1)}</td><td class="num">${fmt(m.overall.evidence.macro_nonempty_gold.f1)}</td><td class="num">${fmt(m.overall.classification.invalid_rate)}</td></tr>`; }).join("")}</table>
      <p class="hint">Decoding: ${esc(ev[keys[0]].meta.decoding)}.</p></div>`;
    }
    const kv = d.bench_kv;
    if (kv) {
      h += `<div class="card"><h2>KV cache vs no cache (${esc(kv.model)}, ${kv.repeats} timed runs after warm-up)</h2><table><tr><th class="num">input</th><th class="num">output</th><th class="num">no-cache s</th><th class="num">KV static s</th><th class="num">KV dynamic s</th><th class="num">speed-up</th><th class="num">KV prefill s</th><th class="num">KV decode tok/s</th><th class="num">no-cache tok/s</th><th class="num">KV MB</th><th class="num">peak ΔRSS MB (kv / no-cache)</th><th>tokens identical</th></tr>
      ${kv.rows.map((r) => `<tr><td class="num">${r.input_tokens}</td><td class="num">${r.output_tokens}</td><td class="num">${r.no_cache ? fmt(r.no_cache.total_s.median, 2) : "–"}</td><td class="num">${fmt(r.kv_static.total_s.median, 2)}</td><td class="num">${fmt(r.kv_dynamic.total_s.median, 2)}</td>
      <td class="num">${r.speedup_total ? fmt(r.speedup_total, 1) + "×" : "–"}</td><td class="num">${fmt(r.kv_static.prefill_s, 3)}</td><td class="num">${fmt(r.kv_static.decode_tokens_per_s, 1)}</td><td class="num">${r.no_cache ? fmt(r.no_cache.tokens_per_s, 1) : "–"}</td><td class="num">${fmt(r.kv_cache_analytic_mb, 1)}</td>
      <td class="num">${fmt(r.kv_static.peak_rss_delta_mb, 0)} / ${r.no_cache ? fmt(r.no_cache.peak_rss_delta_mb, 0) : "–"}</td><td class="${r.token_agreement === false ? "no" : "yes"}">${r.token_agreement === undefined ? "n/a (no ref)" : r.token_agreement ? "yes" : "NO"}</td></tr>`).join("")}</table></div>`;
    }
    const sp = d.bench_spec;
    if (sp) {
      for (const [wl, rows] of Object.entries(sp.workloads)) {
        const keys2 = Object.keys(rows[0]).filter((k) => k.startsWith("draft_"));
        const agg = keys2.map((k) => { const xs = rows.map((r) => r[k]);
          return {k, eq: xs.every((x) => x.equal_to_greedy), acc: xs.reduce((a, x) => a + x.accepted, 0) / xs.reduce((a, x) => a + x.drafted, 0),
            sp: xs.reduce((a, x) => a + x.speedup, 0) / xs.length, tps: xs.reduce((a, x) => a + x.tokens_per_s, 0) / xs.length, tpf: xs.reduce((a, x) => a + x.tokens_per_target_forward, 0) / xs.length}; });
        const g = rows.reduce((a, r) => a + r.greedy.tokens_per_s, 0) / rows.length;
        h += `<div class="card"><h2>Speculative decoding – workload "${esc(wl)}" (${rows.length} validation prompts; greedy target ${fmt(g, 1)} tok/s)</h2><table><tr><th>draft / k</th><th>all outputs identical to greedy</th><th class="num">acceptance rate</th><th class="num">tokens per target pass</th><th class="num">tok/s</th><th class="num">mean speed-up</th></tr>
        ${agg.map((a) => `<tr><td>${esc(a.k)}</td><td class="${a.eq ? "yes" : "no"}">${a.eq ? "yes" : "NO"}</td><td class="num">${fmt(a.acc)}</td><td class="num">${fmt(a.tpf, 2)}</td><td class="num">${fmt(a.tps, 1)}</td><td class="num">${fmt(a.sp, 2)}×</td></tr>`).join("")}</table></div>`;
      }
    }
    const at = d.bench_attention;
    if (at) {
      const worst = {}; at.errors.forEach((r) => worst[r.impl] = Math.max(worst[r.impl] || 0, r.max_abs_err));
      h += `<div class="card"><h2>Attention kernels – numerical error (worst over ${new Set(at.errors.map((r) => r.B + "-" + r.T + "-" + r.S)).size} shapes, vs float64 reference)</h2><table><tr><th>implementation</th><th class="num">worst max |err|</th></tr>
        ${Object.entries(worst).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${v.toExponential(2)}</td></tr>`).join("")}</table>
        <h3>Speed / memory</h3><table><tr><th class="num">T</th><th class="num">S</th><th>impl</th><th class="num">median ms</th><th class="num">peak ΔRSS MB</th><th class="num">analytic intermediate MB</th></tr>
        ${at.speed.map((r) => `<tr><td class="num">${r.T}</td><td class="num">${r.S}</td><td>${esc(r.impl)}</td><td class="num">${fmt(1000 * r.t_median, 2)}</td><td class="num">${fmt(r.peak_rss_delta_mb, 1)}</td><td class="num">${r.analytic_intermediate_mb === null ? "fused" : fmt(r.analytic_intermediate_mb, 2)}</td></tr>`).join("")}</table></div>`;
    }
    out.innerHTML = h || '<p class="hint">No experiment logs found yet – run the scripts in README §Reproduce.</p>';
  } catch (e) { out.innerHTML = `<div class="error">${esc(e.message)}</div>`; }
}

// ---------------------------------------------------------------- history
async function loadHistory() {
  try {
    const h = await api("/api/history");
    $("histOut").innerHTML = h.length ? `<table><tr><th>created (UTC)</th><th>trial</th><th>criteria</th><th>mode</th><th>preliminary outcome</th><th>report</th></tr>
      ${h.map((x) => `<tr><td>${esc(x.created.slice(0, 19).replace("T", " "))}</td><td>${esc(x.trial_id || "–")}</td><td>${x.n_criteria}</td><td>${esc(x.mode)}</td><td>${esc(x.verdict)}</td>
      <td><a href="/api/report/${x.id}.html" target="_blank">view</a> · <a href="/api/report/${x.id}.html?download=1">html</a> · <a href="/api/report/${x.id}.json">json</a></td></tr>`).join("")}</table>` : '<p class="hint">No reports yet.</p>';
  } catch (e) { $("histOut").innerHTML = `<div class="error">${esc(e.message)}</div>`; }
}

poll(); loadExamples();
