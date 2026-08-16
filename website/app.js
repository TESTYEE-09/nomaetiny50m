import { AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

const $ = (selector) => document.querySelector(selector);
const ui = { chat: $("#chat"), empty: $("#empty"), form: $("#composer"), prompt: $("#prompt"), send: $("#send"), status: $("#status"), dot: $("#dot"), runtime: $("#runtime"), clear: $("#clear"), model: $("#model") };
const MODELS = {
  wiki: { label: "Wikipedia", dir: "model-wiki", name: "tiny50m-wiki", tokenizer: "model-wiki", cache: "tiny50m-wiki" },
  v1: { label: "Original", dir: "model", name: "tiny50m-fp16", tokenizer: "model", cache: "tiny50m-v2" },
};
const MAX_CONTEXT = 512, MAX_NEW_TOKENS = 160;
let session, tokenizer, generating = false, turns = [], cacheDtype = "float16", current = MODELS.wiki;

env.allowRemoteModels = false; env.allowLocalModels = true; env.localModelPath = new URL("./", location.href).href;
ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2); ort.env.wasm.simd = true;
ort.env.webgpu.powerPreference = "high-performance";

function setState(label, kind = "") { ui.status.textContent = label; ui.dot.className = kind; }

function urls(m = current) {
  return {
    model: new URL(`./${m.dir}/${m.name}.onnx`, location.href).href,
    manifest: new URL(`./${m.dir}/${m.name}-manifest.json`, location.href).href,
    int8: new URL(`./${m.dir}/${m.name}-int8.onnx`, location.href).href,
  };
}

async function modelBytes(url) {
  const cache = await caches.open(current.cache);
  let response = await cache.match(url);
  if (response) { setState("Loading local copy", "busy"); return response.arrayBuffer(); }
  setState("Downloading 0%", "busy"); response = await fetch(url);
  if (!response.ok) throw new Error(`Model download failed (${response.status})`);
  const total = Number(response.headers.get("content-length")) || 58205563, reader = response.body.getReader(), chunks = [];
  let received = 0;
  while (true) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); received += value.length; setState(`Downloading ${Math.min(100, Math.round(received / total * 100))}%`, "busy"); }
  const bytes = new Uint8Array(received); let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  await cache.put(url, new Response(bytes, { headers: { "content-type": "application/octet-stream" } }));
  return bytes.buffer;
}

async function loadExternalWeights() {
  const u = urls();
  const manifest = await fetch(u.manifest).then((response) => response.json());
  const cache = await caches.open(current.cache);
  let complete = 0;
  const files = await Promise.all(manifest.files.map(async (path) => {
    const url = new URL(`./${current.dir}/${path}`, location.href).href;
    let response = await cache.match(url);
    if (!response) { response = await fetch(url); if (!response.ok) throw new Error(`Weight shard failed (${response.status})`); await cache.put(url, response.clone()); }
    const data = await response.arrayBuffer(); complete += 1;
    setState(`Loading GPU weights ${Math.round(complete / manifest.files.length * 100)}%`, "busy");
    return { path, data };
  }));
  return files;
}

async function load() {
  if (session) return; ui.send.disabled = true;
  try {
    const loadedTokenizer = AutoTokenizer.from_pretrained(current.tokenizer, { local_files_only: true });
    tokenizer = await loadedTokenizer; setState("Starting model", "busy");
    const u = urls(), wantsGpu = "gpu" in navigator;
    if (wantsGpu) {
      try {
        const [bytes, externalData] = await Promise.all([modelBytes(u.model), loadExternalWeights()]);
        session = await ort.InferenceSession.create(bytes, { externalData, executionProviders: ["webgpu", "wasm"], graphOptimizationLevel: "all" });
        cacheDtype = "float16"; ui.runtime.textContent = "On-device · WebGPU"; setState("Ready"); return;
      } catch (error) { console.warn("WebGPU failed, falling back to CPU:", error); }
    }
    const bytes = await modelBytes(u.int8);
    session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    cacheDtype = "float32"; ui.runtime.textContent = "On-device · CPU (int8)"; setState("Ready");
  } catch (error) { console.error(error); setState("Could not load", "error"); ui.runtime.textContent = "Use current Chrome or Edge"; throw error; }
  finally { ui.send.disabled = false; }
}

function addMessage(role, text = "") {
  ui.empty.hidden = true; ui.chat.classList.add("active");
  const node = document.createElement("div"); node.className = `message ${role}`; node.textContent = text; ui.chat.append(node);
  node.scrollIntoView({ behavior: "smooth", block: "end" }); return node;
}

function choose(logits, recent, temperature = .78, topK = 36) {
  const ranked = [], seen = new Set(recent.slice(-64));
  for (let i = 0; i < logits.length; i++) { let score = logits[i] / temperature; if (seen.has(i)) score = score > 0 ? score / 1.12 : score * 1.12; if (ranked.length < topK) { ranked.push([score, i]); ranked.sort((a,b) => a[0] - b[0]); } else if (score > ranked[0][0]) { ranked[0] = [score, i]; ranked.sort((a,b) => a[0] - b[0]); } }
  const max = ranked.at(-1)[0], weights = ranked.map(([score]) => Math.exp(score - max)); let needle = Math.random() * weights.reduce((a,b) => a + b, 0);
  for (let i = 0; i < weights.length; i++) if ((needle -= weights[i]) <= 0) return ranked[i][1]; return ranked.at(-1)[1];
}

function emptyCache() {
  const feeds = {};
  const zeros = cacheDtype === "float16" ? new Uint16Array(0) : new Float32Array(0);
  for (let layer = 0; layer < 12; layer++) for (const kind of ["key", "value"]) feeds[`past_${layer}_${kind}`] = new ort.Tensor(cacheDtype, zeros, [1, 2, 0, 64]);
  return feeds;
}

async function reply(userText, target) {
  const context = turns.slice(0, -1).slice(-4).map(t => `<|${t.role}|>${t.text}`).join("");
  const prompt = `<|bos|><|system|>You are Tiny50M, a concise helpful assistant.<|eos|>${context}<|user|>${userText}<|eos|><|assistant|>`;
  let ids = Array.from((await tokenizer(prompt, { add_special_tokens: false })).input_ids.data, Number).slice(-MAX_CONTEXT);
  let feeds = emptyCache(); feeds.input_ids = new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]);
  let output = await session.run(feeds); const generated = [], started = performance.now(); target.classList.add("cursor");
  for (let step = 0; step < MAX_NEW_TOKENS && generating; step++) {
    const token = choose(output.logits.data, generated); if (token === 1) break; generated.push(token);
    target.textContent = tokenizer.decode(generated, { skip_special_tokens: true }); target.scrollIntoView({ behavior: "smooth", block: "end" }); await new Promise(requestAnimationFrame);
    const next = { input_ids: new ort.Tensor("int64", BigInt64Array.of(BigInt(token)), [1, 1]) };
    for (let layer = 0; layer < 12; layer++) for (const kind of ["key", "value"]) next[`past_${layer}_${kind}`] = output[`present_${layer}_${kind}`];
    output = await session.run(next); const tps = generated.length / ((performance.now() - started) / 1000); ui.runtime.textContent = `On-device · ${tps.toFixed(1)} tok/s`;
  }
  target.classList.remove("cursor"); return target.textContent.trim();
}

function resetChat() {
  generating = false; turns = []; ui.chat.replaceChildren(); ui.chat.classList.remove("active"); ui.empty.hidden = false; setState("Ready");
}

ui.model.addEventListener("change", async () => {
  if (current.key === ui.model.value) return;
  current = MODELS[ui.model.value]; current.key = ui.model.value;
  session = null; tokenizer = null; resetChat(); ui.runtime.textContent = "Runs locally";
  try { await load(); setState("Ready"); } catch { setState("Could not load", "error"); }
});

ui.form.addEventListener("submit", async (event) => {
  event.preventDefault(); const text = ui.prompt.value.trim(); if (!text || generating) return;
  generating = true; ui.send.disabled = true; addMessage("user", text); turns.push({ role: "user", text }); ui.prompt.value = ""; ui.prompt.style.height = "auto"; const answer = addMessage("assistant");
  try { await load(); setState("Thinking", "busy"); const result = await reply(text, answer); turns.push({ role: "assistant", text: result }); setState("Ready"); }
  catch { answer.textContent = "Model could not start on this browser."; }
  finally { generating = false; ui.send.disabled = false; ui.prompt.focus(); }
});
ui.prompt.addEventListener("input", () => { ui.prompt.style.height = "auto"; ui.prompt.style.height = `${ui.prompt.scrollHeight}px`; });
ui.prompt.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); ui.form.requestSubmit(); } });
ui.clear.addEventListener("click", resetChat);