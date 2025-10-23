// ===== Tone.js graph =====
const ctx = {
  master: new Tone.Gain(0.9).toDestination(),
  a: { synth: new Tone.Synth({oscillator:{type:"sawtooth"}, envelope:{attack:0.01, release:0.1}}),
       fx: new Tone.Filter(400, "lowpass") },
  b: { synth: new Tone.Synth({oscillator:{type:"square"}, envelope:{attack:0.02, release:0.2}}),
       fx: new Tone.Chorus(4, 2.5, 0.2).start() },
  drums: { kick: new Tone.MembraneSynth({octaves:10, pitchDecay:0.02}),
           hat: new Tone.NoiseSynth({volume:-20, envelope:{attack:0.005, decay:0.05, sustain:0}}) },
  mixAB: new Tone.CrossFade(0.5),
  meter: new Tone.Meter()
};
ctx.a.synth.chain(ctx.a.fx, ctx.mixAB.a);
ctx.b.synth.chain(ctx.b.fx, ctx.mixAB.b);
ctx.drums.kick.connect(ctx.master);
ctx.drums.hat.connect(ctx.master);
ctx.mixAB.connect(ctx.master);
ctx.master.connect(ctx.meter);

// ===== State =====
let state = { t:0, alpha:0.5, features:{d_head:0, d_brake:0, d_thr:0}, sim:{} };
let mapping = defaultMapping;  // 可被 live editor 覆盖

// ===== Utils =====
function clamp(x, a, b){ return Math.max(a, Math.min(b, x)); }
function scale(x, inLo, inHi, outLo, outHi){
  const r = (x - inLo) / (inHi - inLo);
  return outLo + Math.max(0, Math.min(1, r)) * (outHi - outLo);
}

// ==== 7个调式（Ionian=Major 作为 0，依次循环）====
const MODE_SCALES = [
  [0,2,4,5,7,9,11], // Ionian (Major)
  [0,2,3,5,7,9,10], // Dorian
  [0,1,3,5,7,8,10], // Phrygian
  [0,2,4,6,7,9,11], // Lydian
  [0,2,4,5,7,9,10], // Mixolydian
  [0,2,3,5,7,8,10], // Aeolian (Minor)
  [0,1,3,5,6,8,10]  // Locrian
];
let CURRENT_SCALE = MODE_SCALES[0];
const ROOT = 57; // A3

function setMode(modeIndex){
  const idx = Math.max(0, Math.min(6, Math.floor(modeIndex)));
  CURRENT_SCALE = MODE_SCALES[idx];
}

function quantizeMidi(m) {
  // 把任意 MIDI 映射到 当前调式 最近音级
  const rel = m - ROOT;
  const oct = Math.floor(rel / 12);
  const frac = rel - oct*12;
  let best = CURRENT_SCALE[0], bestDist = 999;
  for (const step of CURRENT_SCALE){
    const d = Math.abs(step - frac);
    if (d < bestDist){ bestDist = d; best = step; }
  }
  return ROOT + oct*12 + best;
}

/* ----------------------------------------------------------------
   🎵 音乐引擎：统一拍速 + 固定鼓型 + 旋律/贝斯序列（音高量化）
------------------------------------------------------------------*/

// 1) 全局 Transport
Tone.Transport.bpm.value = 120;
Tone.Transport.swing = 0.04;
Tone.Transport.swingSubdivision = "8n";

// === 动态调式/音阶 ===
const SCALES = {
  major:     [0,2,4,5,7,9,11],
  minor:     [0,2,3,5,7,8,10],
  dorian:    [0,2,3,5,7,9,10],
  pentatonic:[0,3,5,7,10]
};
let currentScale = SCALES.pentatonic;
let currentRoot  = 57; // A3 默认根音

function setScaleByName(name, rootMidi){
  currentScale = SCALES[name] || currentScale;
  if (typeof rootMidi === 'number') currentRoot = rootMidi|0;
}
function quantizeToScale(midi){
  // 把任意 midi 量化到当前调式 & 根音附近
  const rel = midi - currentRoot;
  const oct = Math.floor(rel / 12);
  const within = rel - oct*12;
  // 找到当前音阶里距离最近的音
  let best = currentScale[0], bestDist = 999;
  for (const st of currentScale){
    const d = Math.abs(within - st);
    if (d < bestDist){ bestDist = d; best = st; }
  }
  return currentRoot + oct*12 + best;
}

// 3) 与特征相连的“连续参数”（mapping 里更新）
let hatDensity = 0.6;   // 0..1
let kickLevel  = 0.8;   // 0..1
let snrLevel   = 1.0;   // 0..1 （目前用 hat 代替军鼓触发）
let bassDepth  = 0.6;   // 0..1
let leadBright = 0.7;   // 0..1

// 4) 鼓：16 分音符 Loop（kick固定踏位、snare在 2/4 拍、hat 密度）
const drumLoop = new Tone.Loop((time) => {
  const step = Math.floor((Tone.Transport.ticks / Tone.Transport.PPQ) * 4) % 16;

  // kick：1/3 拍位固定；在 3.75拍附近偶尔补一脚
  if (step % 8 === 0 || (step % 8 === 6 && Math.random() < 0.3 * kickLevel)) {
    ctx.drums.kick.triggerAttackRelease("C2", "8n", time, kickLevel);
  }

  // “snare”：用 hat 代替的 2/4 拍重音
  if (step === 4 || step === 12) {
    if (snrLevel > 0.05) ctx.drums.hat.triggerAttackRelease("8n", time, snrLevel);
  }

  // hat：16 分音符，密度控制概率
  if (Math.random() < hatDensity) {
    ctx.drums.hat.triggerAttackRelease("16n", time, 0.6);
  }
}, "16n");
drumLoop.start(0);

// 5) Bass：四分音符走位，根音上下小摆动（量化）
const bassLoop = new Tone.Loop((time) => {
  const drift = scale(state.features.d_head, -1.5, 1.5, -7, 7);
  const base  = quantizeToScale(currentRoot + Math.round(drift));
  ctx.b.synth.triggerAttackRelease(
    Tone.Midi(base - 12).toFrequency(),
    "8n", time, bassDepth
  );
  ctx.b.fx.depth = 0.2 + 0.3 * Math.abs(state.features.d_brake);
}, "4n");
bassLoop.start(0);

// 6) Lead：八分音符，音高随 d_thr 上下摆（量化到当前调式）；混合 & FX 随相似度
const leadLoop = new Tone.Loop((time) => {
  const alpha = clamp(state.alpha, 0, 1);
  ctx.mixAB.fade.value = alpha; // A/B timbre crossfade

  const mov  = scale(state.features.d_thr, -1.2, 1.2, -5, 5);
  const note = quantizeToScale(currentRoot + 12 + Math.round(mov));
  ctx.a.synth.triggerAttackRelease(Tone.Midi(note).toFrequency(), "8n", time, leadBright);

  const hybrid = (state.sim && Object.keys(state.sim).find(k => k.startsWith('Hybrid'))) || null;
  const simH   = hybrid ? state.sim[hybrid] : 0.3;
  ctx.a.fx.frequency.value = 800 + 2400 * simH;
  ctx.b.fx.depth = 0.1 + 0.5 * simH;
}, "8n");
leadLoop.start("8n");

/* ----------------------------------------------------------------
   🎚️ 默认映射：只调整“连续参数”（让音乐稳定），不直接裸触发
   —— 融合服务端密度参数（state._kickDensity/_hatDensity/_snareDensity）
------------------------------------------------------------------*/
function defaultMapping(frame){
  const { alpha, features:{d_head, d_brake, d_thr} } = frame;

  // 混合推子：NOR/VER等类 vs Hybrid
  ctx.mixAB.fade.value = clamp(alpha, 0, 1);

  // 与服务端密度融合（0.5:0.5）
  const kBias = typeof state._kickDensity  === 'number' ? state._kickDensity  : 0.5;
  const hBias = typeof state._hatDensity   === 'number' ? state._hatDensity   : 0.5;
  const sBias = typeof state._snareDensity === 'number' ? state._snareDensity : 0.5;

  // 鼓的强弱 / 密度
  kickLevel  = clamp(0.5 + Math.abs(d_brake)*0.8, 0.2, 1.0)*0.5 + 0.5*kBias;
  snrLevel   = clamp(0.6 + Math.abs(d_brake)*0.5, 0.2, 1.0)*0.5 + 0.5*sBias;
  hatDensity = clamp(0.3 + Math.abs(d_thr)*0.7,   0.1, 0.95)*0.5 + 0.5*hBias;

  // 音色亮度
  leadBright = clamp(0.5 + Math.abs(d_head)*0.5, 0.2, 1.0);
  bassDepth  = clamp(0.4 + Math.abs(d_thr)*0.6,  0.2, 1.0);

  // BPM 轻微随 α 漂移（若服务端没覆写）
  // （若服务端推了 bpm，会在 applySynthParams 生效）
  if (!state._bpmLocked) {
    Tone.Transport.bpm.rampTo(110 + 40*alpha, 0.2);
  }
}

/* ----------------------------------------------------------------
   Live coding 编辑器（保留）
------------------------------------------------------------------*/
const codeEl = document.getElementById('code');
if (codeEl) codeEl.value = `function mapping(frame){ defaultMapping(frame); }`;

const applyBtn = document.getElementById('apply');
if (applyBtn) {
  applyBtn.onclick = ()=>{
    try{
      const fn = new Function('ctx','state','defaultMapping', codeEl.value + '; return mapping;');
      mapping = fn(ctx, state, defaultMapping);
      alert('Applied!');
    }catch(e){ alert('Error:\n' + e.message); }
  };
  document.addEventListener('keydown', (e)=>{
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter'){ applyBtn.click(); }
  });
}

/* ----------------------------------------------------------------
   开始音频（必须用户点击）+ 启动 Transport
------------------------------------------------------------------*/
const startBtn = document.getElementById('start');
if (startBtn){
  startBtn.onclick = async ()=>{
    await Tone.start();
    if (Tone.Transport.state !== "started") Tone.Transport.start("+0.05"); // 稍微延时启动更稳
    startBtn.innerText = 'Running';
  };
}

/* ----------------------------------------------------------------
   合成参数（来自服务端 hybrid_params）
   支持：bpm / mode / lead / bass / kick / hat / snare + α 融合
------------------------------------------------------------------*/
function applySynthParams(params={}, alpha=0.5){
  if (typeof params.bpm === 'number'){
    state._bpmLocked = true;
    Tone.Transport.bpm.value = params.bpm;
  }
  if (typeof params.mode === 'string'){
    // 允许服务端切调式；根音可按赛道/车手映射（默认 A3）
    setScaleByName(params.mode, currentRoot);
  }

  function applySynthParams(params={}, alpha=0.5){
  // BPM / Swing
  if (typeof params.bpm === 'number'){
    Tone.Transport.bpm.rampTo(params.bpm, 0.1);
  }
  if (typeof params.swing === 'number'){
    Tone.Transport.swing = params.swing;             // 0..0.1
    Tone.Transport.swingSubdivision = "8n";
  }

  // 调式
  if (typeof params.mode === 'number'){
    setMode(params.mode);
  }

  // A/B timbre crossfade
  ctx.mixAB.fade.value = clamp(alpha, 0, 1);

  // 轨道“强度/亮度”（0..1）→ 音量/滤波
  const vol = (x)=> Tone.gainToDb(clamp(x,0,1));
  if (typeof params.lead === 'number')  ctx.a.synth.volume.value = vol(params.lead);
  if (typeof params.bass === 'number')  ctx.b.synth.volume.value = vol(params.bass);

  // 打击乐密度写入到状态，让 Loop 使用
  state._kickDensity  = clamp(params.kick ?? 0.5, 0, 1);
  state._hatDensity   = clamp(params.hat  ?? 0.5, 0, 1);
  // 你没有专门的 snare 音源，上面用 hat 代理了 snare 强度可忽略或复用
  }
  // 音量从 0..1 转 dB
  const vol = (x)=> Tone.gainToDb(clamp(x,0,1));
  if (typeof params.lead === 'number')  ctx.a.synth.volume.value = vol(params.lead);
  if (typeof params.bass === 'number')  ctx.b.synth.volume.value = vol(params.bass);

  // 节奏密度参数（与默认映射融合）
  state._kickDensity  = clamp(params.kick ?? 0.5, 0, 1);
  state._hatDensity   = clamp(params.hat  ?? 0.5, 0, 1);
  state._snareDensity = clamp(params.snare?? 0.5, 0, 1);

  // 同时更新 AB Crossfade（也用 alpha）
  ctx.mixAB.fade.value = clamp(alpha, 0, 1);
}

/* ----------------------------------------------------------------
   WebSocket 连接（保留）——建议把 window.WS_URL 写在 html 里
------------------------------------------------------------------*/
const wsSpan  = document.getElementById('ws');
const framePre= document.getElementById('frame');

function pickWSUrl(){
  const url = new URL(window.location.href);
  const q = url.searchParams.get('ws');
  if (q) return q;
  if (window.WS_URL) return window.WS_URL;
  return 'wss://YOUR-BRIDGE.onrender.com/ws';
}

function connect(){
  const ws = new WebSocket(pickWSUrl());
  ws.onopen = ()=>{
    if (wsSpan){ wsSpan.textContent='connected'; wsSpan.style.color='#10b981'; }
  };
  ws.onclose = ()=>{
    if (wsSpan){ wsSpan.textContent='disconnected'; wsSpan.style.color='#ef4444'; }
    setTimeout(connect, 1000);
  };
  ws.onmessage = (ev)=>{
    let msg = {};
    try{ msg = JSON.parse(ev.data); }catch(e){ return; }

    // A) 参数消息：来自 Streamlit 每 N 帧推送（风格→音乐）
    if (msg.type === 'hybrid_params' && msg.params){
      applySynthParams(msg.params, msg.alpha ?? state.alpha);
      if (framePre) framePre.textContent = JSON.stringify({hybrid_params: msg}, null, 2);
      return;
    }

    // B) 帧消息：{t, alpha, features, sim}
    if (typeof msg.t === 'number' && msg.features){
      state = msg;
      if (framePre) framePre.textContent = JSON.stringify(state, null, 2);
      if (Tone.getContext().state === 'running'){
        const f = structuredClone(state);
        const d = (x)=> (typeof x === 'number'? clamp(x,0,1):0.5);
        f.features._kickDensity  = d(state._kickDensity);
        f.features._hatDensity   = d(state._hatDensity);
        f.features._snareDensity = d(state._snareDensity);
        mapping(f); // 只调参数，音序由 Loop 播放
      }
    }
  };
}
connect();
