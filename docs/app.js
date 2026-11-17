// ===== Tone.js graph =====
const ctx = {
  master: new Tone.Gain(0.9).toDestination(),
  a: { synth: new Tone.Synth({oscillator:{type:"sawtooth"}, envelope:{attack:0.01, release:0.1}}),
       fx: new Tone.Filter(400, "lowpass") },
  b: { synth: new Tone.Synth({oscillator:{type:"square"}, envelope:{attack:0.02, release:0.2}}),
       fx: new Tone.Chorus(4, 2.5, 0.2).start() },
  drums: {
    kick: new Tone.MembraneSynth({octaves:10, pitchDecay:0.02}),
    hat:  new Tone.NoiseSynth({volume:-20, envelope:{attack:0.005, decay:0.05, sustain:0}})
  },
  mixAB: new Tone.CrossFade(0.5),
  meter: new Tone.Meter()
};
ctx.a.synth.chain(ctx.a.fx, ctx.mixAB.a);
ctx.b.synth.chain(ctx.b.fx, ctx.mixAB.b);
ctx.drums.kick.connect(ctx.master);
ctx.drums.hat.connect(ctx.master);
ctx.mixAB.connect(ctx.master);
ctx.master.connect(ctx.meter);

// ---- Master bus polish: compressor + reverb + limiter ----
ctx.bus = new Tone.Gain(1);
ctx.comp = new Tone.Compressor(-14, 3);
ctx.rev  = new Tone.Reverb({ decay: 2.4, wet: 0.12 });
ctx.lim  = new Tone.Limiter(-1);
ctx.master.disconnect();
ctx.master.connect(ctx.bus);
ctx.bus.chain(ctx.comp, ctx.rev, ctx.lim, Tone.Destination);

// ===== Add real snare & better hat tone =====
ctx.drums.snare = new Tone.NoiseSynth({
  volume: -10,
  envelope: { attack: 0.004, decay: 0.18, sustain: 0 }
});
ctx.drums.snareFx = new Tone.Filter(1800, "bandpass");
ctx.drums.snare.chain(ctx.drums.snareFx, ctx.master);

ctx.drums.hatFx = new Tone.Filter(9000, "highpass");
ctx.drums.hat.disconnect();
ctx.drums.hat.connect(ctx.drums.hatFx);
ctx.drums.hatFx.connect(ctx.master);

// ===== State =====
let state = { t:0, alpha:0.5, features:{d_head:0, d_brake:0, d_thr:0}, sim:{} };
let mapping = defaultMapping;  // can be overridden by the live editor

// ===== Utils =====
function clamp(x, a, b){ return Math.max(a, Math.min(b, x)); }
function scale(x, inLo, inHi, outLo, outHi){
  const r = (x - inLo) / (inHi - inLo);
  return outLo + Math.max(0, Math.min(1, r)) * (outHi - outLo);
}

// ==== Modes & quantisation (accepts names or numbers) ====
const SCALES = {
  ionian:     [0,2,4,5,7,9,11], // Major
  dorian:     [0,2,3,5,7,9,10],
  phrygian:   [0,1,3,5,7,8,10],
  lydian:     [0,2,4,6,7,9,11],
  mixolydian: [0,2,4,5,7,9,10],
  aeolian:    [0,2,3,5,7,8,10], // Minor
  locrian:    [0,1,3,5,6,8,10],
  pentatonic: [0,3,5,7,10]
};
const MODE_INDEX = ["ionian","dorian","phrygian","lydian","mixolydian","aeolian","locrian"];

let currentScale = SCALES.pentatonic;
let currentRoot  = 57; // A3

function setScaleByName(name, rootMidi){
  const key = (name||"").toLowerCase();
  if (SCALES[key]) currentScale = SCALES[key];
  if (typeof rootMidi === 'number') currentRoot = rootMidi|0;
}
function setMode(index){ // 0..6 maps to the seven church modes
  const nm = MODE_INDEX[(index|0) % MODE_INDEX.length];
  setScaleByName(nm, currentRoot);
}
function quantizeToScale(midi){
  const rel = midi - currentRoot;
  const oct = Math.floor(rel / 12);
  const within = rel - oct*12;
  let best = currentScale[0], bestDist = 999;
  for (const st of currentScale){
    const d = Math.abs(within - st);
    if (d < bestDist){ bestDist = d; best = st; }
  }
  return currentRoot + oct*12 + best;
}

/* ----------------------------------------------------------------
   🎵 Music engine: shared tempo + stable drum pattern + quantised melody/bass
------------------------------------------------------------------*/
Tone.Transport.bpm.value = 120;
Tone.Transport.swing = 0.04;
Tone.Transport.swingSubdivision = "8n";

// Continuous parameters (updated by mapping)
let hatDensity = 0.6;   // 0..1
let kickLevel  = 0.8;   // 0..1
let snrLevel   = 1.0;   // 0..1
let bassDepth  = 0.6;   // 0..1
let leadBright = 0.7;   // 0..1

// Drum loop: steady kicks on beats 1/3 + snare on 2/4 + hat density
const drumLoop = new Tone.Loop((time) => {
  const step = Math.floor((Tone.Transport.ticks / Tone.Transport.PPQ) * 4) % 16;

  // Kick: downbeats + occasional extra hits (probability on steps 6/14)
  const kickVel = 0.5 + 0.5 * (state._kickDensity ?? 0.7);
  if (step % 8 === 0 || (step % 8 === 6 && Math.random() < (state._kickDensity ?? 0.5)*0.6)) {
    ctx.drums.kick.triggerAttackRelease("C2", "8n", time, kickVel);
  }

  // Snare: beats 2 and 4
  const snrVel = 0.4 + 0.6 * (state._snareDensity ?? 0.8);
  if (step === 4 || step === 12) {
    ctx.drums.snare.triggerAttackRelease("8n", time, snrVel);
  }

  // Hat: 16th notes, density-controlled
  const hatP = 0.2 + 0.75 * (state._hatDensity ?? 0.6);
  if (Math.random() < hatP) {
    ctx.drums.hat.triggerAttackRelease("16n", time, 0.6);
  }
}, "16n");
drumLoop.start(0);

// Bass: quarter-note root that shifts slightly with d_head, quantised
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

// Lead: eighth notes, pitch follows d_thr, quantised; mix/FX follow similarity
const leadLoop = new Tone.Loop((time) => {
  const alpha = clamp(state.alpha, 0, 1);
  ctx.mixAB.fade.value = alpha;

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
   🎚️ Default mapping: continuous parameters (blended with server-side density)
------------------------------------------------------------------*/
function defaultMapping(frame){
  const { alpha, features:{d_head, d_brake, d_thr} } = frame;
  ctx.mixAB.fade.value = clamp(alpha, 0, 1);

  const kBias = typeof state._kickDensity  === 'number' ? state._kickDensity  : 0.5;
  const hBias = typeof state._hatDensity   === 'number' ? state._hatDensity   : 0.5;
  const sBias = typeof state._snareDensity === 'number' ? state._snareDensity : 0.5;

  kickLevel  = clamp(0.5 + Math.abs(d_brake)*0.8, 0.2, 1.0)*0.5 + 0.5*kBias;
  snrLevel   = clamp(0.6 + Math.abs(d_brake)*0.5, 0.2, 1.0)*0.5 + 0.5*sBias;
  hatDensity = clamp(0.3 + Math.abs(d_thr)*0.7,   0.1, 0.95)*0.5 + 0.5*hBias;

  leadBright = clamp(0.5 + Math.abs(d_head)*0.5, 0.2, 1.0);
  bassDepth  = clamp(0.4 + Math.abs(d_thr)*0.6,  0.2, 1.0);

  if (!state._bpmLocked) {
    Tone.Transport.bpm.rampTo(110 + 40*alpha, 0.2);
  }
}

/* ----------------------------------------------------------------
   Live coding editor (kept)
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
    }catch(e){ alert('Error:\\n' + e.message); }
  };
  document.addEventListener('keydown', (e)=>{
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter'){ applyBtn.click(); }
  });
}

/* ----------------------------------------------------------------
   Start audio (requires a user click) + start Transport
------------------------------------------------------------------*/
const startBtn = document.getElementById('start');
if (startBtn){
  startBtn.onclick = async ()=>{
    await Tone.start();
    if (Tone.Transport.state !== "started") Tone.Transport.start("+0.05");
    startBtn.innerText = 'Running';
  };
}

/* ----------------------------------------------------------------
   Synth parameters (from server-side hybrid_params)
------------------------------------------------------------------*/
function applySynthParams(params={}, alpha=0.5){
  // BPM / Swing
  if (typeof params.bpm === 'number'){
    state._bpmLocked = true;
    Tone.Transport.bpm.rampTo(params.bpm, 0.1);
  }
  if (typeof params.swing === 'number'){
    Tone.Transport.swing = clamp(params.swing, 0, 0.25);
    Tone.Transport.swingSubdivision = "8n";
  }

  // Mode: accepts a number (0..6) or a string
  if (typeof params.mode === 'number') setMode(params.mode);
  if (typeof params.mode === 'string') setScaleByName(params.mode, currentRoot);

  // AB crossfade
  ctx.mixAB.fade.value = clamp(alpha, 0, 1);

  // Track level -> dB
  const vol = (x)=> Tone.gainToDb(clamp(x,0,1));
  if (typeof params.lead === 'number')  ctx.a.synth.volume.value = vol(params.lead);
  if (typeof params.bass === 'number')  ctx.b.synth.volume.value = vol(params.bass);

  // Rhythm density
  state._kickDensity  = clamp(params.kick ?? 0.5, 0, 1);
  state._hatDensity   = clamp(params.hat  ?? 0.5, 0, 1);
  state._snareDensity = clamp(params.snare?? 0.5, 0, 1);

  updateHUD({
    bpm: Tone.Transport.bpm.value,
    swing: Tone.Transport.swing,
    ...params
  });
}

/* ----------------------------------------------------------------
   Tiny HUD (shows current parameters in the top-right corner)
------------------------------------------------------------------*/
const hud = document.createElement('div');
hud.style.cssText = `
  position:fixed; top:12px; right:12px; z-index:9999;
  background:rgba(17,17,17,.66); color:#cbd5e1; font:12px/1.4 ui-monospace,monospace;
  padding:10px 12px; border-radius:10px; box-shadow:0 6px 24px rgba(0,0,0,.35);
`;
document.body.appendChild(hud);
function updateHUD(p){
  hud.innerHTML = `
    <b>Live Params</b><br/>
    BPM: <b>${Math.round(p.bpm||0)}</b> &nbsp;
    Swing: <b>${(p.swing||0).toFixed(2)}</b><br/>
    Lead: ${(p.lead??0).toFixed(2)} &nbsp;
    Bass: ${(p.bass??0).toFixed(2)}<br/>
    Kick: ${(p.kick??0).toFixed(2)} &nbsp;
    Hat: ${(p.hat??0).toFixed(2)}
  `;
}

/* ----------------------------------------------------------------
   WebSocket connection
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

    // A) Parameter message
    if (msg.type === 'hybrid_params' && msg.params){
      applySynthParams(msg.params, msg.alpha ?? state.alpha);
      if (framePre) framePre.textContent = JSON.stringify({hybrid_params: msg}, null, 2);
      return;
    }

    // B) Frame message: {t, alpha, features, sim}
    if (typeof msg.t === 'number' && msg.features){
      state = msg;
      if (framePre) framePre.textContent = JSON.stringify(state, null, 2);
      if (Tone.getContext().state === 'running'){
        const f = structuredClone(state);
        const d = (x)=> (typeof x === 'number'? clamp(x,0,1):0.5);
        f.features._kickDensity  = d(state._kickDensity);
        f.features._hatDensity   = d(state._hatDensity);
        f.features._snareDensity = d(state._snareDensity);
        mapping(f);
      }
    }
  };
}
connect();
