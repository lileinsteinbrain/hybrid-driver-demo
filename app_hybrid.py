# app_hybrid.py  — Hybrid demo (Qualifying only)
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import streamlit as st

# ==== render backend safe ====
os.environ.setdefault("MPLBACKEND", "Agg")

# ==== paths ====
ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(ROOT, "driver-fingerprint", "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

RESULT_DIR   = os.path.join(ROOT, "driver-fingerprint", "results", "integration_Q5")
MODEL_PATH   = os.path.join(RESULT_DIR, "model.pth")
SUMMARY_JSON = os.path.join(RESULT_DIR, "summary_integration.json")

# ==== fastf1 cache: use /tmp on cloud ====
import fastf1
FASTF1_CACHE = os.path.join("/tmp", "fastf1_cache")
os.makedirs(FASTF1_CACHE, exist_ok=True)
fastf1.Cache.enable_cache(FASTF1_CACHE)

# ==== our utils / model ====
from utils_fp import (
    PolicyNet, nll_weighted,
    collect_driver_laps_resampled, build_dataset
)

# ==== Streamlit page ====
st.set_page_config(page_title="Hybrid Driver Demo", layout="wide")
st.sidebar.title("Hybrid Demo")
st.sidebar.caption("Real telemetry → similarity vs driver fingerprints (with Hybrid mix)")

# ---------------- Session selector ----------------
YEAR_OPTS = [2023, 2024]
EVENT_BY_YEAR = {
    2023: [
        "British Grand Prix",
        "United States Grand Prix",
        "Australian Grand Prix",
        "Bahrain Grand Prix",
        "Brazilian Grand Prix",
    ],
    2024: [
        "British Grand Prix",
        "United States Grand Prix",
        "Australian Grand Prix",
        "Bahrain Grand Prix",
        "Brazilian Grand Prix",
    ],
}

with st.sidebar.expander("Session"):
    year = st.selectbox("Year", YEAR_OPTS, index=0)
    event_name = st.selectbox("Event", EVENT_BY_YEAR[year], index=0)
    st.write("Session: **Qualifying (Q)**")

# read classes & best z from summary (fallback if not found)
@st.cache_resource(show_spinner=True)
def load_model_info(summary_json: str):
    classes = ["NOR", "RUS", "VER"]
    z_dim = 16
    if os.path.exists(summary_json):
        try:
            info = json.load(open(summary_json, "r"))
            classes = info.get("classes", classes)
            z_dim = int(info.get("best", {}).get("z_dim", 16))
        except Exception:
            pass
    return classes, z_dim

classes, best_z = load_model_info(SUMMARY_JSON)
drv_to_id = {d: i for i, d in enumerate(classes)}

# ---------------- Playback controls ----------------
with st.sidebar.expander("Playback"):
    play_speed = st.slider("Playback speed (steps/sec)", 1, 12, 4)
    auto_loop  = st.checkbox("Loop", True)

# ---------------- Hybrid controls ----------------
with st.sidebar.expander("Hybrid mix"):
    colA, colB = st.columns(2)
    with colA:
        drv_A = st.selectbox("Driver A", classes, index=0)
    with colB:
        drv_B = st.selectbox("Driver B", classes, index=1)
    alpha = st.slider("α (A weight)", 0.0, 1.0, 0.5, 0.05)

st.title("Hybrid Driver Similarity — Qualifying Telemetry vs Driver Fingerprints")
st.caption("Pick session & telemetry, then mix **Driver A/B** with α to generate a *Hybrid fingerprint* in real-time.")

# ---------------- Load session ----------------
@st.cache_data(show_spinner=True, ttl=3600)
def load_session(year: int, gp_name: str):
    sess = fastf1.get_session(year, gp_name, "Q")
    sess.load()
    return sess

session = load_session(year, event_name)

# ---------------- Collect laps (resampled) ----------------
@st.cache_data(show_spinner=True, ttl=900)
def get_resampled_laps(_session, codes, n_pts=220, max_laps=3):
    # underscore to avoid streamlit hashing the Session object
    packs = {}
    for code in codes:
        try:
            S_list, A_list, M_list = collect_driver_laps_resampled(
                _session, code, n_pts=n_pts, max_laps=max_laps
            )
        except Exception:
            S_list, A_list, M_list = [], [], []
        packs[code] = (S_list, A_list, M_list)
    return packs

packs = get_resampled_laps(session, classes, n_pts=220, max_laps=3)

# Telemetry driver + which lap
c1, c2 = st.columns([2, 1])
with c1:
    telem_drv = st.selectbox("Pick a driver (telemetry source)", classes, index=0)
S_list, A_list, M_list = packs.get(telem_drv, ([], [], []))
with c2:
    avail = len(S_list)
    lap_idx = st.number_input("Lap index (fastest-first)", min_value=0, max_value=max(0, avail-1), value=0, step=1)

if not S_list:
    st.error("No laps available for this driver/session. Try another driver or event.")
    st.stop()

# windows same as training
S_all, A_all, meta, D_lap, lap_sizes = build_dataset(
    [S_list[lap_idx]], [A_list[lap_idx]], [M_list[lap_idx]],
    use_window=True, windows=((0.15, 0.55), (0.60, 0.85))
)

# ---------------- Build model & load weights ----------------
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
sd, ad = S_all.shape[1], A_all.shape[1]

model = PolicyNet(sd=sd, ad=ad, n=len(classes), z_dim=best_z, hidden=96, use_embedding=True)
if os.path.exists(MODEL_PATH):
    try:
        state = torch.load(MODEL_PATH, map_location="cpu")
        # allow slight key mismatch
        model.load_state_dict(state, strict=False)
    except Exception as e:
        st.warning(f"Loading state_dict for PolicyNet: size mismatch (already handled); detail: {e}")
else:
    st.warning("model.pth not found — using randomly initialized weights (demo only).")
model.to(device).eval()

# normalize actions per-lap (same as training inference)
A_mu = A_all.mean(0, keepdims=True)
A_std = A_all.std(0, keepdims=True) + 1e-6
A_std_all = (A_all - A_mu) / A_std

to_t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
S_t = to_t(S_all)
A_t = to_t(A_std_all)

# time index state
if "t_idx" not in st.session_state:
    st.session_state.t_idx = 0
nT = S_t.shape[0]

# ---------------- Core helpers ----------------
def get_z(driver_code: str):
    """Return embedding vector z for a given driver code as [1, z_dim]."""
    idx = drv_to_id[driver_code]
    with torch.no_grad():
        z = model.emb.weight[idx:idx+1]
    return z.to(device)


def _fwd_with_z_override_or_bypass(S_step: torch.Tensor, z: torch.Tensor):
    """
    尝试优先用 model 的 z_override 接口；不行就手工绕过 embedding，
    直接 [S, z] → model.net(...)，再按 PolicyNet 的方式切分 m/lv。
    """
    # 1) 尝试新签名：forward(s, d=None, z_override=z)
    try:
        return model(S_step, None, z_override=z)
    except TypeError:
        pass
    except Exception:
        pass

    # 2) 兜底：直连 MLP
    if hasattr(model, "net"):
        x = torch.cat([S_step, z], dim=-1)        # [1, sd + z_dim]
        out = model.net(x)                         # [1, ad*2]
        m, lv = torch.chunk(out, 2, dim=-1)
        lv = torch.clamp(lv, -4.0, 2.0)
        return m, lv
    else:
        raise RuntimeError(
            "Model does not expose z_override nor .net for bypass. "
            "Please enable z_override in PolicyNet.forward."
        )


def step_nll_with_z(ti: int, z_override: torch.Tensor):
    """NLL at a single step using a provided z_override (Hybrid/任意 z)."""
    with torch.no_grad():
        S_step = S_t[ti:ti+1]
        m, lv = _fwd_with_z_override_or_bypass(S_step, z_override)
        nll = nll_weighted(m, lv, A_t[ti:ti+1]).mean().item()
        return float(nll)


def step_nll_with_driver_id(ti: int, driver_code: str):
    """NLL using该车手自己的 embedding（通过 id 索引）"""
    d_id = drv_to_id[driver_code]
    with torch.no_grad():
        S_step = S_t[ti:ti+1]
        # 一定要 long 索引
        d_idx = torch.tensor([d_id], dtype=torch.long, device=device)
        m, lv = model(S_step, d_idx)
        nll = nll_weighted(m, lv, A_t[ti:ti+1]).mean().item()
        return float(nll)
    

def step_similarities(ti: int, drvA: str, drvB: str, alpha: float):
    # classes nll
    nlls = []
    for name in classes:
        nlls.append(step_nll_with_driver_id(ti, name))
    nlls = np.array(nlls, dtype=np.float64)

    # hybrid nll
    zA = get_z(drvA)
    zB = get_z(drvB)
    z_mix = alpha * zA + (1.0 - alpha) * zB
    nll_h = step_nll_with_z(ti, z_mix)

    # convert to exp(-NLL)
    sim = np.exp(-nlls)
    sim = sim / (sim.sum() + 1e-9)
    sim_h = float(np.exp(-nll_h))

    # bar entries (classes + hybrid)
    labels = classes + [f"Hybrid[{drvA}/{drvB}|α={alpha:.2f}]"]
    values = list(sim) + [sim_h]
    return labels, values

# ---------------- UI: playback controls ----------------
ctrl_col, bar_col = st.columns([1, 3])
with ctrl_col:
    if st.button("⏮ Reset"):
        st.session_state.t_idx = 0
    play = st.toggle("▶ Play", value=True)

labels, values = step_similarities(st.session_state.t_idx, drv_A, drv_B, alpha)

with bar_col:
    st.subheader("Live similarity (higher = closer to fingerprint)")
    df_bar = pd.DataFrame({"driver": labels, "similarity": values})
    st.bar_chart(df_bar.set_index("driver"))

# bottom curves
st.subheader(f"Lap playback — step {st.session_state.t_idx+1}/{nT}  |  Event: {event_name} {year} (Q)")
cols = st.columns(3)
labels_curve = ["d_heading", "d_brake", "d_throttle"]
for i, c in enumerate(cols):
    with c:
        st.line_chart(pd.DataFrame({labels_curve[i]: A_all[:st.session_state.t_idx+1, i]}))

# timeline
prog = st.slider("timeline", 0, nT-1, st.session_state.t_idx, key="timeline")
st.session_state.t_idx = prog

# autoplay
if play:
    time.sleep(1.0 / max(1, int(play_speed)))
    st.session_state.t_idx = (st.session_state.t_idx + 1) % nT if auto_loop else min(nT-1, st.session_state.t_idx+1)
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

# =======================
#  SEGMENT-BY-SEGMENT EXPLAIN (DEBUG banner)
# =======================
st.markdown("### 🔧 DEBUG: Explain block is LOADED")

def make_segments_by_phase(meta_df: pd.DataFrame, n_segs: int = 6):
    phases = meta_df["phase"].to_numpy() if isinstance(meta_df, pd.DataFrame) and ("phase" in meta_df.columns) else np.linspace(0, 1, len(meta_df))
    idx = np.arange(len(phases))
    edges = np.linspace(0.0, 1.0, n_segs + 1)
    segs = []
    for i in range(n_segs):
        lo, hi = edges[i], edges[i+1]
        m = (phases >= lo - 1e-9) & (phases <= hi + 1e-9)
        rows = idx[m]
        if len(rows) == 0:
            continue
        segs.append((rows.min(), rows.max()+1, f"S{i+1} {lo:.2f}-{hi:.2f}"))
    return segs

@st.cache_data(show_spinner=False)
def _softmax(x):
    x = np.array(x, dtype=np.float64)
    x = x - x.min()
    p = np.exp(-x)
    return p / (p.sum() + 1e-12)

def explain_one_segment(s_lo, s_hi, S_t, A_t, classes, drv_to_id, model, device):
    with torch.no_grad():
        nlls = []
        for name in classes:
            d_id = drv_to_id[name]
            d_vec = torch.full((s_hi - s_lo,), d_id, device=device, dtype=torch.long)
            m, lv = model(S_t[s_lo:s_hi], d_vec)
            nll = nll_weighted(m, lv, A_t[s_lo:s_hi]).mean().item()
            nlls.append(nll)
        probs = _softmax(nlls)
        sims = [(classes[i], float(probs[i]), float(nlls[i])) for i in range(len(classes))]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims

def action_summary(A_all, s_lo, s_hi, labels=("d_heading", "d_brake", "d_throttle")):
    seg = A_all[s_lo:s_hi]
    mean = seg.mean(axis=0)
    std  = seg.std(axis=0)
    return {labels[i]: (float(mean[i]), float(std[i])) for i in range(min(len(labels), seg.shape[1]))}

st.markdown("---")
st.subheader("Explain this lap (segment-by-segment)")
n_segs = st.slider("How many segments", 4, 12, 6, help="按 phase 等分切段；后续可换成赛道/弯角切分")

if st.button("🧠 Explain current lap"):
    # 1) 构造分段
    if isinstance(meta, pd.DataFrame) and ("phase" in meta.columns):
        segs = make_segments_by_phase(meta, n_segs=n_segs)
    else:
        L = S_t.shape[0]
        cuts = np.linspace(0, L, n_segs+1).astype(int)
        segs = [(int(cuts[i]), int(cuts[i+1]), f"S{i+1}") for i in range(n_segs) if cuts[i+1] > cuts[i]]

    rows = []
    for (s_lo, s_hi, seg_name) in segs:
        sims = explain_one_segment(s_lo, s_hi, S_t, A_t, classes, drv_to_id, model, device)
        top1, top2 = sims[0], sims[1] if len(sims) > 1 else ("-", 0.0, 0.0)
        conf = float(top1[1] - (top2[1] if isinstance(top2, tuple) else 0.0))
        summ = action_summary(A_all, s_lo, s_hi)
        rows.append({
            "segment": seg_name,
            "len": int(s_hi - s_lo),
            "top1_driver": top1[0],
            "top1_sim": round(float(top1[1]), 3),
            "top2_driver": top2[0] if isinstance(top2, tuple) else "-",
            "top2_sim": round(float(top2[1]), 3) if isinstance(top2, tuple) else 0.0,
            "confidence": round(conf, 3),
            "μ d_head": round(summ.get("d_heading", (0,0))[0], 3),
            "μ d_brake": round(summ.get("d_brake", (0,0))[0], 3),
            "μ d_thr": round(summ.get("d_throttle", (0,0))[0], 3),
        })

    df_explain = pd.DataFrame(rows, columns=[
        "segment","len","top1_driver","top1_sim","top2_driver","top2_sim","confidence",
        "μ d_head","μ d_brake","μ d_thr"
    ])
    st.dataframe(df_explain, use_container_width=True)
    st.caption("Top1 driver by segment")
    st.bar_chart(df_explain.set_index("segment")[["top1_sim"]])


st.info("Similarity = exp(-NLL). We show class sims and also the **Hybrid(A,B,α)** sim computed via z_mix = α·z_A + (1-α)·z_B.")
st.caption("Quali only for now. Race & user-upload CSV will come next.")
