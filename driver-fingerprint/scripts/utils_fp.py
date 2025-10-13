import os, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
import fastf1
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from numpy.linalg import norm
from scipy.spatial import procrustes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------- data -----------------
def _phase_resample(phase, y, grid):
    m = ~(np.isnan(phase) | np.isnan(y))
    if m.sum() < 8: return None
    x0, y0 = phase[m], y[m]
    o = np.argsort(x0); x1, y1 = x0[o], y0[o]
    x2, idx = np.unique(x1, return_index=True); y2 = y1[idx]
    return np.interp(grid, x2, y2)

def collect_driver_laps_resampled(session, drvcode, n_pts=220, max_laps=4):
    laps = session.laps.pick_drivers([drvcode])
    laps = laps[laps["LapTime"].notna()].sort_values("LapTime").head(max_laps)
    S_list, A_list, M_list = [], [], []
    for _, lap in laps.iterrows():
        car = lap.get_car_data().add_distance()
        df = car.copy()
        has_xy = False
        try:
            pos = lap.get_pos_data()[["Time","X","Y"]]
            df = car.merge(pos, on="Time", how="left").interpolate(limit_direction="both")
            has_xy = df["X"].notna().any() and df["Y"].notna().any()
        except Exception:
            has_xy = False

        t = (df["Time"] - df["Time"].iloc[0]).dt.total_seconds().to_numpy()
        if len(t) < 30:
            continue

        dist  = df["Distance"].to_numpy().astype(float)
        phase = (dist - np.nanmin(dist)) / max(1e-6, (np.nanmax(dist)-np.nanmin(dist)))
        phase = np.clip(phase, 0, 1)
        v = (df["Speed"].to_numpy().astype(float) * 1000/3600)
        a_long = np.gradient(v, t)

        if has_xy:
            x = df["X"].to_numpy().astype(float); y = df["Y"].to_numpy().astype(float)
            vx = np.gradient(x, t); vy = np.gradient(y, t)
            heading = np.unwrap(np.arctan2(vy, vx))
            ds = np.clip(np.gradient(dist), 1e-3, None)
            kappa  = np.gradient(heading)/ds
            a_lat  = (v**2) * kappa
            d_head = np.gradient(heading, t)
        else:
            a_lat  = np.zeros_like(v)
            d_head = np.zeros_like(v)

        thr = (df["Throttle"].to_numpy().astype(float) / 100.0)
        brk = (df["Brake"].astype(float).to_numpy()>0).astype(float)
        d_thr = np.gradient(thr, t)
        d_brk = np.gradient(brk, t)

        grid = np.linspace(0,1,n_pts)
        feats = [
            phase,
            v/90.0,
            np.tanh(a_lat/6.0),
            np.tanh(a_long/5.0),
            np.tanh(np.sqrt(a_lat**2 + a_long**2)/9.81)
        ]
        acts = [d_head, d_brk, d_thr]

        S_parts, A_parts, ok = [], [], True
        for arr in feats:
            ri = _phase_resample(phase, arr, grid)
            if ri is None: ok=False; break
            S_parts.append(ri)
        if not ok: continue
        for arr in acts:
            ri = _phase_resample(phase, arr, grid)
            if ri is None: ok=False; break
            A_parts.append(ri)
        if not ok: continue

        S_list.append(np.stack(S_parts,1).astype(np.float32))
        A_list.append(np.stack(A_parts,1).astype(np.float32))
        M_list.append(pd.DataFrame({"driver": drvcode, "phase": grid}))
    return S_list, A_list, M_list

def build_dataset(S_list, A_list, M_list, use_window=True, windows=((0.15,0.55),(0.60,0.85))):
    assert len(S_list)>0
    grid = np.linspace(0,1,S_list[0].shape[0])
    if use_window:
        mask = np.zeros_like(grid, dtype=bool)
        for a,b in windows: mask |= (grid>=a) & (grid<=b)
    else:
        mask = slice(None)
    Ss, As, Ms, lap_driver, lap_sizes = [], [], [], [], []
    for S, A, M in zip(S_list, A_list, M_list):
        partS, partA = S[mask], A[mask]
        Ss.append(partS); As.append(partA)
        Ms.append(M.iloc[np.where(mask)[0]] if not isinstance(mask, slice) else M)
        lap_driver.append(M["driver"].iloc[0]); lap_sizes.append(partS.shape[0])
    S_all = np.concatenate(Ss,0).astype(np.float32)
    A_all = np.concatenate(As,0).astype(np.float32)
    meta  = pd.concat(Ms, ignore_index=True)
    return S_all, A_all, meta, np.array(lap_driver), np.array(lap_sizes)

# ----------------- model -----------------
class PolicyNet(nn.Module):
    def __init__(self, sd, ad, n, z_dim=8, hidden=96, use_embedding=True):
        super().__init__()
        self.use_embedding = use_embedding
        self.allow_override = True  # ✅ 允许手动覆盖 embedding

        if use_embedding:
            self.emb = nn.Embedding(n, z_dim)
            in_dim = sd + z_dim
        else:
            self.emb = None
            in_dim = sd

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, ad * 2)
        )

    def forward(self, s, d=None, z_override=None):
        # ✅ 若给了 z_override，就直接用，不查 embedding
        if self.use_embedding:
            if (z_override is not None) and self.allow_override:
                z = z_override
            else:
                z = self.emb(d)
            x = torch.cat([s, z], -1)
        else:
            x = s

        out = self.net(x)
        m, lv = torch.chunk(out, 2, -1)
        return m, torch.clamp(lv, -4.0, 2.0)


def nll_weighted(m, lv, y, W=None):
    per = 0.5*((y-m)**2/ lv.exp() + lv)  # [B, act_dim]
    if W is not None:
        per = per * W
    return per.sum(dim=-1)

def train_and_eval(
    S_all, A_all, D_lap, lap_sizes, outdir,
    use_embedding=True, epochs=20, batch=1024, seed=7, weights=(1.0,1.3,1.0),
    # --- 新增可选超参 ---
    z_dim=8, hidden=96, dropout=0.1, weight_decay=1e-4
):
    """
    兼容旧用法；外部可传 z_dim/hidden/dropout/weight_decay。
    """
    os.makedirs(outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(seed); torch.manual_seed(seed)

    # 标准化动作
    A_mu = A_all.mean(0, keepdims=True); A_std = A_all.std(0, keepdims=True)+1e-6
    A_all_std = (A_all - A_mu)/A_std

    
        # ---- 修正 lap_sizes ----
    lap_sizes = np.asarray(lap_sizes).reshape(-1)
    # 如果 lap_sizes 的长度等于样本数，而不是圈数，则重新推断圈数
    if lap_sizes.size == S_all.shape[0]:
        # 假设每圈样本数相等（取平均）
        avg_len = int(np.mean(lap_sizes))
        n_laps = max(1, S_all.shape[0] // avg_len)
        lap_sizes = np.full(n_laps, avg_len, dtype=int)
        print(f"⚠️ lap_sizes looks per-sample; corrected to {n_laps} laps × {avg_len} samples each.")


    # 圈级切分 -> 展开到样本级索引
    n_laps = len(lap_sizes)
    idx_laps = np.arange(n_laps); np.random.shuffle(idx_laps)
    sp = int(n_laps*0.8); lap_tr, lap_te = idx_laps[:sp], idx_laps[sp:]
    seg = np.cumsum(np.r_[0, lap_sizes])
    def span(i): return np.arange(seg[i], seg[i+1])
    id_tr = np.concatenate([span(i) for i in lap_tr]).astype(np.int64)
    id_te = np.concatenate([span(i) for i in lap_te]).astype(np.int64)

    S_tr, A_tr = S_all[id_tr], A_all_std[id_tr]
    S_te, A_te = S_all[id_te], A_all_std[id_te]
    D_tr = np.concatenate([np.full(span(i).size, D_lap[i]) for i in lap_tr]).astype(np.int64)
    D_te = np.concatenate([np.full(span(i).size, D_lap[i]) for i in lap_te]).astype(np.int64)

    # tensors
    to_t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
    S_tr_t, A_tr_t, D_tr_t = to_t(S_tr), to_t(A_tr), torch.tensor(D_tr, dtype=torch.long, device=device)
    S_te_t, A_te_t, D_te_t = to_t(S_te), to_t(A_te), torch.tensor(D_te, dtype=torch.long, device=device)

    # --- 用上新增的 z_dim / dropout / weight_decay ---
    n_drivers = int(D_lap.max()) + 1 if use_embedding else 1
    model = PolicyNet(
        sd=S_all.shape[1], ad=A_all.shape[1], n=n_drivers,
        z_dim=z_dim, hidden=hidden, dropout=dropout, use_embedding=use_embedding
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=2e-3, weight_decay=weight_decay)
    W = torch.tensor(weights, device=device)

    # 训练
    for ep in range(epochs):
        model.train()
        if len(S_tr_t) == 0: break
        p = torch.randperm(S_tr_t.size(0), device=device)
        for i in range(0, len(p), batch):
            j = p[i:i+batch]
            if use_embedding:
                m, lv = model(S_tr_t[j], D_tr_t[j])
            else:
                m, lv = model(S_tr_t[j], None)
            loss = nll_weighted(m, lv, A_tr_t[j], W).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    # Swap（确定性错配）
    model.eval()
    with torch.no_grad():
        if use_embedding:
            m_ok, lv_ok = model(S_te_t, D_te_t)
            wrong = (D_te_t + 1) % (int(D_lap.max()) + 1)
            m_bd, lv_bd = model(S_te_t, wrong)
        else:
            m_ok, lv_ok = model(S_te_t, None)
            m_bd, lv_bd = m_ok, lv_ok
        nll_ok = nll_weighted(m_ok, lv_ok, A_te_t, W).mean().item()
        nll_bd = nll_weighted(m_bd, lv_bd, A_te_t, W).mean().item()
        rel = (nll_bd - nll_ok) / (abs(nll_ok)+1e-9) * 100.0

    # Cross-NLL
    with torch.no_grad():
        K = int(D_lap.max()) + 1
        perK = []
        for d_id in range(K):
            d_vec = torch.full_like(D_te_t, d_id) if use_embedding else None
            m, lv = model(S_te_t, d_vec) if use_embedding else model(S_te_t, None)
            perK.append(nll_weighted(m, lv, A_te_t, W))
        perK = torch.stack(perK, dim=1)  # [B, K]
        mat = np.zeros((K, K)); D_np = D_te_t.cpu().numpy()
        for k in range(K):
            idx = np.where(D_np == k)[0]
            if len(idx) == 0: continue
            mat[k] = perK[idx].mean(0).cpu().numpy()
    import pandas as pd
    pd.DataFrame(mat).to_csv(os.path.join(outdir, "cross_nll.csv"), index=False)

    # 存模型
    torch.save(model.state_dict(), os.path.join(outdir, "model.pth"))
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        import json; json.dump({"correct_nll": nll_ok, "wrong_nll": nll_bd, "delta_percent": rel}, f, indent=2)

    return nll_ok, nll_bd, rel, model

def plot_dbrake_ci(A_list, M_list, out_png, n_pts=220):
    from collections import defaultdict
    grid = np.linspace(0,1,n_pts)
    by_drv = defaultdict(list)
    for A,M in zip(A_list, M_list):
        drv = M["driver"].iloc[0]
        by_drv[drv].append(A[:,1])  # d_brake
    plt.figure(figsize=(6,4))
    for drv, mats in by_drv.items():
        M = np.stack(mats,0)
        mean = M.mean(0); std = M.std(0); ci = 1.96*std/np.sqrt(max(1,M.shape[0]))
        plt.plot(grid, mean, label=drv)
        plt.fill_between(grid, mean-ci, mean+ci, alpha=0.15)
        plt.xlabel("phase"); plt.ylabel("d_brake")
    plt.title("d_brake vs phase (mean ±95%CI)")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()

def plot_dbrake_ci_from_flat(A_all, drv_names, lap_sizes, out_png):
    """
    从扁平数组画 d_brake vs phase 曲线（自动按每圈拆分 + 平滑）
    """
    import matplotlib.pyplot as plt
    from collections import defaultdict

    drv_names = np.asarray(drv_names).astype(str)
    lap_sizes = np.asarray(lap_sizes).astype(int)
    A_all = np.asarray(A_all)

    seg = np.cumsum(np.r_[0, lap_sizes])
    by_drv = defaultdict(list)
    for i, d in enumerate(drv_names):
        s, e = seg[i], seg[i+1]
        if e > len(A_all): break
        Ai = A_all[s:e]
        if Ai.ndim == 2:
            dbr = Ai[:, 1]
        else:
            dbr = Ai
        # 统一插值到 220 点
        x = np.linspace(0, 1, len(dbr))
        grid = np.linspace(0, 1, 220)
        by_drv[d].append(np.interp(grid, x, dbr))

    grid = np.linspace(0, 1, 220)
    plt.figure(figsize=(6,4))
    for drv, arrs in by_drv.items():
        M = np.stack(arrs, 0)
        mean = M.mean(0)
        std = M.std(0)
        ci = 1.96 * std / np.sqrt(max(1, M.shape[0]))
        plt.plot(grid, mean, label=drv)
        plt.fill_between(grid, mean-ci, mean+ci, alpha=0.15)
    plt.xlabel("phase")
    plt.ylabel("d_brake")
    plt.title("Integration_Q5 | z=16 | Δ=+3.3%")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
