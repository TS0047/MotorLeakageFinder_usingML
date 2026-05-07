import numpy as np
import pandas as pd

np.random.seed(42)

# ═══════════════════════════════════════
# REAL ANCHOR POINTS (from your Excel)
# ═══════════════════════════════════════
real_RPM   = [1600,  1700,   1900  ]
real_IP    = [1.06,  1.24,   1.40  ]   # Intake Pressure (bar)
real_TURBO = [52000, 80000,  115000]   # Turbo Speed (rpm)

# AF (Airflow g/s) derived from Turbo Speed
# 52000 RPM turbo ≈ 6.0 g/s | 115000 RPM turbo ≈ 11.0 g/s
real_AF  = [(t - 52000)/(115000 - 52000)*(11.0 - 6.0) + 6.0 for t in real_TURBO]

# EGT estimated from RPM (higher RPM + load = hotter exhaust)
real_EGT = [r*0.05 + 250 for r in real_RPM]  # simplified linear

# EP (Exhaust Pressure) ≈ IP - 0.10 bar (pressure drop across engine)
real_EP  = [ip - 0.10 for ip in real_IP]

# Fit linear curves from 3 anchor points
ip_fit  = np.polyfit(real_RPM, real_IP,  1)
af_fit  = np.polyfit(real_RPM, real_AF,  1)
egt_fit = np.polyfit(real_RPM, real_EGT, 1)
ep_fit  = np.polyfit(real_RPM, real_EP,  1)

# ═══════════════════════════════════════
# HEALTHY DATA — 1000 rows
# ═══════════════════════════════════════
N = 1000
rpms = np.random.uniform(1500, 2200, N)

healthy = pd.DataFrame({
    'RPM': rpms.round(1),
    'AF':  (np.polyval(af_fit,  rpms) + np.random.normal(0, 0.15, N)).clip(3.0,  15.0),
    'EGT': (np.polyval(egt_fit, rpms) + np.random.normal(0, 5.0,  N)).clip(200,  650),
    'IP':  (np.polyval(ip_fit,  rpms) + np.random.normal(0, 0.02, N)).clip(0.90, 2.00),
    'EP':  (np.polyval(ep_fit,  rpms) + np.random.normal(0, 0.02, N)).clip(0.80, 1.80),
    'label': 'HEALTHY',
    'zone':  'NONE'
}).round(4)

# ═══════════════════════════════════════
# LEAK DATA — 50 rows per zone (200 total)
# ═══════════════════════════════════════
n = 50

def base(n):
    r = np.random.uniform(1500, 2200, n)
    return dict(
        RPM = r.round(1),
        AF  = np.polyval(af_fit,  r),
        EGT = np.polyval(egt_fit, r),
        IP  = np.polyval(ip_fit,  r),
        EP  = np.polyval(ep_fit,  r),
    )

# Zone A — Airflow drop >10% (airflow meter → turbo inlet)
za = base(n)
za['AF']  *= np.random.uniform(0.70, 0.88, n)          # 12–30% drop
za.update({'label': ['LEAK']*n, 'zone': ['A']*n})

# Zone B — Intake pressure drop >0.12 bar + EGT rise (compressor → intercooler)
zb = base(n)
zb['IP']  -= np.random.uniform(0.13, 0.25, n)          # pressure drop
zb['EGT'] += np.random.uniform(30,   70,   n)          # EGT spike
zb.update({'label': ['LEAK']*n, 'zone': ['B']*n})

# Zone C — EGT rise >25°C only (exhaust manifold → turbine)
zc = base(n)
zc['EGT'] += np.random.uniform(28,   80,   n)
zc.update({'label': ['LEAK']*n, 'zone': ['C']*n})

# Zone D — Exhaust pressure drop >0.08 bar (test cell ducting)
zd = base(n)
zd['EP']  -= np.random.uniform(0.09, 0.20, n)
zd.update({'label': ['LEAK']*n, 'zone': ['D']*n})

leak = pd.concat([pd.DataFrame(z).round(4) for z in [za,zb,zc,zd]], ignore_index=True)

# ═══════════════════════════════════════
# SAVE
# ═══════════════════════════════════════
healthy.to_csv('healthy_train.csv', index=False)
leak.to_csv('leak_test.csv',        index=False)

print("healthy_train.csv →", len(healthy), "rows")
print("leak_test.csv     →", len(leak),    "rows (50 per zone A/B/C/D)")
print("\nHealthy stats:")
print(healthy[['RPM','AF','EGT','IP','EP']].describe().round(3))
print("\nLeak zone counts:", dict(leak['zone'].value_counts()))