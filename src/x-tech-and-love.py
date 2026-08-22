# %%
"""
tech-and-love

Damped oscillations meet a heart of scattered stars.
No words needed.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ── Style ──────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#fef9f4',
    'axes.facecolor': '#fef9f4',
    'axes.edgecolor': 'none',
    'axes.labelcolor': '#ffffff00',
    'xtick.color': '#d5c8bd',
    'ytick.color': '#d5c8bd',
    'grid.color': '#f0e8de',
    'grid.alpha': 0.6,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#fef9f4')

# ── Left: Damped Oscillations ──────────────────────
t = np.linspace(0, 4 * np.pi, 600)
decay = np.exp(-t / (3 * np.pi))
y_sin = decay * np.sin(t)
y_cos = decay * np.cos(t)

# Glow effect layers
for width, alpha in [(4, 0.05), (2.5, 0.1), (1.5, 0.2)]:
    ax1.plot(t, y_sin, color='#ff6b8a', linewidth=width, alpha=alpha)
    ax1.plot(t, y_cos, color='#58a6ff', linewidth=width, alpha=alpha)
# Core lines
ax1.plot(t, y_sin, color='#ff6b8a', linewidth=1.2, path_effects=[
    pe.SimpleLineShadow(offset=(0, 0), shadow_color='#ff6b8a', alpha=0.6, rho=6)])
ax1.plot(t, y_cos, color='#58a6ff', linewidth=1.2, path_effects=[
    pe.SimpleLineShadow(offset=(0, 0), shadow_color='#58a6ff', alpha=0.6, rho=6)])

ax1.set_xlim(0, 4 * np.pi)
ax1.set_ylim(-1.3, 1.3)
ax1.grid(True, alpha=0.15)
ax1.set_xticks([])
ax1.set_yticks([])
for spine in ax1.spines.values():
    spine.set_visible(False)

# ── Right: Cluster Love ────────────────────────────
np.random.seed(42)

# Heart parametric
theta = np.linspace(0, 2 * np.pi, 400)
hx = 16 * np.sin(theta) ** 3
hy = 13 * np.cos(theta) - 5 * np.cos(2 * theta) - 2 * np.cos(3 * theta) - np.cos(4 * theta)
hx = hx / 6
hy = hy / 6

# Four clusters around the heart
clusters = [
    (-2.0,  1.5, 0.35, '#43d9ad'),   # green
    ( 2.0,  1.5, 0.35, '#ff6b8a'),   # rose
    (-2.0, -1.3, 0.35, '#58a6ff'),   # blue
    ( 2.0, -1.3, 0.35, '#f0a060'),   # gold
]

for cx, cy, spread, color in clusters:
    xs = cx + np.random.randn(120) * spread
    ys = cy + np.random.randn(120) * spread
    # Glow
    ax2.scatter(xs, ys, s=12, color=color, alpha=0.08, edgecolors='none')
    ax2.scatter(xs, ys, s=8, color=color, alpha=0.15, edgecolors='none')
    # Core points
    ax2.scatter(xs, ys, s=4, color=color, alpha=0.7, edgecolors='none')

# Heart outline with glow
for width, alpha in [(5, 0.08), (3, 0.15), (1.8, 0.4)]:
    ax2.plot(hx, hy, color='#ff4060', linewidth=width, alpha=alpha)
# Core heart
ax2.plot(hx, hy, color='#ff4060', linewidth=1.0, alpha=0.85,
         path_effects=[pe.SimpleLineShadow(offset=(0, 0), shadow_color='#ff4060', alpha=0.5, rho=8)])

ax2.set_xlim(-3.5, 3.5)
ax2.set_ylim(-3.0, 3.0)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.08)
ax2.set_xticks([])
ax2.set_yticks([])
for spine in ax2.spines.values():
    spine.set_visible(False)

# ── Finish ────────────────────────────────────────
plt.tight_layout(pad=2)
plt.savefig('/tmp/tech-and-love.png', dpi=200,
            bbox_inches='tight', facecolor='#fef9f4')
print("💫 Saved to /tmp/tech-and-love.png")

try:
    display(fig)
except NameError:
    plt.show()

# %%
