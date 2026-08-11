# %%
# #!/usr/bin/env python3
"""
Plot test — runs on master, displays on x1n via VS Code tunnel.
No display server needed.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless, no GUI window
import matplotlib.pyplot as plt

# ── Generate data ──────────────────────────────────────────
x = np.linspace(0, 4 * np.pi, 500)
y1 = np.sin(x) * np.exp(-x / 10)
y2 = np.cos(x) * np.exp(-x / 10)

# ── Create plot ────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: damped sine and cosine
ax1.plot(x, y1, label='Damped sine', color='#e74c3c', linewidth=2)
ax1.plot(x, y2, label='Damped cosine', color='#3498db', linewidth=2)
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitude')
ax1.set_title('Damped Oscillations')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: scatter with love
np.random.seed(42)
n = 200
rx = np.random.randn(n) * 0.3
ry = np.random.randn(n) * 0.3
ax2.scatter(rx + 2, ry + 2, c='#e74c3c', alpha=0.4, edgecolors='none')
ax2.scatter(rx - 2, ry - 2, c='#3498db', alpha=0.4, edgecolors='none')
ax2.scatter(rx - 2, ry + 2, c='#2ecc71', alpha=0.4, edgecolors='none')
ax2.scatter(rx + 2, ry - 2, c='#f39c12', alpha=0.4, edgecolors='none')

# Heart curve
t = np.linspace(0, 2 * np.pi, 300)
hx = 16 * np.sin(t)**3
hy = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
ax2.plot(hx * 0.2, hy * 0.2, '#e74c3c', linewidth=2, label='♡')
ax2.set_aspect('equal')
ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)
ax2.set_title('Cluster Love')
ax2.legend()
ax2.grid(True, alpha=0.2)

plt.show()
plt.tight_layout()
plt.savefig('/tmp/plot_output.png', dpi=150, bbox_inches='tight')
print("✅ Plot saved to /tmp/plot_output.png")
print("📂 Open it in VS Code: Ctrl+O → /tmp/plot_output.png")
print("   (VS Code previews images inline)")
display(fig)
# %%
