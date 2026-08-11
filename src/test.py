# %%
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('My Plot')

display(fig)          # ← use this, not plt.show()
# plt.savefig(...)     # save still works too
# %%
