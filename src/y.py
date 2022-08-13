import matplotlib.pyplot as plt

fig, ax = plt.subplots(2)
ax[1].plot(range(6))
ax[0].plot(range(6, 0, -1))
fig.show()
plt.show()