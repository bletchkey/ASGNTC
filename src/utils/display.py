import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class Image:
    def __init__(self, grid):
        self.grid = grid
        self.gridsize = grid.shape[0]
        self.fig = plt.figure(figsize=(6,6))
        self.ax = self.fig.add_subplot(111)
        self.im = self.ax.imshow(grid, cmap='gray', interpolation='nearest', extent=[0, self.gridsize, 0, self.gridsize])

    def setColorMap(self, cmap):
        self.im.set_cmap(cmap)

    def show(self):
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xticks(np.arange(0, self.gridsize+1, 1))
        plt.yticks(np.arange(0, self.gridsize+1, 1))
        plt.grid(True, color='grey', linestyle='--', linewidth=0.2, which='both')
        self.ax.xaxis.set_tick_params(labelbottom=False)
        self.ax.yaxis.set_tick_params(labelleft=False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.fig.tight_layout(pad=0)
        plt.show()


    def save(self, filename):
        self.fig.savefig(filename)