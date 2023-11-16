import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio

import numpy as np

class Drawable:
    def __init__(self) -> None:
        pass

    def draw(self):
        for row in self:
            print(" ".join(map(str, row)))

class Image:
    def __init__(self, grid):
        self.grid = grid
        self.gridsize = grid.shape[0]
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.im = self.ax.imshow(grid, cmap='gray', interpolation='nearest', extent=[0, self.gridsize, 0, self.gridsize])

    def set_colormap(self, cmap):
        self.im.set_cmap(cmap)

    def _configure_axes(self):
        self.ax.set_aspect('equal', adjustable='box')
        ticks = np.arange(0, self.gridsize + 1, 1)
        self.ax.set_xticks(ticks)
        self.ax.set_yticks(ticks)
        self.ax.grid(True, color='grey', linestyle='--', linewidth=0.2, which='both')
        self.ax.xaxis.set_tick_params(labelbottom=False)
        self.ax.yaxis.set_tick_params(labelleft=False)
        for spine in ['top', 'right', 'bottom', 'left']:
            self.ax.spines[spine].set_visible(False)
        self.fig.tight_layout(pad=0)

    def show(self):
        self._configure_axes()
        plt.show()

    def save(self, filename):
        self._configure_axes()
        self.fig.savefig(filename)


class AnimatedImage():
    def __init__(self, grid_list):
        self.grid_list = grid_list

    def __grid_to_image(self, grid):
        return Image(grid).im

    def save(self, output_filename):
        with imageio.get_writer(output_filename, duration=0.2) as writer:
            for grid in self.grid_list:
                image = self.__grid_to_image(grid)
                writer.append_data(image)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class ImageSlider:
    def __init__(self, grid_list1, grid_list2):
        self.grid_list1 = grid_list1
        self.grid_list2 = grid_list2
        self.current_step = 0

        self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 5))
        plt.subplots_adjust(bottom=0.25)

        self.slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(self.slider_ax, 'Step', 0, len(grid_list1) - 1, valinit=0, valstep=1)

        self.slider.on_changed(self.update)

        self.update(0)  # Initial update

    def update(self, val):
        self.current_step = int(self.slider.val)
        self.show_images()

    def show_images(self):
        self.ax[0].clear()
        self.ax[1].clear()

        image1 = self.grid_list1[self.current_step]
        image2 = self.grid_list2[self.current_step]

        self.ax[0].imshow(image1, cmap='gray')
        self.ax[0].set_title('Image 1')

        self.ax[1].imshow(image2, cmap='gray')
        self.ax[1].set_title('Image 2')

        self.fig.canvas.draw()

    def show(self):
        plt.show()


