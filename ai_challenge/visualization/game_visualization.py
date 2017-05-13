import sys

import numpy as np
import matplotlib
from matplotlib import gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

matplotlib.use('TkAgg')

if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk


class PlotDataManager(object):
    def __init__(self, prop_dict, step_dict, step_proj_dict):
        self._figure = Figure(figsize=(10, 8), dpi=100)
        gs = gridspec.GridSpec(1, len(step_dict) + len(prop_dict),
                               width_ratios=[3] * len(step_dict) + [1] * len(prop_dict))
        self.prop_ax_dict = {name: self._figure.add_subplot(spec) for spec, name in
                             zip(gs[:len(prop_dict)], prop_dict.keys())}
        self.step_ax_dict = {name: self._figure.add_subplot(spec) for spec, name in
                             zip(gs[:len(step_dict)], prop_dict.keys())}

        self.prop_dict = prop_dict
        self.step_dict = step_dict
        self.step_proj_dict = step_proj_dict

    def update(self, cut_step=False):

        for name, ax in self.prop_ax_dict.items():
            ax.clear()
            vec_to_plot = self.prop_dict[name].data
            ax.pcolormesh(vec_to_plot.reshape(-1, np.sqrt(vec_to_plot.shape[1])), cmap='RdBu')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.suptitle(name, fontsize=10)

        for name, ax in self.step_ax_dict.items():
            vec_to_plot = self.step_dict[name].data
            trans_vec_to_plot = self.step_proj_dict[name](vec_to_plot)
            ax.plot(trans_vec_to_plot,
                    '.-b' if not cut_step else '.b')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.suptitle(name, fontsize=10)

    @property
    def figure(self):
        return self._figure


class Plotter(object):
    def __init__(self, plotter):
        self.root = Tk.Tk()
        self.root.wm_title("Embedding in TK")
        self._plotter = plotter
        self.canvas = FigureCanvasTkAgg(self.p.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.canvas.show()
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self.root)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.canvas.mpl_connect('key_press_event', self.on_key_event)
        self.button = Tk.Button(master=self.root, text='Quit', command=self._quit)
        self.button.pack(side=Tk.BOTTOM)

    def on_key_event(self, event):
        self._plotter.update()
        self.canvas.draw()
        key_press_handler(event, self.canvas, self.toolbar)

    def start(self):
        Tk.mainloop()

    def _quit(self):
        self.root.quit()
        self.root.destroy()
