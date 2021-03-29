import visdom
import numpy as np
import torch


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self,
                 env_name='main',
                 vis_port=996):
        self.viz = visdom.Visdom(port=vis_port)
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

if __name__ == "__main__":

    epoch = [1,2,3,4]
    loss = torch.tensor([7, 6 ,5 ,4])
    plotter = VisdomLinePlotter(env_name='Tutorial Plots')
    for x, y in zip(epoch,loss):
        plotter.plot('loss', 'train', 'Class Loss', x, y)