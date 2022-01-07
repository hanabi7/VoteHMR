import numpy as np
import visdom
import time


class VisdomObserver():
    def __init__(self, opt):
        self.use_html = opt.use_html
        self.server = opt.server
        self.port = opt.port
        self.win_size = opt.display_winsize
        self.display_id = opt.display_id
        self.display_single_pane_ncols = opt.display_single_pane_ncols
        self.name = opt.name
        print("=====>")
        print("server: {}, port: {}".format(self.server, self.port))
        self.viz = visdom.Visdom(
            server=self.server,
            port=self.port,
            env='loss',
            use_incoming_socket=False
        )
        self.vis = visdom.Visdom(
            server=self.server,
            port=self.port,
            env='images',
            use_incoming_socket=False
        )
        print("<=====")

    def display_current_results(self, visuals):
        if self.display_id > 0:
            if self.display_single_pane_ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
    table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
    table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
</style>""" % (w, h)
                ncols = self.display_single_pane_ncols
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 1
                for label, image_numpy in visuals.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy)
                    self.vis.image(image_numpy.transpose(2, 0, 1), win=self.display_id + idx,
                        opts={
                        'title': label,
                        'showlegend': True
                    })
                    idx += 1

    def plot_current_errors(self, epoch, counter_ratio, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[], 'Y':[], 'legend':list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.viz.line(
            X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)
