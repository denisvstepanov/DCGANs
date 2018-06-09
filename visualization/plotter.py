import os


from tensorboardX import SummaryWriter

TENSORBOARD_DIR = 'tensorboard/runs/'


class Plotter:
    def on_new_point(self, label, x, y):
        pass

    def on_finish(self):
        pass


class TensorboardPlotter(Plotter):
    """x is step, y is two values: one for discriminator, one for generator."""

    def __init__(self, title):
        path = os.path.join(os.getcwd(), TENSORBOARD_DIR + title)
        self.writer = SummaryWriter(path)

    def on_new_point(self, label, x, y):
        self.writer.add_scalars(
            main_tag=label,
            tag_scalar_dict={'discriminator': y[0], 'generator':y[1]},
            global_step=x
        )

    def on_new_image(self, label, img_tensor, global_step):
        self.writer.add_image(label + str(global_step), img_tensor, global_step)
