from torch.utils import tensorboard

class Visualizer(object):
    def __init__(self, log_dir):
        super(Visualizer, self).__init__()
        self.tb = tensorboard.writer.SummaryWriter(log_dir)

    def log_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        """
        main_tag (string) – The parent name for the tags
        tag_scalar_dict (dict) – Key-value pair storing the tag and corresponding values
        global_step (int) – Global step value to record
        """
        self.tb.add_scalars(main_tag, tag_scalar_dict, global_step)

    def __del__(self):
        self.tb.close()
        super(Visualizer, self).__del__()