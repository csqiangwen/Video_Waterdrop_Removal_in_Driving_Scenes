from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # Data option
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--n_frames', type=int, default=10, help='the number of input frames')
        self.parser.add_argument('--vid_dataroot', type=str, default='/disk1/wenqiang/Documents/data/Waterdrop/')
        self.parser.add_argument('--data_type', type=str, default='real')

        # model option
        self.parser.add_argument('--gpu_ids', type=str, default='2', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints_waterdrop', help='models are saved here')
        self.parser.add_argument('--name', type=str, default='radindrop', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--which_iter', type=str, default=0, help='which epoch to load? set to latest to use latest cached model')

        self.isTrain = False
