from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # Data option
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
        self.parser.add_argument('--loadsize', type=int, default=256, help='input batch size')
        self.parser.add_argument('--vid_dataroot', type=str, default='/disk1/wenqiang/Documents/data/RainDrop/')
        self.parser.add_argument('--n_frames', type=int, default=5, help='the number of input frames')
        self.parser.add_argument('--shuffle', action='store_true', help='dataloader option')

        # model option
        self.parser.add_argument('--gpu_ids', type=str, default='2', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_freq', type=int, default=1000, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints_raindrop', help='models are saved here')
        self.parser.add_argument('--name', type=str, default='radindrop', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--which_iter', type=str, default=0, help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=1e5, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--gradient_clipping', type=int, default=10, help='gradient clipping for generator')
        self.parser.add_argument('--epoch_count', type=int, default=100, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')

        self.isTrain = True
