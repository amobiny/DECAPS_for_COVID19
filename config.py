from optparse import OptionParser


parser = OptionParser()

parser.add_option('-e', '--epochs', dest='epochs', default=500, type='int',
                  help='number of epochs (default: 80)')
parser.add_option('--bs', '--batch-size', dest='batch_size', default=1, type='int',
                  help='batch size (default: 16)')
parser.add_option('--df', '--disp_freq', dest='disp_freq', default=10, type='int',
                  help='frequency of displaying the training results (default: 100)')
parser.add_option('--vf', '--val_freq', dest='val_freq', default=50, type='int',
                  help='run validation for each <val_freq> iterations (default: 2000)')
parser.add_option('-j', '--workers', dest='workers', default=0, type='int',
                  help='number of data loading workers (default: 16)')
# For data

parser.add_option('--dr', '--data_root', dest='data_root',
                  default='/home/cougarnet.uh.edu/amobiny/Desktop/CT_DATA/',
                  help='directory of the root data folder')
parser.add_option('--ag', '--add_gan', dest='add_gan', default=True,
                  help='whether to add GAN images or not')
parser.add_option('--gdir', '--gan_dir', dest='gan_dir',
                  default='/home/cougarnet.uh.edu/amobiny/Downloads/fake_ct',
                  help='directory of the deep-fake images')
parser.add_option('--ih', '--img_h', dest='img_h', default=448, type='int',
                  help='input image height (default: 28)')
parser.add_option('--iw', '--img_w', dest='img_w', default=448, type='int',
                  help='input image width (default: 28)')
parser.add_option('--ic', '--img_c', dest='img_c', default=3, type='int',
                  help='number of input channels (default: 1)')
parser.add_option('--nc', '--num_classes', dest='num_classes', default=2, type='int',
                  help='number of classes (default: 10)')

# Pick the loss type
parser.add_option('--lt', '--loss_type', dest='loss_type', default='spread',
                  help='margin, spread, cross-entropy (default: margin)')

# For Margin loss
parser.add_option('--mp', '--m_plus', dest='m_plus', default=0.9, type='float',
                  help='m+ parameter (default: 0.9)')
parser.add_option('--mm', '--m_minus', dest='m_minus', default=0.1, type='float',
                  help='m- parameter (default: 0.1)')
parser.add_option('--la', '--lambda_val', dest='lambda_val', default=0.5, type='float',
                  help='Down-weighting parameter for the absent class (default: 0.5)')

# For Spread loss
parser.add_option('--mmin', '--m_min', dest='m_min', default=0.2, type='float',
                  help='m_min parameter (default: 0.2)')
parser.add_option('--mmax', '--m_max', dest='m_max', default=0.9, type='float',
                  help='m_min parameter (default: 0.9)')
parser.add_option('--nefm', '--n_eps_for_m', dest='n_eps_for_m', default=1, type='int',
                  help='number of epochs to increment the margin (default: 5)')
parser.add_option('--md', '--m_delta', dest='m_delta', default=0.1, type='float',
                  help='margin increment (default: 0.1)')

# For optimizer
parser.add_option('--lr', '--lr', dest='lr', default=0.00001, type='float',
                  help='learning rate (default: 0.001)')
parser.add_option('--beta1', '--beta1', dest='beta1', default=0.5, type='float',
                  help='beta 1 for Adam optimizer (default: 0.9)')

# For DECAPS
parser.add_option('--ni', '--num_iterations', dest='num_iterations', default=3, type='int',
                  help='number of routing iterations (default: 3)')
parser.add_option('--tc', '--theta_c', dest='theta_c', default=0.5, type='float',
                  help='Peekaboo crops region with activation values higher than this (default: 0.5)')
parser.add_option('--td', '--theta_d', dest='theta_d', default=0.7, type='float',
                  help='Peekaboo drops region with activation values higher than this (default: 0.7)')

parser.add_option('--fe', '--cnn_backbone', dest='cnn_backbone', default='resnet',
                  help='densenet, resnet, inception, custom (default: resnet)')
parser.add_option('--A', '--A', dest='A', default=512, type='int',
                  help='number of filters for the conv1x1 layer (default: 512)')
parser.add_option('--B', '--B', dest='B', default=32, type='int',
                  help='number of capsule maps in primarycaps (default: 32)')
parser.add_option('--C', '--C', dest='C', default=32, type='int',
                  help='number of capsule maps in convcaps1 (default: 32)')
parser.add_option('--D', '--D', dest='D', default=32, type='int',
                  help='number of capsule maps in convcaps2 (default: 32)')
# For Options
parser.add_option('--ws', '--weight_share', dest='share_weight', default=True,
                  help='whether to share W among child capsules of the same type (default: True)')

parser.add_option('--ca', '--add_coord', dest='add_coord', default=False,
                  help='whether to add coordinates to the primary capsules output or not (default: False)')
parser.add_option('--ncc', '--norm_coord', dest='norm_coord', default=False,
                  help='whether to normalize the coordinates in [0, 1] or not (default: False)')

parser.add_option('--sd', '--save-dir', dest='save_dir', default='./save',
                  help='saving directory of .ckpt models (default: ./save)')

parser.add_option('--lp', '--load_model_path', dest='load_model_path',
                  default='/home/cougarnet.uh.edu/amobiny/Desktop/DECAPS_COVID/save/'
                          '20200408_195234/models/25635.ckpt',
                  help='path to load a .ckpt model')

options, _ = parser.parse_args()
