from vid2player.motion_vae.config import *
from vid2player.motion_vae.base import MotionVAEModel 
from argparse import ArgumentParser
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
parser = ArgumentParser()
parser.add_argument('-e', '--exp', action='store', dest='exp_list',
                    type=str, nargs='*', default=['vx.x.x'],
                    help="Examples: -e v8.3.1.0")
parser.add_argument('-p', '--pre-run', action='store_true')
parser.add_argument('--no_log', action='store_true')
parser.add_argument('--continue_train', action='store_true',
                    help='Resume the latest checkpoint, including optimizer and completed epochs.')
parser.add_argument('-t', '--run_test', action='store_true')

parser.add_argument('-to', '--test_only', '--test-only', action='store_true',
                    help='Record a named ground-truth versus MVAE reconstruction MP4.')
parser.add_argument('--name', help='Manifest video/sequence name, source key, or unique export filename stem.')
parser.add_argument('--resample', type=int, help='Reset conditioning pose and root from ground truth every N predictions.')
parser.add_argument('-i', '--interactive', action='store_true')
parser.add_argument('-n', '--nframes', action='store', type=int, default=None,
                    help='Maximum recorded frames (test_only defaults to the full sequence).')
parser.add_argument('-a', '--nactors', action='store', type=int, default=5)
parser.add_argument('--ntests', action='store', type=int, default=1)
parser.add_argument('--suffix', action='store', type=str, default='')
parser.add_argument('--record_html', action='store_true')
parser.add_argument('--record_cv', action='store_true')

args = parser.parse_args()
if args.test_only and not args.name:
    parser.error('--test_only requires --name')
if args.resample is not None and args.resample <= 0:
    parser.error('--resample must be a positive integer')
if (args.name is not None or args.resample is not None) and not args.test_only:
    parser.error('--name and --resample require --test_only')
if args.nframes is not None and args.nframes < 2:
    parser.error('--nframes must be at least 2')
if args.test_only and (args.interactive or args.record_html):
    parser.error('--test_only records MP4 and PNG directly; omit --interactive/--record_html')


if __name__ == '__main__':

    for version in args.exp_list:
        # load option 
        opt = MotionVAEOption()
        opt.load(version)
        if args.continue_train:
            opt.continue_train = True
        if args.pre_run:
            opt.quick_load = True
            opt.n_epochs = 5
            opt.n_epochs_decay = 5
            opt.nseqs = 1000
        opt.no_log = args.no_log

        if not args.test_only:
            motion_vae = MotionVAEModel(opt)
            print(motion_vae.model)
            opt.print()
            motion_vae.train()

        if args.test_only:
            from vid2player.motion_vae.reconstruction import record_reconstruction
            record_reconstruction(opt, args.name, resample=args.resample,
                                  nframes=args.nframes, suffix=args.suffix)

        # Preserve the random-generation preview after training.
        elif args.run_test:
            from vid2player.motion_vae.test import test_motion_vae_randomwalk

            print("Running test ... ")
            opt.test_only = True
            opt.infer_racket = True
            test_motion_vae_randomwalk(opt, 
                same_init_state=True, 
                num_test=args.ntests, 
                num_runner=args.nactors, 
                result_dir_suffix=args.suffix,
                nframes=args.nframes or 1000,
                interactive=args.interactive,
                record_html=args.record_html,
                record_cv=args.record_cv,
            )
