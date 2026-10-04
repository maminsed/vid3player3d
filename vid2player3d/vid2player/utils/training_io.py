"""CPU-only output helpers for controller training."""
import glob
import json
import os
import numbers


def check_training_output(config, args, model_dir):
    if not config.get('protect_existing_checkpoints', False) or not args.train:
        return
    if args.resume:
        if not args.checkpoint:
            raise ValueError('Resuming requires --checkpoint, for example --checkpoint latest.')
        return
    existing = glob.glob(os.path.join(model_dir, '*.pth'))
    if existing:
        raise FileExistsError(
            'Existing checkpoints in {}. Use a new --results_dir, or explicitly '
            'resume with --resume --checkpoint latest.'.format(model_dir))


def write_epoch_metrics(writer, metrics, frame):
    """Write controller scalars to the shared TensorBoard event stream."""
    for key, value in metrics.items():
        if hasattr(value, 'item'):
            value = value.item()
        if isinstance(value, numbers.Real):
            writer.add_scalar(key, value, frame)


def write_run_config(filename, args, config):
    with open(filename, 'w') as stream:
        json.dump({'arguments': vars(args), 'config': config}, stream,
                  indent=2, default=str)
