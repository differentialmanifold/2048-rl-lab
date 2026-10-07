"""Legacy command bridge; new runs use algorithms.latent_imagination.pipeline."""
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == 'imagine-ppo':
        from algorithms.latent_imagination.train import main as train
        return train(args[1:])
    if args and args[0] == 'repair-world':
        from algorithms.latent_imagination.world.trajectories import main as trajectories
        if trajectories(args[1:]) is False:
            raise SystemExit(2)
        return True
    from algorithms.latent_imagination.world.base import main as base
    return base(args)


if __name__ == '__main__':
    main()
