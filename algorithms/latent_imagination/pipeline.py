"""Latent Imagination RL: build a world, then learn through imagined games."""
import argparse
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == 'world':
        from .world.pipeline import main as world
        if world(args[1:]) is False:
            raise SystemExit(2)
        return True
    if args and args[0] == 'imagine':
        from .train import main as imagine
        return imagine(args[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('world', 'imagine'))
    parser.parse_args(args)


if __name__ == '__main__':
    main()
