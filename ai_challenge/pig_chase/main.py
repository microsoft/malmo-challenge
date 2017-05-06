import argparse
import ai_challenge.pig_chase.experiments as experiments


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('clients', default=['127.0.0.1:10000', '127.0.0.1:10001'])
    arg_parser.add_argument('--name', '-n', required=True,
                            help='Name of the experiment from experiments package')
    args = arg_parser.parse_args()

    # run selected experiment with passed clients
    getattr(experiments, args.name)(args.clients)


if __name__ == "__main__":
    main()
