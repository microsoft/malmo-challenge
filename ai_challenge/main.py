import argparse
import logging
from ai_challenge import experiments
from utils import parse_clients_args


def main():

    logging.basicConfig(filename='pig_chase.log', level=logging.DEBUG)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('clients', nargs='*',
                            default=['127.0.0.1:10000', '127.0.0.1:10001'],
                            help='Minecraft clients endpoints (ip(:port)?)+')
    arg_parser.add_argument('--exp', '-e', required=True,
                            help='Name of the experiment from experiments package')
    args = arg_parser.parse_args()
    clients = parse_clients_args(args.clients)
    # run selected experiment with passed clients
    getattr(experiments, args.exp)(clients)

if __name__ == "__main__":
    main()
