import argparse
import logging
from ai_challenge import experiments
from utils import parse_clients_args

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(filename='ai_challenge.log', level=logging.INFO)
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('clients', nargs='*',
                            default=['127.0.0.1:10000', '127.0.0.1:10001'],
                            help='Minecraft clients endpoints (ip(:port)?)+')
    arg_parser.add_argument('--exp', '-e', required=True,
                            help='Name of the experiment from experiments package')
    args = arg_parser.parse_args()
    clients = parse_clients_args(args.clients)
    # run selected experiment with passed clients
    logging.log(msg='Starting exp {} with clients {}.'.format(args.exp, args.clients),
                level=logging.INFO)
    getattr(experiments, args.exp)(clients)
    logging.log(msg='Exiting main.', level=logging.INFO)


if __name__ == "__main__":
    main()
