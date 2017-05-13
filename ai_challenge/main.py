import argparse
import logging
from ai_challenge import experiments
from utils import parse_clients_args

logger = logging.getLogger(__name__)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('clients',
                            nargs='*',
                            default=['127.0.0.1:10000', '127.0.0.1:10001'],
                            help='Minecraft clients endpoints in the form ip:port')
    arg_parser.add_argument('--exp', '-e',
                            required=True,
                            help='Name of the experiment from experiments package')
    arg_parser.add_argument('--cfg', '-c',
                            required=True,
                            help='Name of the config from config directory')
    arg_parser.add_argument('--logs_path', '-l',
                            help='Relative path to store logs.',
                            default='ai_challenge.logs')
    args = arg_parser.parse_args()
    logging.basicConfig(filename=args.logs_path, level=logging.INFO)
    clients = parse_clients_args(args.clients)
    # run selected experiment with passed clients
    logging.log(
        msg='Starting exp {} with config {} and with clients {}.'
            .format(args.exp, args.cfg, args.clients), level=logging.INFO)

    getattr(experiments, args.exp)(clients, args.cfg)


    logging.log(msg='Exiting main.', level=logging.INFO)


if __name__ == "__main__":
    main()
