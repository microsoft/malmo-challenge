import re
import subprocess
import argparse


def change_command(exp_name, model, reps):
    with open('template.yml', 'r') as handle:
        content = handle.read()
        full_command = '  && '.join([r'python main.py malmo1:10000 malmo2:10000 '
                                               '-e {} -m {}'.format(exp_name,
                                                                    model)] * reps) + '''"'''
        modified = re.sub(r'python(.*)', full_command,
                          content)
    with open('docker-compose.yml', 'w') as handle:
        handle.write(modified)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--exp', '-e', required=True,
                            help='Name of the experiment from experiments package')
    arg_parser.add_argument('--model', '-m', required=True,
                            help='Name of the model from chainerrl')
    arg_parser.add_argument('--reps', '-r', required=True,
                            help='Repetitions')
    args = arg_parser.parse_args()
    change_command(args.exp, args.model, int(args.reps))
    subprocess.call('docker-compose up', shell=True)


if __name__ == '__main__':
    main()
