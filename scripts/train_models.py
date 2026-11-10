"""Launch a training command from its checked-in JSON profile."""
from pathlib import Path
import argparse,json,subprocess,sys

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',choices=['detector','captioner'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--print-command',action='store_true')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    command=[args.python,'-m',f'src.training.{args.model}']
    for key,value in config.items():
        if value is not None:command.extend(['--'+key.replace('_','-'),str(value)])
    print(json.dumps(command))
    if not args.print_command:subprocess.run(command,cwd=Path(__file__).resolve().parents[1],check=True)
