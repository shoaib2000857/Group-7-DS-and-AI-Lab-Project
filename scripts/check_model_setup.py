"""Check model assets and separate Python environments without downloading weights."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.inference.preflight import inspect_config

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/pipeline.json')
    parser.add_argument('--environments',action='store_true')
    parser.add_argument('--strict',action='store_true')
    args=parser.parse_args()
    result=inspect_config(json.loads(Path(args.config).read_text()),environments=args.environments)
    print(json.dumps(result,indent=2))
    if args.strict and not result['ready']:raise SystemExit(1)

if __name__=='__main__':main()
