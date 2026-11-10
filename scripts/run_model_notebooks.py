"""Execute model notebooks with an explicit interpreter and save outputs in place."""
from pathlib import Path
import argparse,json,subprocess
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--python');args=parser.parse_args()
    for name in ['03_Model_Experiments.ipynb','04_Evaluation.ipynb']:
        path=ROOT/'notebooks'/name;notebook=nbformat.read(path,as_version=4)
        manager=KernelManager(kernel_name='python3')
        if args.python:
            interpreter=Path(args.python).resolve();manager.kernel_spec.argv=[str(interpreter),'-m','ipykernel_launcher','-f','{connection_file}']
        client=NotebookClient(notebook,km=manager,timeout=240,resources={'metadata':{'path':str(ROOT)}})
        try:client.execute()
        finally:
            if manager.has_kernel:manager.shutdown_kernel(now=True)
        nbformat.validate(notebook);nbformat.write(notebook,path)
        print('Executed:',name,flush=True)
        folder=ROOT/'outputs/notebook_previews';folder.mkdir(parents=True,exist_ok=True)
        from nbconvert import HTMLExporter
        html,_=HTMLExporter().from_notebook_node(notebook);(folder/path.with_suffix('.html').name).write_text(html,encoding='utf-8')
if __name__=='__main__':main()
