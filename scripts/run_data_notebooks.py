"""Execute the main data notebooks and retain real tables/figures in their files."""
import asyncio
import json
from pathlib import Path
import sys
import warnings
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]


def main():
    if sys.platform=='win32':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',DeprecationWarning)
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    names=sys.argv[1:] or ['01_EDA.ipynb','02_Preprocessing.ipynb']
    result={}
    for name in names:
        path=ROOT/'notebooks'/name;nb=nbformat.read(path,as_version=4)
        client=NotebookClient(nb,timeout=900,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
        client.execute();nbformat.validate(nb)
        errors=[o for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']
        if errors:raise RuntimeError(f'Notebook execution contains errors: {name}')
        nbformat.write(nb,path)
        result[name]={'code_cells':sum(c.cell_type=='code' for c in nb.cells),
                      'executed_cells':sum(c.cell_type=='code' and c.execution_count is not None for c in nb.cells),
                      'figure_outputs':sum('image/png' in o.get('data',{}) for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[])),
                      'execution_errors':len(errors)}
        print(name,result[name],flush=True)
    out=ROOT/'docs/data/notebook_validation.json';out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
