"""Export a run directory with relative paths, file hashes and a reproducibility receipt."""
from pathlib import Path
import argparse,hashlib,json,platform,subprocess,zipfile

def package_run(run,output):
    run,output=Path(run).resolve(),Path(output).resolve()
    if not run.is_dir():raise ValueError('Run directory does not exist')
    if output==run or run in output.parents:raise ValueError('ZIP must be outside the input run directory')
    files=sorted(path for path in run.rglob('*') if path.is_file())
    if any(path.is_symlink() or not path.resolve().is_relative_to(run) for path in files):raise ValueError('Run may not contain external symlink files')
    root=Path(__file__).resolve().parents[1]
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    manifest={'git_revision':revision,'python':platform.python_version(),'files':[]}
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            relative=path.relative_to(run).as_posix();manifest['files'].append({'path':relative,'sha256':digest,'bytes':path.stat().st_size})
            archive.write(path,relative)
        archive.writestr('_export_manifest.json',json.dumps(manifest,indent=2))
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    print(package_run(args.run,args.output))
