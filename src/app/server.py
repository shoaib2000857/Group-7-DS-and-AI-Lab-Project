"""Local claim review for TerraWatch pipeline reports and their evidence assets."""
import argparse
import hashlib
import json
import mimetypes
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

APP=Path(__file__).resolve().parent
ROOT=APP.parents[1]


def load_reports(folders):
    records={};assets={}
    def register(path,folder,run_id):
        path=Path(path)
        if not path.is_absolute():path=folder/path
        path=path.resolve()
        if not path.is_file() or path.suffix.lower() not in {'.png','.jpg','.jpeg','.webp'}:
            raise ValueError(f'Missing or unsupported image: {path}')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        key=f'{run_id}-{digest}'
        assets[key]=path
        return {'url':f'/asset/{key}','sha256':digest}
    for folder in folders:
        folder=Path(folder).resolve();raw=(folder/'report.json').read_bytes();run=json.loads(raw)
        run_id=hashlib.sha256(raw).hexdigest()
        sidecar=folder/'review_inputs.json'
        if sidecar.exists():
            inputs=json.loads(sidecar.read_text());images=inputs['images'];mask=inputs['mask']
        else:
            images=json.loads((folder/'detector_request.json').read_text())['images']
            response=json.loads((folder/'detector_response.json').read_text())
            mask=folder/'detector_response.mask.png'
            if not mask.exists():mask=Path(response['mask_path'])
        originals=[register(path,folder,run_id) for path in images]
        if [image['sha256'] for image in originals]!=run['input_sha256']:
            raise ValueError(f'Input image hashes disagree with {folder}/report.json')
        mask_asset=register(mask,folder,run_id)
        for claim in run['claims']:
            record_id=f"{run_id}:{claim['claim_id']}"
            crops=[]
            for path in claim['evidence']['crop_paths']:
                portable=folder/'crops'/Path(path.replace('\\','/')).name
                crops.append(register(portable if portable.exists() else path,folder,run_id))
            records[record_id]={'id':record_id,'run_id':run_id,'execution_mode':run['execution_mode'],
                'pair_id':run['pair_id'],'claim':claim,'images':originals,'mask':mask_asset,'crops':crops,
                'detector':run['model_metadata'].get('detector',{})}
    if not records:raise ValueError('No claim records found in the selected runs')
    return records,assets


class ReviewStore:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute('CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY, claim_id TEXT NOT NULL, event TEXT NOT NULL)')

    def list(self,record_id):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            return [json.loads(row[0]) for row in connection.execute('SELECT event FROM reviews WHERE claim_id=? ORDER BY id',(record_id,))]

    def append(self,record,value):
        if not isinstance(value,dict):raise ValueError('Expected a review object')
        reviewer=value.get('reviewer');notes=value.get('notes','');correction=value.get('correction','')
        decision=value.get('decision');flagged=value.get('flagged',False)
        if not isinstance(reviewer,str) or not reviewer.strip() or len(reviewer)>120:raise ValueError('Enter a reviewer name (up to 120 characters)')
        if decision not in (None,'accept','reject','defer') or type(flagged) is not bool:raise ValueError('Invalid decision or flag')
        if any(not isinstance(s,str) or len(s)>2000 for s in (notes,correction)):raise ValueError('Notes and corrections are limited to 2,000 characters')
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            row=connection.execute('SELECT event FROM reviews WHERE claim_id=? ORDER BY id DESC LIMIT 1',(record['id'],)).fetchone()
            previous=json.loads(row[0]) if row else {}
            event={'claim_id':record['claim']['claim_id'],'run_id':record['run_id'],
                'reviewer':reviewer.strip(),'timestamp':datetime.now(timezone.utc).isoformat(),
                'decision':decision,'flagged':flagged,'correction':correction.strip(),'notes':notes.strip(),
                'original_outcome':record['claim']['decision']['outcome'],
                'previous_review':{key:previous.get(key) for key in ('decision','flagged','correction','reviewer','timestamp')} if previous else None}
            connection.execute('INSERT INTO reviews(claim_id,event) VALUES (?,?)',(record['id'],json.dumps(event)))
        return event


def make_server(records,assets,store,port):
    class Handler(BaseHTTPRequestHandler):
        def send_body(self,status,body,mime):
            self.send_response(status);self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)

        def send_json(self,status,value):
            self.send_body(status,json.dumps(value).encode(),'application/json')

        def do_GET(self):
            path=urlsplit(self.path).path
            if path=='/':self.send_body(200,(APP/'review.html').read_bytes(),'text/html; charset=utf-8')
            elif path=='/api/reports':
                self.send_json(200,[{**record,'reviews':store.list(key)} for key,record in records.items()])
            elif path.startswith('/asset/') and path[7:] in assets:
                asset=assets[path[7:]];self.send_body(200,asset.read_bytes(),mimetypes.guess_type(asset)[0])
            else:self.send_json(404,{'error':'Resource unavailable'})

        def do_POST(self):
            if self.path!='/api/review':self.send_json(404,{'error':'Resource unavailable'});return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16384:raise ValueError('Invalid review size')
                if self.headers.get_content_type()!='application/json':raise ValueError('Use a JSON review body')
                value=json.loads(self.rfile.read(length))
                if not isinstance(value,dict) or not isinstance(value.get('id'),str) or value['id'] not in records:
                    raise ValueError('Unknown claim record')
                event=store.append(records[value['id']],value)
            except (ValueError,UnicodeError) as error:
                self.send_json(400,{'error':str(error)});return
            self.send_json(201,event)
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--database',type=Path,default=ROOT/'outputs/pipeline_runs/review_events.sqlite3')
    parser.add_argument('--port',type=int,default=8010)
    args=parser.parse_args();records,assets=load_reports(args.runs)
    with make_server(records,assets,ReviewStore(args.database),args.port) as server:
        print(f'TerraWatch evidence review at http://127.0.0.1:{server.server_port}',flush=True)
        server.serve_forever()

if __name__=='__main__':main()
