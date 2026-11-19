"""
TerraWatch Interactive Evidence Review Server
Author: Priyanshu Agarwal (Roll No: 23f2003488), IIT Madras BSDA4001 Group 7
Purpose: Serves the evidence review web app and handles audit trail persistence.
"""

import http.server
import json
import os
import socketserver
from pathlib import Path

PORT = 8000
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent

class TerraWatchAuditHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_DIR), **kwargs)

    def do_POST(self):
        if self.path == '/api/save_audit_trail':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                
                # Write to src/app/audit_trail.json
                app_audit_file = APP_DIR / 'audit_trail.json'
                with open(app_audit_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                
                # Also synchronize to outputs/audit_trail.json
                out_dir = PROJECT_ROOT / 'outputs'
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / 'audit_trail.json', 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                response = {'status': 'success', 'message': 'Audit trail saved successfully.'}
                self.wfile.write(json.dumps(response).encode('utf-8'))
                print(f"[AUDIT] Updated audit trail with {len(data.get('records', []))} records.")
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                err_response = {'status': 'error', 'message': str(e)}
                self.wfile.write(json.dumps(err_response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def main():
    os.chdir(APP_DIR)
    with socketserver.TCPServer(("", PORT), TerraWatchAuditHandler) as httpd:
        print("=" * 70)
        print("  TerraWatch Interactive Evidence Review Workbench")
        print("  IIT Madras BSDA4001 DS&AI Lab - Group 7")
        print(f"  Live Server running at: http://localhost:{PORT}")
        print("  Open this URL in any browser to review claims & audit visual evidence.")
        print("  Press Ctrl+C to terminate the server.")
        print("=" * 70)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server gracefully...")

if __name__ == '__main__':
    main()
