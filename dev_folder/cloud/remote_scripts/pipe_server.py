import http.server
import socketserver
import os

# Define the path to your named pipe
PIPE_PATH = '/tmp/synapse_fifo'

class PipeHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        
        with open(PIPE_PATH, 'rb') as pipe:
            while True:
                chunk = pipe.read(4)
                if not chunk:
                    break
                self.wfile.write(chunk)

if __name__ == '__main__':
    PORT = 8005
    
    with socketserver.TCPServer(("", PORT), PipeHandler) as httpd:
        print(f"Serving on port {PORT}")
        httpd.serve_forever()