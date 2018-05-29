import sys
import time
import json
from http.client import HTTPConnection

v_file = sys.argv[-1]
t = time.time()
conn = HTTPConnection('127.0.0.1', 8081, timeout=1000)
conn.request('GET', v_file)
rsp = conn.getresponse()
data_received = json.loads(rsp.read().decode("utf-8"))
print(time.time()-t)
#print(json.dumps(data_received))
with open('pred.json', 'w') as outfile:
    json.dump(data_received, outfile)
conn.close()
