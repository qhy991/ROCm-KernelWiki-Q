import urllib.request
import sys

try:
    url = "https://github.com/ROCm/triton/pull/464.patch"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10) as response:
        print(response.read().decode('utf-8'))
except Exception as e:
    print(f"Error: {e}")
