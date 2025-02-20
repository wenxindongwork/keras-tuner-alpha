# Kithara ReadTheDocs Site

To build:

```
# Install Sphinx deps.
pip install -r docs/requirements.txt

make -C docs html

# Preview locally
<!-- python -m http.server -d /tmp/jaxloop_docs/html -->
Go to http://127.0.0.1:8000
```