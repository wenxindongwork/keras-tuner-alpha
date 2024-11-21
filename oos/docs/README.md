# jaxloop ReadTheDocs Site

To build:

```
# Install Sphinx deps.
pip install -r oss/docs/requirements.txt

make -C oss/docs html

# Preview locally
<!-- python -m http.server -d /tmp/jaxloop_docs/html -->
Go to http://127.0.0.1:8000
```