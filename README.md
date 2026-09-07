# esstec-ca-website
Website for esstec.ca

Edit `index.html`, `assets/site.css`, and `assets/coin.js` directly; no build step is required.

Preview locally with `python3 -m http.server 8000`, then open `http://localhost:8000`.

Run the animation lifecycle regression checks with `node --test tests/coin.test.cjs`.
These use mocked browser/GPU APIs; verify visual rendering in a WebGPU-capable browser too.
