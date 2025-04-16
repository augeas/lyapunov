git merge main
rm -r docs
marimo export html-wasm lyapunov.py  -o docs --mode run

