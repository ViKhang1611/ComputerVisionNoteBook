import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient

root = Path(__file__).parent
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = str(root / '.cvip-mpl')
os.environ['IPYTHONDIR'] = str(root / '.cvip-ipython')
os.environ['JUPYTER_RUNTIME_DIR'] = str(root / '.cvip-jupyter-runtime')
os.environ['JUPYTER_PATH'] = str(root / '.cvip-jupyter')
kernel_dir = root / '.cvip-jupyter' / 'kernels' / 'cvip-verify'
kernel_dir.mkdir(parents=True, exist_ok=True)
import json
(kernel_dir / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'CVIP verification', 'language': 'python',
}), encoding='utf-8')
path = root / 'OpenCV_Curriculum_Hoc_Chi_Tiet.ipynb'
nb = nbformat.read(path, as_version=4)
nbformat.validate(nb)
# Use inline figures in the executed deliverable.
for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.source = '%matplotlib inline\n' + cell.source
        break
client = NotebookClient(nb, timeout=180, kernel_name='cvip-verify',
                        resources={'metadata': {'path': str(root)}})
try:
    client.execute()
except Exception:
    nbformat.write(nb, root / 'cvip_execution_debug.ipynb')
    raise
nb.metadata.kernelspec = {'display_name': 'Python 3 (ipykernel)',
                          'language': 'python', 'name': 'python3'}
nbformat.write(nb, path)
figures = 0
for i, cell in enumerate(nb.cells):
    for output in cell.get('outputs', []):
        if output.output_type == 'stream':
            print(f'Cell {i}: {output.text.strip()}')
        if 'image/png' in output.get('data', {}):
            import base64
            figures += 1
            image_dir = root / '.cvip-qa'
            image_dir.mkdir(exist_ok=True)
            (image_dir / f'figure_{figures:02}.png').write_bytes(
                base64.b64decode(output.data['image/png']))
nbformat.validate(nb)
print(f'PASS: all code cells executed, {figures} figures embedded.')
