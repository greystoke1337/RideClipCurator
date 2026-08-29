# Setup guide (Windows)

This gets the pipeline running on the PC with your RTX 3070 and your footage.
Everything below runs in **PowerShell** (Start menu → type "PowerShell" → open it).
Run commands one at a time and read the output — if something errors, stop
and fix that before moving to the next command.

## 0. Check what you already have

```powershell
python --version
git --version
nvidia-smi
```

- `python --version` should print `3.10` or `3.11` (not 3.13 — some of the ML
  libraries below don't support it yet). If missing, or you get an older/newer
  version, install Python 3.11 from python.org — **during install, tick "Add
  python.exe to PATH"**, that checkbox is the #1 cause of "python not found"
  errors.
- `git --version` should print something. If missing, install Git from
  git-scm.com (defaults are fine).
- `nvidia-smi` should print a table with your RTX 3070 and a driver/CUDA
  version in the top right. If this errors, install/update your NVIDIA
  driver from nvidia.com first — everything GPU-related depends on this.

Close and reopen PowerShell after installing anything, so PATH changes take effect.

## 1. Get the code

```powershell
cd $HOME\Documents
git clone https://github.com/greystoke1337/RideClipCurator.git
cd RideClipCurator
git checkout claude/project-setup-beginners-ezemm0
```

## 2. Create a virtual environment

Keeps this project's Python packages separate from anything else on your machine.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If activation fails with a "running scripts is disabled" error, run this once
(it only relaxes the restriction for your user account), then retry:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

You'll know it worked because your prompt now starts with `(.venv)`. **Every
time you come back to a new PowerShell window to work on this project, run
`.venv\Scripts\Activate.ps1` again first** — it doesn't stay active between
sessions.

## 3. Install ffmpeg

```powershell
winget install ffmpeg
```

Reopen PowerShell afterwards, then check it worked:

```powershell
ffmpeg -version
ffprobe -version
```

(If `winget` itself isn't found, install "App Installer" from the Microsoft
Store first, or grab ffmpeg's Windows build directly from ffmpeg.org and add
its `bin` folder to your PATH manually.)

## 4. Install PyTorch with CUDA support

This has to happen **before** the rest of the requirements, and it's not a
plain `pip install torch` — that gives you a CPU-only build. Check the CUDA
version `nvidia-smi` printed in step 0, then go to pytorch.org, use their
"Get Started" install matrix (Stable / Windows / Pip / Python / your CUDA
version) to get the exact command — it'll look like:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

(`cu121` etc. changes depending on your driver — use whatever pytorch.org's
selector gives you.)

Verify the GPU is actually visible to PyTorch before moving on:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

This must print `True` and `NVIDIA GeForce RTX 3070`. If it prints `False`,
stop here and re-check the CUDA version match — nothing downstream will use
your GPU until this is fixed.

## 5. Install the rest of the Python packages

```powershell
pip install -r requirements.txt
```

This installs Streamlit, OpenCV, TensorFlow (CPU — used only for YAMNet),
Whisper, and their dependencies. It'll take a few minutes.

## 6. Install RAM (Recognize Anything Model)

RAM isn't on PyPI — install it straight from its GitHub repo:

```powershell
pip install git+https://github.com/xinyu1205/recognize-anything.git
```

Then download its checkpoint file (`ram_plus_swin_large_14m.pth`) — search
"RAM++ checkpoint download" on the recognize-anything GitHub repo's README,
which links to the Hugging Face/release download. Put the downloaded file at:

```
RideClipCurator\models\ram_plus_swin_large_14m.pth
```

(create the `models` folder if it doesn't exist — it's gitignored on purpose,
checkpoints are large binary files that don't belong in git).

RAM's `bert.py` imports three functions (`apply_chunking_to_forward`,
`find_pruneable_heads_and_indices`, `prune_linear_layer`) from
`transformers.modeling_utils` — that import path was removed from
`transformers` (moved to `transformers.pytorch_utils`) after RAM's package
was last updated, so tagging will crash with `ImportError: cannot import
name 'apply_chunking_to_forward'` until you patch the installed copy. Open
`.venv\Lib\site-packages\ram\models\bert.py` in a text editor, find this
(around line 39):

```python
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
```

and replace it with:

```python
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
```

(This patches your local venv's copy of a third-party package, so it'll need
redoing if you ever recreate the venv or reinstall RAM.)

## 7. Put your footage where the app expects it

Copy your GoPro and DJI clips (both cameras, all in one flat folder is fine)
into:

```
RideClipCurator\data\raw\
```

## 8. Run the app

```powershell
streamlit run app\streamlit_app.py
```

This opens a browser tab. In the **Process** tab, confirm the folder paths,
then click through the pipeline stage buttons (or "Run all") — the first run
takes a while, and progress prints per stage. Once clips have been processed,
switch to the **Review** tab to browse, filter, and select the ones you want.

## Before running on all ~2-3 hours of footage

Per the project's build approach (`docs/spec.md` §10), run the pipeline
against a small subset first — e.g. copy 15-20 clips spanning both cameras
into `data\raw\` and use `scripts\run_pilot.py` (or just the Process tab) to
sanity-check the tags, steadiness numbers, and dedup clusters before
trusting it at full scale. Weights in `ridecurator/config.py`
(`SCORE_WEIGHTS`) are starting points — expect to retune them once you see
real output.

## Common errors

- **`ffmpeg: command not found` / `FileNotFoundError`** — ffmpeg isn't on
  PATH. Reopen PowerShell after installing it; run `ffmpeg -version` to confirm.
- **`torch.cuda.is_available()` is `False`** — the PyTorch build doesn't
  match your CUDA driver. Redo step 4 with the exact command from pytorch.org.
- **RAM / Whisper very slow** — check they're actually using the GPU (step 4
  above); if `device="cpu"` is selected in the Process tab's model settings,
  switch it to `cuda`.
- **Streamlit shows a blank page or won't start** — make sure `(.venv)` is
  showing in your prompt (step 2) before running `streamlit run`.
