# QEterminal: Quantum Structure Studio

This folder contains the Python application and its runtime modules. No Quantum ESPRESSO executable is required for viewing or editing structures and cube files.

## Windows setup

Open PowerShell in this folder and run these commands once to create the environment and install the required libraries:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then launch the app from the activated environment:

```powershell
python main.py
```

For later launches, open a terminal in this folder, activate the environment with `.\.venv\Scripts\Activate.ps1`, then run `python main.py`. In Command Prompt, activate with `.venv\Scripts\activate.bat` instead.

If you already installed all libraries into the Python interpreter you intend to use, you can run `python main.py` directly without creating a virtual environment. The app uses PySide6, ASE, NumPy, Matplotlib, SciPy, PyVista, and PyVistaQt for the GUI and 3D view. The requirements also include the libraries needed by the built-in analysis tools and SSH terminal.

A structure file may be opened at startup by passing its path, for example:

```powershell
python main.py "C:\path\to\structure.cif"
```

For a computer without a working OpenGL driver, install the dependencies as above and launch the Matplotlib renderer with:

```powershell
python main.py --renderer matplotlib
```

## Optional OVITO renderer

The default install uses PyVista / VTK. If you specifically want the OVITO renderer, install its compatible versions into the activated environment:

```powershell
python -m pip install "PySide6==6.10.3" "ovito>=3.16,<3.17"
python main.py --renderer ovito
```
