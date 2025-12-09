## Running the notebooks

This project uses uv to manage the Python environment and dependencies reproducibly
(pyproject.toml + uv.lock specify everything).

1. Install uv (if not already installed)
`curl -LsSf https://astral.sh/uv/install.sh | sh`


or with Homebrew:

`brew install uv`

2. Create the project environment

From the root of the repository:

`uv sync`


This creates a virtual environment (usually .venv/) with the exact versions of all packages pinned in uv.lock.

3. Start Jupyter Lab
`uv run jupyter lab`

4. Select the correct kernel in Jupyter

The first time only, register a Jupyter kernel for this project:

`uv run python -m ipykernel install --user --name fixed_point_privacy`


Then in Jupyter Lab:

open a notebook

go to Kernel → Change Kernel

choose fixed_point_privacy

After the kernel is selected, notebooks will run in the same environment that uv sync created.
