# Artifact Appendix

Paper title: Preserving Target Distributions With Differentially Private Count Mechanisms

Requested Badge(s):
  - [X] **Available**
  - [X] **Functional**
  - [X] **Reproduced**


## Description 

Paper Title: Preserving Target Distributions With Differentially Private Count Mechanisms

Authors: Nitin Kohli and Paul Laskowski

Year: 2026 (Issue 2)

Artifact Description: This Artifact contains python scripts and jupyter notebooks to recreate the Figures (and Table 3) in the paper. 

### Security/Privacy Issues and Ethical Concerns 

Our paper proposes a novel differentially private approach to protect data, specifically tables of counts. For our simulation experiments, we relied on publicly available datasets (as well as simulated data), which are available in the simulations_data folder.


## Basic Requirements 

### Hardware Requirements 

This technical artifact can run on a laptop (No special hardware requirements). The code in this artifact (with the exception of Experiment 5) were run on an Apple M2 laptop with 8 GB of memory to generate the paper's figures. Experiment 5 was run on an Apple 2.3 GHz 8-Core Intel Core i9 laptop with 32 GB 2400 MHz DDR4 (See Experiment 5 below for more details).

### Software Requirements 

This artifact was run on MacOS Tahoe Version 26.1. In general, this artifact should be able to run on any Unix or Linux machine that can support Python and Jupyter Lab. `Python >=3.12` is specified in the `uv.lock` but we note that earlier versions of Python may also work.

The packages used in the artifact can be found in the `uv.lock` and `pyproject.toml` files. Below are the packages listed in the .toml, along with their version numbers.

- "ipykernel>=7.1.0",
- "matplotlib==3.10.7",
- "numpy==2.3.5",
- "opendp==0.14.1",
- "pandas==2.3.3",
- "pywavelets>=1.9.0",
- "scipy==1.16.3",
- "seaborn==0.13.2"

The datasets required the run the artifact can all be found in the simulations_data folder. These include:

- bimodal_hist.csv
- binom_hist_data.csv
- bird_flu.csv
- crime_data.csv
- left-skewed_hist.csv
- right-skewed_hist.csv
- top-inflated_hist.csv
- uniform_hist.csv
- us-public-schools.csv


### Estimated Time and Storage Consumption

The overall compute time to run this artifact is approximately 2 hours. The overall human time to run the artifact is no more than 15 minutes. This is simply the time to open each notebook in Jupyter, select the "Run" tab in top of the screen, and select "Run All Cells". This will then run all the cells, and a human reviewer can observe the output in the notebook.

The space consumed by the artifact is 476 MB. Additional space may be required for machines that do not yet have Jupyter installed.

## Environment 

### Accessibility 

This artifact is publicly accessible at 

```
https://github.com/nitinkohli/fixed_point_privacy
```

### Set up the environment 

This project uses uv to manage the Python environment and dependencies in a reproducible manner (That is, `pyproject.toml` + `uv.lock` specify everything).

Step 0: Navigate to the location on your local machine where you want this repository to reside. Then, clone the project:

```
git clone https://github.com/nitinkohli/fixed_point_privacy.git
cd fixed_point_privacy
```

Step 1: Install uv (if not already installed) using curl:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or with Homebrew:

```
brew install uv
```

Step 2: Create the project environment. From the root of the repository:

```
uv sync
```

This creates a virtual environment (usually .venv/) with the exact versions of all packages pinned in uv.lock.

Step 3: The first time only, register a Jupyter kernel for this project:

```
uv run python -m ipykernel install --user --name fixed_point_privacy
```

Step 4: Start Jupyter Lab

```
uv run jupyter lab
```

If JupyterLab is not installed, please install it first using: `uv pip install jupyterlab`

Step 5: In Jupyter Lab, open a notebook. Navigate to to Kernel -> Change Kernel and choose fixed_point_privacy.

After the kernel is selected, notebooks will run in the same environment that `uv sync` created.

### Testing the Environment 

After opening any of the experiment notebooks, the user should be able to select `Change Kernel...` from the Kernel menu in Jupyter. If `fixed_point_privacy` is available as a Python Kernel, the environment is set up.


## Artifact Evaluation

### Main Results and Claims

#### Main Result 1: Cyclic Laplace results in lower distributional error than existing alternatives as n increases.

The cyclic Laplace mechanism can be used to privatize a distribution of counts with low distribution error, when compared to state of the art distribution privatizers that are not specialized to run on distributions of counts. This can be seen in Section 3 of the paper, along with Figure 3. This can be reproduced in Experiment 2 (see below for more details).

#### Main Result 2: When using a two-stage framework to privatize a table of counts, between the three performance criteria of (1) distribution accuracy, (2) count accuracy, and (3) runtime, there are tradeoffs when adopting a fixed-point constraint. However, we find that at times the tradeoff is favorable: large reductions in distribution error can be achieved with modest reductions in count accuracy and modest increases in runtime.

Section 6.1 of our paper demonstrates distribution accuracy, with Figure 5 displaying low distribution error for our fixed point methods compared to alternative approaches. Section 6.2 of our paper demonstrates count accuracy, with Figure 6 displaying the increase in count error that our two-stage framework induces. Section 6.3 of our paper demonstrates the execution time of our approach, with Figure 7 displaying that (a) a fixed-point method requires larger runtime compared to existing baselines, but (b) the use of a heuristic constructor means the runtime is still competitive with other approaches. This main result is evidenced in Experiments 3 - 5, described below. (Note: Experiment 1 displays the distributional properties of the data used in Experiments 3-5)

### Experiments

#### Experiment 1: Dataset Visualization

See `dataset_visualization.ipynb`. This notebook generates Figures 1 and 4 from the paper. Figure 1 displays an example of the data products we study in this paper: a table of counts, with its corresponding distribution of counts. Figure 4 display the distribution of counts for the 3 main datasets we use in our experiments. These figures mainly provide context for the subsequent experiments.

To run this notebook, open it in Jupyter. In the "Run" tab in top of the screen, select "Run All Cells". This will then run all the cells, and present the plots used in Figures 1 and 4 (Note: all other notebooks are run in a similar manner). The compute time of this is 1 minute, with only 1 minute of human time to press the run button and check the figures.


#### Experiment 2: Distribution Privatizer Comparison

See `distribution_privatizer_comparison.ipynb`. This notebook generates Figure 3 from the paper, which displays the lower distributional error of our cyclic Laplace mechanism compared to other existing alternatives using Wasserstein distance. The notebook also generates plots using KS distance and total deviation instead of Wasserstein distance (demonstrating the efficacy of our method using other measures of distributional accuracy).

This experiment reproduces "Main Result 1: Cyclic laplace results in lower distributional error than existing alternatives as n increases." This can be seen visually in the cell that produced Figure 3. The x-axis displays n, and the y-axis displays the distribution error. As n increases, the cyclic Laplace mechanism has the smallest distribution error compared to the other methods. The compute time of this is 15 minutes, with only a couple of minutes of human time to press the run button and check the figures.


#### Experiment 3: Epsilon Split Experiment

See `epsilon_split_experiment.ipynb`. This notebook contains details of the "rule of thumb" utilized for our experiments in Section 6, which are detailed in Appendix F of our paper. This notebook also generates the values in Table 3 and Figure 8, with additional plots for other datasets. (Note: the .ipynb contains headers which point the reader to which cell holds the Figure / Table in the paper)

This experiment provides inputs to Experiment 4, which will support "Main Result 2". The compute time of this is 20 minutes, with only a couple of minutes of human time to press the run button and check the figure and table values. 

#### Experiment 4: Distribution and Count Error Experiments

See `accuracy_experiment.ipynb`.This notebook generates Figures 5 and 6 from the paper, which display the distribution error and count error of various approaches. These errors are operationalized as expected absolute deviation and Wasserstain distance respectively. This notebook also contains additional plots using other measures of count error and distributional error as robustness checks. (Note: the .ipynb contains headers which point the reader to which cell holds Figures 5 and 6 in the paper)

This experiment reproduces "Main Result 2" and -- taken together -- demonstrates the tradeoff between minimizing count error and distribution error. The compute time of this is 75 minutes, with a few minutes of human time to press the run button and check the figure and table values. 

#### Experiment 5: Execution Time Experiments

See `runtime_experiment.ipynb`. This notebook generates Figure 7 from the paper, which displays the execution time of our approach relative to alternative approaches. 

This experiment reproduces "Main Result 2": a fixed-point method requires larger runtime compared to existing baselines, but the use of a heuristic constructor means the runtime is still competitive with other approaches. Note that the execution time values on the y-axis of this plot may differ if the execution of this notebook occurs on a machine with different hardware than the one it was originally generated on (Apple 2.3 GHz 8-Core Intel Core i9 laptop with 32 GB 2400 MHz DDR4). However, we expect qualitative features of this Figure to remain true regardless even if other hardware is used (in particular, while the y-values for each plot may be lower if a faster computer is used -- or that the y-values for each plot may be higher if a slower computer is used -- the ordering of the curves from left to right is expected to remain consistent). The compute time of this is 20 minutes, with only a couple of minutes of human time to press the run button and check the figure and table values. 

## Limitations 

N/A

## Notes on Reusability 

We believe parts of our code can be used in future attempts to privatize tables of counts.
