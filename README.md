# Preserving Target Distributions With Differentially Private Count Mechanisms

This repository contains the source code for the experimental results in "Preserving Target Distributions With Differentially Private Count Mechanisms" by Nitin Kohli and Paul Laskowski, accepted at Proceeding on Privacy Enhancing Technologies, 2026.

> Differentially private mechanisms are increasingly used to publish tables of counts, where each entry represents the number of individuals belonging to a particular category. A distribution of counts summarizes the information in the count column, unlinking counts from categories. This object is useful for answering a class of research questions, but it is subject to statistical biases when counts are privatized with standard mechanisms. This motivates a novel design criterion we term accuracy of distribution.

> This study formalizes a two-stage framework for privatizing tables of counts that balances  accuracy of distribution with two standard criteria of accuracy of counts and runtime. In the first stage, a distribution privatizer generates an estimate for the true distribution of counts. We introduce a new mechanism, called the cyclic Laplace, specifically tailored to distributions of counts, that outperforms existing general-purpose differentially private histogram mechanisms. In the second stage, a constructor algorithm generates a count mechanism, represented as a transition matrix, whose fixed-point is the privatized distribution of counts. We develop a mathematical theory that describes such transition matrices in terms of simple building blocks we call epsilon-scales. This theory informs the design of a new constructor algorithm that generates transition matrices with favorable properties more efficiently than standard optimization algorithms. We explore the practicality of our framework with a set of experiments, highlighting situations in which a fixed-point method provides a favorable tradeoff among performance criteria.

## Environment Setup

This project uses uv to manage the Python environment and dependencies in a reproducible manner (`pyproject.toml` + `uv.lock` specify everything).

Step 0: Navigate to the location on your local machine where you want this repository to clone. Then, clone the project:

```
git clone nitinkohli/fixed_point_privacy
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

Step 5: In Jupyter Lab, open a notebook. Navigate to to Kernel -> Change Kernel and choose fixed_point_privacy.

After the kernel is selected, notebooks will run in the same environment that `uv sync` created.

## Running the Notebooks

This repository contains 5 notebooks that can be run indepedently of one another (described below). Once the environment is setup (as described in Steps 1-5 above), each notebook can be run by executing the cells in order. 

- `accuracy_experiment.ipynb`: This notebook generates Figures 5 and 6 from the paper, which display the count error and distributional error of various approaches. These errors are opertationalized as expected absolute deviation and Wasserstain distance respectively. This notebook also contains additional plots using other measures of count error and distributional error as robustness checks.
- `dataset_visualization.ipynb`: This notebook generates Figures 1 and 4 from the paper. Figure 1 displays an example of the data products we study in this paper: a table of counts, with its corresponding distribution of counts. Figure 4 display the distribution of counts for the 3 main datasets we use in our experiments.
- `distribution_privatizer_comparison.ipynb`: This notebook generates Figure 3 from the paper, which displays the lower distributional error of our cyclic Laplace mechanism compared to other existing alternatives using Wasserstein distance. The notebook also generates plots using KS distance and total deviation instead of Wasserstein distance (demonstrating the efficacy of our method using other measures of distributional accuracy).
- `epsilon_split_experiment.ipynb`: This notebook contains details of the "rule of thumb" utilized for our experiments, which are detailed in Appendix F of our paper. This notebook also generates the values in Table 3 and Figure 8, with additional plots for other datasets.
- `runtime_experiment.ipynb`: This notebook generates Figure 7 from the paper, which displays the execution time of our approach relative to alternative approaches. Note that the execution time values on the y-axis of this plot may differ if the execution of this notebook occurs on a machine with different hardware than the one it was originally generated on (see ARTIFACT-APPENDIX.md for hardware details).

These notebooks utilize the python packages in the `uv.lock` file, as well as other functionality within the `pets_utilities.py` and `plotting_utilities.py` files.
