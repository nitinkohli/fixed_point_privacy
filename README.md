# Preserving Target Distributions With Differentially Private Count Mechanisms

```
Differentially private mechanisms are increasingly used to publish tables of counts, where each entry represents the number of individuals belonging to a particular category. A \textit{distribution of counts} summarizes the information in the count column, unlinking counts from categories. This object is useful for answering a class of research questions, but it is subject to statistical biases when counts are privatized with standard mechanisms. This motivates a novel design criterion we term \textit{accuracy of distribution}.

This study formalizes a two-stage framework for privatizing tables of counts that balances  accuracy of distribution with two standard criteria of accuracy of counts and runtime. In the first stage, a \textit{distribution privatizer} generates an estimate for the true distribution of counts. We introduce a new mechanism, called the cyclic Laplace, specifically tailored to distributions of counts, that outperforms existing general-purpose differentially private histogram mechanisms. In the second stage, a \textit{constructor algorithm} generates a count mechanism, represented as a transition matrix, whose fixed-point is the privatized distribution of counts. We develop a mathematical theory that describes such transition matrices in terms of simple building blocks we call $\epsilon$-scales. This theory informs the design of a new constructor algorithm that generates transition matrices with favorable properties more efficiently than standard optimization algorithms. We explore the practicality of our framework with a set of experiments, highlighting situations in which a fixed-point method provides a favorable tradeoff among performance criteria.
```

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

3. The first time only, register a Jupyter kernel for this project:

`uv run python -m ipykernel install --user --name fixed_point_privacy`

4. Start Jupyter Lab  `uv run jupyter lab`

5. In Jupyter Lab, open a notebook. Go to Kernel → Change Kernel and choose fixed_point_privacy.

After the kernel is selected, notebooks will run in the same environment that uv sync created.
