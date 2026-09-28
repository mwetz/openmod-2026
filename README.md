# PIPS-IPM++ workshop

A hands-on session on solving energy system models with
[PIPS-IPM++](https://gitlab.com/pips-ipmpp/pips-ipmpp), a parallel interior-point solver for linear
programs whose constraint matrix is doubly-bordered block-diagonal, and its Python interface
[`pipsipmpp`](https://pypi.org/project/pipsipmpp/).

---

## Setup

You can either use [uv](https://docs.astral.sh/uv/) or [pixi](https://pixi.sh) for the fastet setup using 
the following commands to start the Jupyter notebooks. If you are on a Linux machine and want to run the 
solver install it with the `solver` extra, otherwise you can just leave it out.


Setup for uv:
```console
$ git clone
$ cd notebooks 
$ uv run --extra solver jupyter lab 
```

Setup for pixi
```console
$ git clone
$ pixi install
$ pixi run -e solver lab
```

If you are using `WSL`, run the start command inside WSL and open the `http://localhost:8888/...` link it prints in
the browser of your choice.

If you prefer using VS Code directly, you first need to install the Jupyter kernel and activate it using
`Select kernel` on the top right.

Setting up the Jupyter kernel:
```console
$ uv run python -m ipykernel install --user --name pipsipmpp-workshop \
    --display-name "PIPS-IPM++ workshop"
```


### Optional parts

| extra | what it adds | uv | pixi |
|---|---|---|---|
| `solver` | MPI (`mpi4py` and an Open MPI runtime), which solving needs even on one process. Linux x86_64 only; on other systems the extra is empty | `--extra solver` | `-e solver` |
| `pyoptinterface` | the PyOptInterface backend for 01c. It compiles from source, so uv needs a C++ compiler on your machine | `--extra pyoptinterface` | `-e pyoptinterface` (includes `solver`) |

With uv, pass the same `--extra` flags to every `uv sync` and `uv run`: a plain `uv sync` removes
the extras again.

### What runs where

Everything installs on Linux, macOS and Windows. Two things do not run everywhere:

| | Linux x86_64 and WSL | macOS | Windows |
|---|---|---|---|
| build models, export them, visualize and judge their structure | yes | yes | yes |
| find a structure with the regex method | yes | yes | yes |
| find a structure with the hypergraph method (mt-KaHyPar) | yes | yes | no, use WSL |
| solve with PIPS-IPM++ | yes, with the `solver` extra | no, use WSL or a Linux cluster | no, use WSL or a Linux cluster |

On every machine the notebooks run top to bottom: cells that need the solver begin with
`%%requires solver` and are skipped with a note instead of failing. Each notebook puts the part
that runs everywhere first and the solve at the end, so you can do most of the work locally and
run the skipped cells on WSL or a Linux cluster, where the same install works unchanged.

The first cell of every notebook, `workshop.check()`, tells you what applies to your machine.

---

## The concept behind PIPS-IPM++

PIPS-IPM++ does not look for structure in your model. You state it, by giving every variable a
block number:

```
0      root   shared by every block   investment capacity, a shared budget
1..N   leaf   local to one block      dispatch in one hour, one region, one scenario
```

Rows touching only leaf k are local to it; rows touching several leaves become linking rows. The
leaves go to different MPI processes and are worked on at the same time.

```
        root      leaf 1    leaf 2    leaf 3         variables
      +--------+---------+---------+---------+
 root |   A0   |         |         |         |   constraints on root variables only
      +--------+---------+---------+---------+
leaf 1|   A1   |   B1    |         |         |   constraints local to block 1
      +--------+---------+---------+---------+
leaf 2|   A2   |         |   B2    |         |   constraints local to block 2
      +--------+---------+---------+---------+
leaf 3|   A3   |         |         |   B3    |   constraints local to block 3
      +--------+---------+---------+---------+
 link |  Bl0   |   Bl1   |   Bl2   |   Bl3   |   constraints spanning several blocks
      +--------+---------+---------+---------+
       ^ left border                  ^ diagonal blocks, one per rank
```

Every notebook uses the same sample model (invest in wind and gas, dispatch them hour by hour,
stay under an emission budget), so you can compare them side by side. Its data lives in
`notebooks/sample_data.py`.

---

## The notebooks

| step | notebook | content |
|---|---|---|
| 00 | [Start here](notebooks/00-getting-started.ipynb) | technical checks, PIPS-IPM++ and its block structure, the sample model |
| 01 | pick one, ideally the framework you already know | build the sample model, export it, visualize its matrix, change the model and watch the structure change; then solve |
| | [01a linopy](notebooks/01a-linopy.ipynb) | also the way in for PyPSA users |
| | [01b Pyomo](notebooks/01b-pyomo.ipynb) | |
| | [01c PyOptInterface](notebooks/01c-pyoptinterface.ipynb) | needs the `pyoptinterface` extra |
| | [01d GAMSPy](notebooks/01d-gamspy.ipynb) | |
| 02 | [PyPSA](notebooks/02-pypsa.ipynb) | the same on a PyPSA network, builds on 01a |
| 03 | [Structure detection](notebooks/03-structure-detection.ipynb) | find a structure in a model file: a REMix model with regex, then hypergraph |
| 04 | [Deep dive](notebooks/04-deep-dive.ipynb) | your own matrices, the distributed layout, solving on remote machines |

---

## If something goes wrong

| Symptom | Cause and fix |
|---|---|
| `workshop.check()` says the solver extra is not installed | on Linux: start JupyterLab with `uv run --extra solver jupyter lab` or `pixi run -e solver lab` |
| `workshop.check()` says the solver is built for Linux x86_64 only | expected on macOS and Windows; run the solve cells on WSL or a Linux cluster |
| `No module named 'mpi4py'` from a solve | the same: the `solver` extra is missing |
| `pyoptinterface  not installed` in 01c | start JupyterLab with the `pyoptinterface` extra, see Setup |
| `'NoneType' object has no attribute 'solve'` (Pyomo) | Pyomo's way of saying the solver is not available; see 01b |
| `Variable '...' has no PIPS-IPM++ block annotation` (Pyomo) | every variable needs an entry in the `pips_block` Suffix; `0` for the root |
| `No solution found ... only linopy<0.7.0 is available` from uv | your uv cache holds a copy of the linopy fork from before it was tagged: `uv cache clean`, then `uv sync` |
| edited a model, nothing happened | the model lives in a file: run the `%%writefile` cell and the `%run` cell again |
| edited `sample_data.py`, nothing happened | Python keeps the module it imported first: restart the kernel |
| `PIPS requires at least one leaf block per rank` | more processes than leaf blocks: use fewer processes, or more blocks |
| PyPSA behaves nothing like the notebook | `pypsa.__version__` must be 1.2.x; reinstall with `uv sync` or `pixi install` |

`pipsipmpp.libpips_info()` is the first thing to print when a solve behaves oddly: which library
was loaded, from where, for which MPI, and which version.

---

## What is in here

- `pyproject.toml`: the single install definition. `uv.lock` and `pixi.lock` pin every package.
- `notebooks/`: the notebooks, plus three helper modules.
  - `workshop.py` has the platform check, `%%requires`, `visualize` and launching MPI; 
  - `sample_data.py` the data of the sample model;
  - `pipsipmpp_interface.py` the GAMSPy bridge used by 01d.
- `notebooks/data/`: created by the notebooks for downloaded and exported models.

Documentation for the interface: <https://pips-ipmpp.gitlab.io/pipsipmpppy>.
