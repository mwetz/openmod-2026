"""Helpers shared by the workshop notebooks.

Nothing in here is part of pipsipmpp. It only keeps the notebooks free of
boilerplate: finding out what this machine can do, skipping the cells it cannot
run, looking at an exported matrix, and launching a script under MPI.

    import workshop
    workshop.check()                  # what works here, and why not

    workshop.visualize("data/toy")    # draw data/toy.amat.parquet and summarise it
    workshop.pipstools("annotate", "model.mps.gz", "--blocks", 8)

    %%requires solver                 # first line of a cell that needs PIPS-IPM++
    %%requires mtkahypar              # ... or the hypergraph partitioner
    %%requires pyoptinterface         # ... or the optional PyOptInterface backend
    %%requires pyoptinterface solver  # ... or both

    workshop.mpirun("script.py", ranks=4)
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = ["CAN_PARTITION", "CAN_SOLVE", "check", "mpirun", "pipstools", "structure",
           "visualize"]

_SOLVER_PLATFORM = sys.platform == "linux" and platform.machine() in ("x86_64", "AMD64")


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _solver_status() -> tuple[bool, str]:
    import pipsipmpp

    try:
        pipsipmpp.require_solver()
    except ImportError as err:
        if not _SOLVER_PLATFORM:
            return False, ("the solver is built for Linux x86_64 only. "
                           "Run these cells on WSL or a Linux cluster.")
        if not _installed("mpi4py"):
            return False, ("the solver extra is not installed. Restart JupyterLab with "
                           "'uv run --extra solver jupyter lab' or 'pixi run -e solver lab'.")
        return False, str(err).splitlines()[0]
    return True, ""


CAN_SOLVE, _WHY_NO_SOLVER = _solver_status()
CAN_PARTITION = _installed("mtkahypar")

_REASONS = {
    "solver": (
        lambda: CAN_SOLVE,
        lambda: f"Skipped, this cell runs PIPS-IPM++ and {_WHY_NO_SOLVER}",
    ),
    "pyoptinterface": (
        lambda: _installed("pyoptinterface"),
        lambda: ("Skipped, this cell needs the optional PyOptInterface install. Start "
                 "JupyterLab with 'uv run --extra pyoptinterface jupyter lab' (add "
                 "'--extra solver' to solve) or 'pixi run -e pyoptinterface lab'."),
    ),
    "mtkahypar": (
        lambda: CAN_PARTITION,
        lambda: ("Skipped, this cell needs mt-KaHyPar for hypergraph partitioning, which "
                 "is not published for Windows. The regex method needs nothing extra, "
                 "and WSL has the full toolchain."),
    ),
}

_PACKAGES = ["pipsipmpp", "pipstools", "linopy", "pyomo", "gamspy", "pypsa", "pyoptinterface",
             "mpi4py"]


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def check() -> None:
    """Print what this machine can and cannot do in the workshop."""
    print(f"{platform.system()} {platform.machine()}, Python {sys.version.split()[0]}")
    print()
    for package in _PACKAGES:
        print(f"  {package:15s} {_version(package)}")
    print()
    rows = [
        ("build models, export and visualize them", True, ""),
        ("find a structure with the regex method", True, ""),
        ("find a structure with the hypergraph method (mt-KaHyPar)", CAN_PARTITION,
         "mt-KaHyPar is not published for Windows. Use the regex method, or WSL."),
        ("solve with PIPS-IPM++", CAN_SOLVE, _WHY_NO_SOLVER),
    ]
    for what, ok, why in rows:
        print(f"  {'yes' if ok else 'no ':3s}  {what}")
        if not ok:
            print(f"        {why}")
    if not CAN_SOLVE:
        print()
        print("Cells that start with  %%requires solver  are skipped on this machine.")


def _bin(name: str) -> str:
    """A command from this environment first, so the bundled MPI wins over any other."""
    exe = Path(sys.executable)
    for local in (exe.with_name(name), exe.with_name(f"{name}.exe"),
                  exe.parent / "Scripts" / f"{name}.exe"):
        if local.exists():
            return str(local)
    return shutil.which(name) or name


def pipstools(*args, cwd: str | Path = "data", quiet: bool = False) -> subprocess.CompletedProcess:
    """Run the pipstools command line in ``cwd`` and print what it says.

    The command is printed first, so it can be repeated in a terminal as it is.
    ``quiet`` prints neither, for when a cell reads the output itself.
    """
    Path(cwd).mkdir(parents=True, exist_ok=True)
    args = [str(arg) for arg in args]
    done = subprocess.run([_bin("pipstools"), *args], capture_output=True, text=True, cwd=cwd)
    if not quiet:
        print(f"$ cd {cwd} && pipstools {shlex.join(args)}")
        print(done.stdout or done.stderr)
    return done


def _files(stem: str | Path) -> tuple[Path, str]:
    stem = Path(stem)
    name = stem.name.removesuffix(".amat.parquet")
    return stem.parent, name


def structure(stem: str | Path, quiet: bool = False) -> dict:
    """Count the borders of an annotated problem written as parquet at ``stem``.

    Reads the partition of every row and column, which is where a structure
    stands or falls: how many columns sit in the root, how the leaves compare in
    size, and how many rows link blocks, and of those how many link only two
    neighbouring ones (a 2-link row, which PIPS-IPM++ handles cheaply).
    """
    import numpy as np
    import polars as pl

    folder, name = _files(stem)
    cols = pl.read_parquet(folder / f"{name}.cols.parquet", columns=["partition"])
    rows = pl.read_parquet(folder / f"{name}.rows.parquet", columns=["partition"])
    col_p = cols["partition"].to_numpy().astype(np.int64)
    row_p = rows["partition"].to_numpy().astype(np.int64)
    n_leaves = int(col_p.max()) - 1
    linking = n_leaves + 2

    # the leaves every linking row touches: a 2-link row touches exactly two neighbours
    amat = pl.read_parquet(folder / f"{name}.amat.parquet", columns=["row", "col"])
    r, c = amat["row"].to_numpy(), amat["col"].to_numpy()
    keep = (row_p[r] == linking) & (col_p[c] > 1)
    touched = (
        pl.DataFrame({"row": r[keep], "leaf": col_p[c[keep]]})
        .unique()
        .group_by("row")
        .agg(pl.len().alias("leaves"), (pl.col("leaf").max() - pl.col("leaf").min()).alias("span"))
    )
    two_link = touched.filter((pl.col("leaves") == 2) & (pl.col("span") == 1)).height

    leaf_cols = np.bincount(col_p, minlength=linking)[2:linking]
    info = {
        "columns": int(col_p.size),
        "rows": int(row_p.size),
        "nonzeros": int(r.size),
        "leaves": n_leaves,
        "root_columns": int((col_p == 1).sum()),
        "linking_rows": int((row_p == linking).sum()),
        "two_link_rows": two_link,
        "leaf_columns": leaf_cols.tolist(),
    }
    if not quiet:
        print(f"{info['columns']:,} columns, {info['rows']:,} rows, "
              f"{info['nonzeros']:,} nonzeros in {n_leaves} leaves")
        print(f"  root columns  {info['root_columns']:,}")
        print(f"  linking rows  {info['linking_rows']:,}, of them 2-link {two_link:,}")
        sizes = leaf_cols if n_leaves <= 8 else [leaf_cols.min(), "...", leaf_cols.max()]
        print(f"  columns per leaf  {', '.join(f'{s:,}' if s != '...' else s for s in sizes)}")
    return info


def visualize(stem: str | Path, width: int = 360, summary: bool = True) -> Path:
    """Draw the annotated problem at ``stem`` with ``pipstools visualize``.

    Shows the picture in the notebook and, with ``summary``, the border counts
    from :func:`structure` above it. Returns the path of the PNG.
    """
    folder, name = _files(stem)
    # numbered axes: pipstools 0.3.1 fails on a name tick that lands on the last column
    done = pipstools("visualize", f"{name}.amat.parquet", "--outputfile", f"{name}.png",
                     "--no-names", cwd=folder, quiet=True)
    picture = folder / f"{name}.png"
    if done.returncode != 0 or not picture.exists():
        print(done.stdout, done.stderr[-2000:])
        raise RuntimeError(f"pipstools visualize failed for {folder / name}")
    if summary:
        structure(folder / name)
    try:
        from IPython.display import Image, display

        display(Image(str(picture), width=width))
    except ImportError:
        print(f"wrote {picture}")
    return picture


def mpirun(script: str | Path, ranks: int, *args, timeout: float | None = None,
           show: str | None = None) -> subprocess.CompletedProcess:
    """Run ``python script`` on ``ranks`` MPI processes and print its output.

    More ranks than CPU cores is allowed (it is slower, not wrong), which laptops
    with few cores need. ``show`` keeps only the lines starting with that prefix,
    which hides the solver's own log.
    """
    if not CAN_SOLVE:
        print(f"⏭  mpirun {script} on {ranks} ranks skipped, {_WHY_NO_SOLVER}")
        return subprocess.CompletedProcess([], 0, "", "")

    launcher = _bin("mpiexec")
    command = [launcher, "-n", str(ranks)]
    version = subprocess.run([launcher, "--version"], capture_output=True, text=True)
    if "Open MPI" in version.stdout + version.stderr and ranks > (os.cpu_count() or 1):
        command.append("--oversubscribe")
    command += [sys.executable, str(script), *map(str, args)]

    done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    lines = done.stdout.splitlines()
    if show is not None:
        lines = [line for line in lines if line.startswith(show)]
    print("\n".join(lines) if lines else "(no output)")
    if done.returncode != 0:
        print(done.stderr[-2000:])
        raise RuntimeError(f"mpiexec exited with code {done.returncode}")
    return done


def _register_magics() -> None:
    try:
        from IPython import get_ipython
        from IPython.core.magic import register_cell_magic
    except ImportError:
        return
    shell = get_ipython()
    if shell is None:
        return

    @register_cell_magic
    def requires(line, cell):
        """Run the cell only if this machine has what it needs: %%requires solver"""
        needs = line.split()
        unknown = [need for need in needs if need not in _REASONS]
        if not needs or unknown:
            raise ValueError(f"%%requires takes one or more of {', '.join(_REASONS)}, "
                             f"not {line.strip()!r}")
        for need in needs:
            available, reason = _REASONS[need]
            if not available():
                print(f"⏭  {reason()}")
                return
        shell.run_cell(cell).raise_error()


_register_magics()
