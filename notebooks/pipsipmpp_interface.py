from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from gams.core import gmo

__all__ = ["Instance", "Solution", "apply", "extract", "read_parquet_solution",
           "solve", "write_parquet"]

ROOT_STAGE = 1
_EQUALITY, _LOWER, _UPPER = gmo.gmoequ_E, gmo.gmoequ_G, gmo.gmoequ_L
_QUADRATIC = {gmo.gmoProc_qcp, gmo.gmoProc_miqcp, gmo.gmoProc_rmiqcp}


def _rc(r):
    """GMO wrappers return either an int or [rc, msg]."""
    return r[0] if isinstance(r, (list, tuple)) else r


def _doubles(h, fn, n, *extra):
    buf = gmo.doubleArray(n)
    if _rc(fn(h, buf, *extra)) != 0:
        raise RuntimeError(f"{fn.__name__} failed")
    return np.fromiter((buf[i] for i in range(n)), np.float64, n)


def _ints(h, fn, n):
    buf = gmo.intArray(n)
    if _rc(fn(h, buf)) != 0:
        raise RuntimeError(f"{fn.__name__} failed")
    return np.fromiter((buf[i] for i in range(n)), np.int32, n)


def _jacobian(h, m, n, nnz):
    """The constraint matrix, which gmoGetMatrixRow fills as CSR directly."""
    rowstart, colidx = gmo.intArray(m + 1), gmo.intArray(nnz)
    jacval, nlflag = gmo.doubleArray(nnz), gmo.intArray(nnz)
    if _rc(gmo.gmoGetMatrixRow(h, rowstart, colidx, jacval, nlflag)) != 0:
        raise RuntimeError("gmoGetMatrixRow failed")
    return sp.csr_array(
        (np.fromiter((jacval[k] for k in range(nnz)), np.float64, nnz),
         np.fromiter((colidx[k] for k in range(nnz)), np.int64, nnz),
         np.fromiter((rowstart[i] for i in range(m + 1)), np.int64, m + 1)),
        shape=(m, n),
    )


def _require_linear(h):
    """Turn away anything PIPS-IPM++ cannot solve."""
    quadratic = gmo.gmoModelType(h) in _QUADRATIC

    if gmo.gmoNLNZ(h):
        kind = "quadratic" if quadratic else "nonlinear"
        raise ValueError(
            f"PIPS-IPM++ solves linear problems, and this instance has "
            f"{gmo.gmoNLNZ(h)} {kind} constraint terms"
        )

    if gmo.gmoObjNLNZ(h):
        if quadratic:
            raise ValueError(
                f"PIPS-IPM++ does not take a quadratic objective yet, and this "
                f"instance has {gmo.gmoObjNLNZ(h)} quadratic objective terms"
            )
        raise ValueError(
            f"PIPS-IPM++ solves linear problems, and this instance has "
            f"{gmo.gmoObjNLNZ(h)} nonlinear objective terms"
        )

    if gmo.gmoNDisc(h):
        raise ValueError(
            f"PIPS-IPM++ solves continuous problems and does not branch, and this "
            f"instance has {gmo.gmoNDisc(h)} discrete columns"
        )


def _blocks_from_stages(stage, n_blocks):
    """The block of every column, from the ``.stage`` GAMS recorded for it."""
    if not np.all(np.isfinite(stage)) or not np.all(stage == np.rint(stage)):
        bad = stage[~np.isfinite(stage) | (stage != np.rint(stage))][:3]
        raise ValueError(f"every column needs a whole-numbered .stage, found {bad}")
    if stage.min() < ROOT_STAGE:
        raise ValueError(
            f"stages start at {ROOT_STAGE}, which is the root, but found {stage.min():g}. "
            "A column whose .stage is never set keeps the GAMS default of 1."
        )

    var_block = np.rint(stage).astype(np.int64) - ROOT_STAGE
    highest = int(var_block.max())
    if n_blocks is None:
        n_blocks = highest
    if n_blocks < 1:
        raise ValueError(
            "no leaf blocks: every column sits at the root. Set .stage to "
            f"{ROOT_STAGE + 1} + k on the columns of leaf block k."
        )
    if highest > n_blocks:
        raise ValueError(
            f"stages have to lie in {ROOT_STAGE}..{ROOT_STAGE + n_blocks}, "
            f"found {highest + ROOT_STAGE}"
        )
    return var_block, n_blocks


@dataclass(frozen=True)
class Instance:
    """A generated GAMSPy model, annotated and in the form PIPS-IPM++ takes."""

    problem: "object"                # pipsipmpp.StructuredProblem
    minimize: bool
    is_equality: np.ndarray          # (m,) which generated rows are equalities
    var_name: np.ndarray             # (n,) column order of the primal vector
    equ_name: np.ndarray             # (m,) row order of the marginals


def extract(model, *, n_blocks=None, annotation=None, regex=None,
            annotation_options=None, blocks=True) -> Instance:
    r"""Generate ``model`` and hand it back as an annotated ``StructuredProblem``.

    The model is generated, not solved. ``n_blocks`` defaults to the highest stage
    found, so it only has to be given to insist on a particular number, for
    instance when a block came out empty.

    ``annotation`` takes the other path, for a model that sets no ``.stage`` at
    all: the structure is derived from the matrix with pipstools instead of being
    read off the model, and ``n_blocks`` then says how many leaf blocks to aim
    for. ``"hypergraph"`` partitions the matrix, ``"regex"`` groups the columns by
    a capture taken from the names GAMS gave them::

        extract(model, n_blocks=4, annotation="hypergraph")
        extract(model, n_blocks=4, annotation="regex", regex=r",(\d+)\)$")

    GAMS names a generated column after its symbol and keys, ``gen(gas,17)``, so
    the pattern above captures the last key: a column without one, ``cap(gas)``,
    matches nothing and stays in the root, which is where a variable shared by
    every block belongs. Deriving needs the optional pipstools dependency
    (``pip install "pipsipmpp[annotate]"``).

    ``blocks=False`` asks for the ordering alone: the model is generated and its
    columns and rows are laid out as usual, but no block structure is read or
    derived, so a model that states none is accepted. That is what reading a
    solution back needs, since only the order has to agree.
    """
    was_frozen = getattr(model, "_is_frozen", False)
    if not was_frozen:
        model.freeze(modifiables=[])          # generates only, no solver is called
    try:
        return _read(model.instance._gmo, n_blocks, annotation, regex,
                     annotation_options, blocks)
    finally:
        if not was_frozen:
            model.unfreeze()


def _derive_blocks(problem, var_name, n_blocks, annotation, regex, options):
    """Hand the matrix to pipstools and take the annotation back."""
    from pipsipmpp import annotate

    method = annotation or "regex"
    options = dict(options or {})
    if regex is not None:
        options["regex"] = regex
    names = list(var_name) if method == "regex" else None
    return annotate(problem, n_blocks, method=method, names=names, **options)


def _read(h, n_blocks, annotation=None, regex=None, annotation_options=None,
          blocks=True) -> Instance:
    from pipsipmpp import StructuredProblem

    gmo.gmoIndexBaseSet(h, 0)                 # 0-based, so a GMO index is a numpy index
    gmo.gmoObjStyleSet(h, gmo.gmoObjType_Fun)  # drop the synthetic objective row and
    gmo.gmoObjReformSet(h, 1)                  # column, and give the gradient instead

    _require_linear(h)

    m, n = gmo.gmoM(h), gmo.gmoN(h)
    derive = annotation is not None or regex is not None
    if derive:
        if n_blocks is None:
            raise ValueError(
                "Deriving a block structure needs n_blocks, the number of leaf "
                "blocks to aim for."
            )
        # every column in the root until pipstools says otherwise; .stage is not
        # read at all, since the path exists for a model that never set it
        var_block = np.zeros(n, dtype=np.int64)
    elif blocks:
        var_block, n_blocks = _blocks_from_stages(
            _doubles(h, gmo.gmoGetVarStage, n), n_blocks
        )
    else:
        # the ordering alone was asked for, so a model that states no structure
        # is fine; the problem built here is never solved
        var_block, n_blocks = np.zeros(n, dtype=np.int64), 1

    lower = _doubles(h, gmo.gmoGetVarLower, n)
    upper = _doubles(h, gmo.gmoGetVarUpper, n)
    lower[lower <= gmo.gmoMinf(h)] = -np.inf   # GAMS infinity is +-1e299
    upper[upper >= gmo.gmoPinf(h)] = np.inf

    # the second buffer flags nonlinear terms, which gmoObjNLNZ already ruled out
    cost = _doubles(h, gmo.gmoGetObjVector, n, gmo.intArray(n))
    matrix = _jacobian(h, m, n, gmo.gmoNZ(h))
    rhs = _doubles(h, gmo.gmoGetRhs, m)

    sense = _ints(h, gmo.gmoGetEquType, m)
    unknown = set(np.unique(sense)) - {_EQUALITY, _LOWER, _UPPER}
    if unknown:
        raise ValueError("only =e=, =g= and =l= rows map onto a PIPS-IPM++ row; "
                         f"found GMO equation type(s) {sorted(unknown)}")
    is_equality = sense == _EQUALITY
    is_lower = sense[~is_equality] == _LOWER
    rhs_inequality = rhs[~is_equality]

    # PIPS-IPM++ minimises; a maximisation is negated here and turned back in solve
    minimize = gmo.gmoSense(h) == gmo.gmoObj_Min
    sign = 1.0 if minimize else -1.0

    problem = StructuredProblem(
        n_blocks=n_blocks,
        var_block=var_block,
        c=sign * cost,
        xlow=lower,
        xupp=upper,
        A_eq=sp.csr_array(matrix[is_equality]),
        b_eq=rhs[is_equality],
        A_ineq=sp.csr_array(matrix[~is_equality]),
        ineq_low=np.where(is_lower, rhs_inequality, -np.inf),
        ineq_upp=np.where(is_lower, np.inf, rhs_inequality),
        objconst=sign * gmo.gmoObjConst(h),
    )
    problem.validate()

    var_name = np.array([gmo.gmoGetVarNameOne(h, j) for j in range(n)])
    if derive:
        problem = _derive_blocks(
            problem, var_name, n_blocks, annotation, regex, annotation_options
        )

    return Instance(
        problem=problem,
        minimize=minimize,
        is_equality=is_equality,
        var_name=var_name,
        equ_name=np.array([gmo.gmoGetEquNameOne(h, i) for i in range(m)]),
    )


# ------------------------------------------------------------------- the answer
@dataclass(frozen=True)
class Solution:
    """What PIPS-IPM++ found, in the order the generated model states it.

    The four vectors are the GAMS ``.l`` and ``.m`` of the columns and the rows.
    They are gathered on rank 0 only; on every other rank they are ``None``, as
    are the names. The scalars are the same everywhere.
    """

    status: "object"                 # pipsipmpp.TerminationStatus
    objective: float
    iterations: int
    runtime: float
    var_level: np.ndarray | None = None      # (n,) column values, in var_name order
    var_marginal: np.ndarray | None = None   # (n,) reduced costs
    equ_level: np.ndarray | None = None      # (m,) row activities, in equ_name order
    equ_marginal: np.ndarray | None = None   # (m,) row multipliers
    var_name: np.ndarray | None = None
    equ_name: np.ndarray | None = None

    @property
    def optimal(self) -> bool:
        return bool(self.status.is_optimal)

    def levels(self) -> dict[str, float]:
        """Column values by GAMS name, as ``{"cap(wind)": 2.55, ...}``."""
        return _by_name(self.var_name, self.var_level)

    def marginals(self) -> dict[str, float]:
        """Row multipliers by GAMS name."""
        return _by_name(self.equ_name, self.equ_marginal)


def _by_name(names, values) -> dict[str, float]:
    if names is None or values is None:
        return {}
    return {str(name): float(value) for name, value in zip(names, values)}


# ------------------------------------------------- writing it onto the GAMS model
def _split_name(name):
    """``"gen(gas,17)"`` -> ``("gen", ["gas", "17"])``, a scalar giving ``[]``.

    The generated instance identifies a column or a row by the name GAMS gave it,
    which is the handle to reach back to the symbol it came from. This is only
    used to report a solution; the block structure is read from ``.stage``.
    """
    symbol, _, keys = str(name).partition("(")
    return symbol, keys.rstrip(")").split(",") if keys else []


def _records(container, names, level, marginal):
    """Group flat solver vectors into one ``setRecords`` frame per GAMS symbol."""
    import pandas as pd

    rows: dict[str, list] = {}
    for name, lvl, marg in zip(names, level, marginal):
        symbol, keys = _split_name(name)
        if symbol in container:
            rows.setdefault(symbol, []).append([*keys, float(lvl), float(marg)])

    for symbol, values in rows.items():
        labels = list(container[symbol].domain_names)
        yield container[symbol], pd.DataFrame(values, columns=[*labels, "level", "marginal"])


def apply(solution: Solution, model) -> None:
    """Write a solution onto the GAMSPy model it was generated from.

    Afterwards the model carries the answer the way a GAMS solve would leave it:
    ``cap.records`` and ``cap.l[...]`` hold the levels, ``.m`` the marginals, the
    equations carry their activity and their multiplier, and
    ``model.objective_value`` is set. Does nothing on a rank that gathered no
    solution.
    """
    if solution.var_level is None:
        return

    container = model.container
    for symbol, frame in _records(container, solution.var_name,
                                  solution.var_level, solution.var_marginal):
        symbol.setRecords(frame)
    for symbol, frame in _records(container, solution.equ_name,
                                  solution.equ_level, solution.equ_marginal):
        symbol.setRecords(frame)

    # objective_value is read-only on the model, since GAMS normally fills it
    model._objective_value = float(solution.objective)


def _rows(is_equality, from_equalities, from_inequalities):
    """Weave the two solver vectors back into the row order of the model."""
    woven = np.empty(is_equality.size, dtype=float)
    woven[is_equality] = from_equalities
    woven[~is_equality] = from_inequalities
    return woven


def solve(build_model, *, n_blocks=None, annotation=None, regex=None,
          annotation_options=None, comm=None, options=None,
          options_file=None, load_solution=True) -> Solution:
    """Generate the model on rank 0 and solve it with PIPS-IPM++ on every rank.

    ``build_model`` is a callable returning the ``gamspy.Model`` to solve. It runs
    on rank 0 only, so GAMS generates the instance once however many ranks the job
    has, and the other ranks wait for the blocks they own. The call is collective:
    every rank has to reach it.

    With ``load_solution`` the answer is written back onto the model that was
    built, so it can be read from the GAMSPy symbols as after a GAMS solve. See
    :func:`apply`.

    ``annotation`` derives the block structure from the matrix rather than
    reading it off ``.stage``, for a model that sets none; see :func:`extract`.

    ``options`` and ``options_file`` are handed to PIPS-IPM++, and ``comm``
    defaults to ``MPI.COMM_WORLD``.
    """
    import pipsipmpp

    comm = pipsipmpp.resolve_comm(comm)

    model = instance = None
    if comm.Get_rank() == 0:
        model = build_model()
        instance = extract(model, n_blocks=n_blocks, annotation=annotation,
                           regex=regex, annotation_options=annotation_options)

    result = pipsipmpp.solve(
        instance.problem if instance is not None else None,
        comm, options=options, options_file=options_file,
    )

    solution = _solution(instance, result)
    if load_solution:
        apply(solution, model)
    return solution


def _solution(instance, result) -> Solution:
    """A PIPS-IPM++ result in the order and the sense the generated model states."""
    common = dict(status=result.status, objective=result.objective,
                  iterations=result.iterations, runtime=result.runtime)
    if instance is None or result.primal is None:
        return Solution(**common)

    # the solver was handed a minimisation, so a maximised model gets its own
    # objective and multipliers back here
    sign = 1.0 if instance.minimize else -1.0
    is_eq = instance.is_equality

    return Solution(
        var_level=result.primal,
        var_marginal=sign * result.reduced_costs,
        equ_level=_rows(is_eq, result.eq_activity, result.ineq_activity),
        equ_marginal=sign * _rows(is_eq, result.dual_eq, result.dual_ineq),
        var_name=instance.var_name, equ_name=instance.equ_name,
        **{**common, "objective": sign * common["objective"]},
    )


def write_parquet(model, path, *, layout="monolithic", names=True, n_blocks=None,
                  annotation=None, regex=None, annotation_options=None):
    """Write the generated model as annotated parquet, without solving it.

    ``layout`` is ``"monolithic"`` for one set of files holding the whole model,
    or ``"distributed"`` for one set per block, which lets each rank of a later
    solve read only its own share. ``names`` records each column and row under
    the name GAMS gave it, so a matrix plotted from the files is labelled the way
    the model reads.

    The block structure is taken from ``.stage``, or derived with ``annotation``,
    exactly as :func:`solve` takes it. The files can be handed to PIPS-IPM++
    directly (``pipsparquet model``), inspected with pipstools, or read back with
    pipsipmpp. Returns the stem they were written to.
    """
    from pipsipmpp import write_problem

    instance = extract(model, n_blocks=n_blocks, annotation=annotation,
                       regex=regex, annotation_options=annotation_options)
    return write_problem(
        instance.problem,
        path,
        layout=layout,
        names=_parquet_names(instance) if names else None,
        # the problem carries minimisation costs; the sense records how the model
        # stated them
        sense=1.0 if instance.minimize else -1.0,
    )


def _parquet_names(instance) -> dict:
    """Names in the order the format stores them, which is every equality first."""
    is_eq = instance.is_equality
    return {
        "cols": [str(name) for name in instance.var_name],
        "rows": [str(name) for name in instance.equ_name[is_eq]]
        + [str(name) for name in instance.equ_name[~is_eq]],
    }


def read_parquet_solution(model, path, *, load_solution=True) -> Solution:
    """Read a solution written beside the parquet problem at ``path``.

    This is the other half of :func:`write_parquet`: write the model, solve it
    elsewhere with ``pipsparquet ... writesol`` or
    ``pipsipmpp.solve_dataset(..., write_solution=True)``, then read the answer
    back onto the model it came from::

        write_parquet(model, "model", annotation="hypergraph", n_blocks=4)
        # ... solved elsewhere ...
        solution = read_parquet_solution(model, "model")
        print(model.container["cap"].records)

    The :class:`Solution` is the one :func:`solve` would have returned, and it is
    written onto the model the same way, so levels and marginals read back
    identically. Neither MPI nor the compiled solver is needed for this.

    ``model`` has to be the one the files were written from: the solution is
    matched to it by position, so the model is generated again here to recover
    that order. The block structure is not: a derived one would not repeat, and
    only the ordering is read.
    """
    from pipsipmpp import read_solution

    instance = extract(model, blocks=False)
    solution = _solution(instance, read_solution(path))
    if load_solution:
        apply(solution, model)
    return solution
