"""Data of the sample model, shared by every notebook so their answers compare.

    from sample_data import TECHS, STORAGE, SOLAR, budget, profiles

    demand, avail = profiles(T)          # demand[t], avail["wind"][t], ...

A technology is a row of ``TECHS``: its investment cost per unit of capacity,
its cost per unit generated, its emissions per unit generated, and the name of
its availability profile. Add a row to add a technology to every model.
"""

import numpy as np

TECHS = {
    #        investment  variable  emission  availability
    "wind": (1.0,        0.0,      0.0,      "wind"),
    "gas":  (0.4,        0.5,      0.4,      "firm"),
}

# the technology the exercises add: TECHS | {"solar": SOLAR}
SOLAR = (0.6, 0.0, 0.0, "solar")

# the storage the exercises add: investment per unit of energy capacity, and the
# share of the charged energy that arrives in the store
STORAGE = {"investment": 0.2, "efficiency": 0.9}

BUDGET_PER_DAY = 0.1


def profiles(T: int):
    """Demand in every hour, and the availability profiles by name."""
    rng = np.random.default_rng(0)
    demand = 1.0 + 0.2 * rng.random(T)
    hour_of_day = np.arange(T) % 24
    avail = {
        "wind": 0.3 + 0.5 * rng.random(T),
        "solar": np.clip(np.sin(np.pi * (hour_of_day - 6) / 12), 0.0, None),
        "firm": np.ones(T),
    }
    return demand, avail


def budget(T: int) -> float:
    """The emission budget for ``T`` hours, the same per day however long the horizon."""
    return BUDGET_PER_DAY * (T / 24)
