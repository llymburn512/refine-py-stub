# REFINE-python-stub

Depends on [cyipopt](https://cyipopt.readthedocs.io/en/stable/tutorial.html) (see instructinos for installation instructions with HSL libraries) and [zonopy](https://github.com/roahmlab/zonopy)


`refine_backend.py` contains a minimal generic implementation of REFINE using `zonopy`

`refine_tiny_demo.py` generates a small set of obstacles and an FRS and then attempts to optimize towards a waypoint that is in collision. If all installation is done correctly, the solution point should be near `0.56519148`
