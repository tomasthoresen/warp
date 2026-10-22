Speed up `tile_cholesky()` and `tile_cholesky_solve()` with a vector right-hand side on HIP for matrices that fit in one wavefront: one wavefront factors and solves with the rows held in registers.
