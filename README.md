# Assignment 01 - Coordinate Frames and 2D Pose

Submission deadline: **4 November 2026, 23:59 CET**.
The exercise handout is provided separately on the course page.
This repository contains the Python starter and public checks.

## Work and submit

Use that account for every assignment, and inform the lecture team before changing your account or username.

## Numerical conventions

- `T_A_B` maps coordinates from B into A; its translation is expressed in A.
- Use metres, radians and positive counterclockwise angles.
- Point/vector batches have shape `(N, 2)`, with one item per row.
- Keep inputs unchanged; return unrounded floating-point NumPy arrays.
  `is_rotation` instead returns a Boolean. Inputs may have integer dtype.
- `is_rotation` uses `atol=1e-8, rtol=0` for both rotation conditions.
  Other numerical comparisons use `atol=1e-8, rtol=1e-8`; float64 is recommended.
- `selfcheck.py` runs locally and does not submit work or trigger grading.
