"""Small public examples for your pose.py; run: python selfcheck.py.

Some expected example outputs are visible here. These are preliminary checks,
not the official grader or general solution implementations. The final grader
uses additional inputs, shapes, graph walks and consistency checks.
Running this file does not submit work or trigger Classroom 50 grading.
"""
import sys
import traceback
import numpy as np
import pose


def same(actual, expected):
    expected = np.asarray(expected, dtype=float)
    assert isinstance(actual, np.ndarray), "return a NumPy array"
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.issubdtype(actual.dtype, np.floating), "return floating-point arrays"
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)


def rotation():
    for matrix, expected in [([[1., 0.], [0., 1.]], True),
                             ([[1., 0.], [0., -1.]], False),
                             ([[2., 0.], [0., 2.]], False)]:
        result = pose.is_rotation(np.array(matrix))
        assert isinstance(result, (bool, np.bool_)), "return a Boolean"
        assert result == expected
    same(pose.rot2(-np.pi/3), [[.5, np.sqrt(3)/2], [-np.sqrt(3)/2, .5]])
    same(pose.rot2(0), np.eye(2))


def pose_and_inverse():
    T = pose.make_pose(.4, -1.3, np.pi/4)
    c = np.sqrt(2)/2
    same(T, [[c, -c, .4], [c, c, -1.3], [0, 0, 1]])
    U = pose.invert_pose(T)
    assert isinstance(U, np.ndarray) and U.shape == (3, 3)
    same(T @ U, np.eye(3))
    same(U @ T, np.eye(3))


def points_and_directions():
    T = pose.make_pose(.6, -1.1, np.pi/3)
    P = np.array([[.2, -.5], [1.4, .8]])
    same(pose.transform_points(T, P), [[1.1330127018922194, -1.1767949192431124], [0.6071796769724491, 0.512435565298214]])
    same(pose.transform_directions(T, P), [[0.5330127018922193, -0.07679491924311233], [0.007179676972449151, 1.612435565298214]])
    same(pose.transform_points(T, np.empty((0, 2))), np.empty((0, 2)))
    same(pose.transform_directions(T, np.empty((0, 2))), np.empty((0, 2)))


def lidar():
    # These inputs match Question 3's example, including a rotated mount.
    T_W_B = pose.make_pose(-1.2, 2.4, np.pi/6)
    T_B_S = pose.make_pose(.3, -.2, -np.pi/3)
    ranges = np.array([2., 0., 1.5])
    bearings = np.array([-np.pi/6, np.pi/4, np.pi/2])
    same(pose.scan_to_world(T_W_B, T_B_S, ranges, bearings),
         [[.15980762113533187, .6447441116742347],
          [-.8401923788646684, 2.376794919243112],
          [-.09019237886466824, 3.67583302491977]])


def body_world():
    # These four inputs match Question 4's examples.
    T = pose.make_pose(-2., .75, -np.pi/3)
    D_translate = pose.make_pose(.6, -.25, 0)
    D_rotate = pose.make_pose(0, 0, np.pi/6)
    body_t = pose.apply_increment(T, D_translate, "body")
    world_t = pose.apply_increment(T, D_translate, "world")
    body_r = pose.apply_increment(T, D_rotate, "body")
    world_r = pose.apply_increment(T, D_rotate, "world")
    for U in [body_t, world_t, body_r, world_r]:
        assert isinstance(U, np.ndarray) and U.shape == (3, 3)
    same(body_t[:2, 2], [-1.9165063509461095, .10538475772933686])
    same(world_t[:2, 2], [-1.4, .5])
    same(body_r[:2, 2], [-2., .75])
    same(world_r[:2, 2], [-2.107050807568877, -.3504809471616709])
    same(body_t[:2, :2], T[:2, :2])
    same(world_t[:2, :2], T[:2, :2])
    R = np.array([[np.sqrt(3)/2, .5], [-.5, np.sqrt(3)/2]])
    same(body_r[:2, :2], R)
    same(world_r[:2, :2], R)


def graph():
    T = pose.make_pose(-.3, 1.7, np.pi/5)
    reverse = pose.solve_pose_path({("W", "B"): T}, ["B", "W"])
    assert isinstance(reverse, np.ndarray) and reverse.shape == (3, 3)
    same(T @ reverse, np.eye(3))
    same(reverse @ T, np.eye(3))
    same(pose.solve_pose_path({}, ["W"]), np.eye(3))
    for edges, path, expected_error in [({}, [], ValueError),
                                       ({}, ["A", "B"], KeyError)]:
        try:
            pose.solve_pose_path(edges, path)
        except expected_error:
            pass
        else:
            raise AssertionError(f"Expected {expected_error.__name__} for {path}")


def docking():
    K = pose.make_pose(.15, -.35, np.pi/6)
    S = pose.make_pose(.45, .1, -np.pi/4)
    D = pose.make_pose(1.3, -.6, np.pi/3)
    same(pose.docking_pose(K, S, D), [[0.9659258262890683, 0.2588190451025208, 0.24171688399516778], [-0.2588190451025208, 0.9659258262890683, -1.171283569534304], [0.0, 0.0, 1.0]])


CHECKS = [rotation, pose_and_inverse, points_and_directions, lidar,
          body_world, graph, docking]

if __name__ == "__main__":
    failures = 0
    for check in CHECKS:
        try:
            check()
            print("PASS", check.__name__)
        except Exception:
            failures += 1
            print("FAIL", check.__name__)
            traceback.print_exc(limit=2)
    print(f"{len(CHECKS)-failures}/{len(CHECKS)} public examples passed; not a predicted grade")
    sys.exit(bool(failures))
