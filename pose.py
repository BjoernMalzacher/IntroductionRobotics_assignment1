"""Assignment 01: implement the functions below using NumPy.

T_A_B is the pose of B in A, mapping coordinates B -> A.
Angles are radians, positive counter-clockwise; distances are metres.
Point/direction batches have shape (N, 2), one vector per row.
Inputs are finite, valid SE(2) matrices unless stated otherwise.
Inputs may have integer dtype. Return floating arrays (float64 recommended),
except is_rotation, which returns a Boolean.
Do not mutate inputs. Use only Python's standard library and NumPy.
"""
import numpy as np


def is_rotation(R):
    """Return bool (or np.bool_) for a finite real ndarray of shape (2, 2).

    R is a candidate matrix, not guaranteed to be a valid rotation.
    Return True when both rotation conditions hold within tolerance,
    and False otherwise. Keep R unchanged.
    Apply the planar rotation conditions, using atol=1e-8 and rtol=0
    for both the matrix comparison and the scalar comparison.
    """
    #RT * R = I Check
    I = R @ R.T
    if np.allclose(I, np.eye(2), atol=1e-8, rtol=0):
        #det(R) = 1 Check
        if np.isclose(np.linalg.det(R), 1, atol=1e-8, rtol=0):
            # R Inverse = R Transpose Check
            if np.allclose(np.linalg.inv(R), R.T, atol=1e-8, rtol=0):
                return True
    
    return False


def rot2(theta):
    """Return a float ndarray of shape (2, 2) for scalar theta."""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]], dtype=float)
 


def make_pose(x, y, theta):
    """Return T_A_B, shape (3, 3); (x,y) is B's origin expressed in A."""
    T_A_B = np.eye(3)
    T_A_B[0:2, 0:2] = rot2(theta)
    T_A_B[0:2, 2] = [x, y]
    

    return T_A_B


def invert_pose(T_A_B):
    """Return T_B_A using blocks; do not call np.linalg.inv/solve here."""
    t = T_A_B[0:2, 2]
    T_B_A = np.eye(3)
    T_B_A[0:2, 0:2] = T_A_B[0:2, 0:2].T
    T_B_A[0:2, 2] = -T_A_B[0:2, 0:2].T @ t
    return T_B_A


def transform_points(T_A_B, points_B):
    """Return points_A, shape (N,2). Empty (0,2) input stays (0,2)."""
    # Convert points_B to homogeneous coordinates
    points_B_homogeneous = np.hstack((points_B, np.ones((points_B.shape[0], 1))))
 
    points_A_homogeneous = (T_A_B @ points_B_homogeneous.T).T
    
    return points_A_homogeneous[:, :2]


def transform_directions(T_A_B, directions_B):
    """Return free vectors in A, shape (N,2); apply no translation."""
    # Convert directions_B to homogeneous coordinates
    directions_B_homogeneous = np.hstack((directions_B, np.zeros((directions_B.shape[0], 1))))
    
    directions_A_homogeneous = (T_A_B @ directions_B_homogeneous.T).T
    
    return directions_A_homogeneous[:, :2]

def scan_to_world(T_W_B, T_B_S, ranges, bearings):
    """Return (N,2) world points for nonnegative ranges and radian bearings.

    ranges and bearings are 1D arrays of equal length (possibly zero).
    Do not filter returns, round results or change their order.
    """
    # Convert polar coordinates to Cartesian coordinates in the sensor frame
    points_S = np.column_stack((ranges * np.cos(bearings), ranges * np.sin(bearings)))

    # T_B_S @ points_S = points_B
    points_B = transform_points(T_B_S, points_S)

    # T_W_B @ points_B = points_W
    points_W = transform_points(T_W_B, points_B)

    return points_W    


def apply_increment(T_W_B, D, expressed_in):
    """Return updated pose for a full rigid displacement D.

    expressed_in == 'body': D is relative to the current body frame.
    expressed_in == 'world': D acts on the entire pose in world coordinates,
    so a rotational D also rotates the body's origin about the world origin.
    Any other value must raise ValueError.
    """
    
    if expressed_in == 'body':
        return T_W_B @ D
    elif expressed_in == 'world':
        return D @ T_W_B
    else:
        raise ValueError("Invalid value")

    


def solve_pose_path(edges, path):
    """Return T_path[0]_path[-1] for an explicitly supplied frame walk.

    path is a list or tuple of frame-name strings.
    The result maps coordinates from path[-1] into path[0].
    edges[(A,B)] stores T_A_B. Use its inverse for traversal B -> A.
    Input never contains both (A,B) and (B,A) for distinct A,B.
    len(path)==1 returns np.eye(3), even if that frame has no stored edge.
    Empty path raises ValueError. Missing adjacent edge raises KeyError.
    Repeated frames are allowed; adjacent frames differ (an input guarantee).
    No graph search, optimisation or consistency selection is required.
    """

    if len(path) == 0:
        raise ValueError("Empty path")
    elif len(path) == 1:
        return np.eye(3)
    
    #Start with identity matrix = W
    T = np.eye(3)
    # Walk though the graph along the path, multiplying the transformations together
    for i in range(len(path) - 1):
        A, B = path[i], path[i + 1]
        if (A, B) in edges:
            T = T @ edges[(A, B)]
        elif (B, A) in edges:
            T = T @ invert_pose(edges[(B, A)])
        else:
            raise KeyError(f"Missing edge for {A} -> {B}")

    return T    


def docking_pose(T_B_K, T_B_S, T_S_D):
    """Return T_K_D: dock pose expressed in the coupling frame K."""


    T_B_D = T_B_S @ T_S_D
    T_K_D = invert_pose(T_B_K) @ T_B_D
    return T_K_D
