from enum import Enum, auto

import numpy as np

# from .connections import (
# KeypointScheme,
# KeypointConnections,
# )


class KeypointScheme(Enum):
    # our 17-keypoint scheme
    _17 = auto()


KeypointConnections = {
    KeypointScheme._17: [
        [15, 13],
        [13, 11],
        [16, 14],
        [14, 12],
        [11, 12],
        [5, 11],
        [6, 12],
        [5, 6],
        [5, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        #        [1, 2],
        #        [0, 1],
        #        [0, 2],
        #        [1, 3],
        #        [2, 4],
        #        [0, 5],
        #        [0, 6]
    ]
}

# Не использвуем лицевые точки, т.к они все устремляются в 0.0.0 если
# пропадают из вида камеры


def extract_lengths(
    keypoints: np.ndarray, keypoint_scheme: KeypointScheme
) -> np.ndarray:
    """
    Takes keypoints and connection scheme.
    Returns distances between connection endpoints.
    """
    connections = KeypointConnections[keypoint_scheme]
    dists = []
    for i, j in connections:
        # extract coordinates of a connection points
        endpoints1 = keypoints[:, i, :]
        endpoints2 = keypoints[:, j, :]
        # find Euclidean distances
        dists.append(np.sqrt(np.sum(np.square(endpoints1 - endpoints2), axis=1)))
    dists = np.stack(dists, axis=1)
    return dists


def extract_near_cosines(
    keypoints: np.ndarray,
    keypoint_scheme: KeypointScheme,
) -> np.ndarray:
    keypoint_scheme = KeypointConnections[keypoint_scheme]
    cosines = []
    # find set of vertices
    pts = set()
    for edge in keypoint_scheme:
        i, j = edge
        pts.add(i)
        pts.add(j)
    pts = list(pts)
    pts = sorted(pts)
    # find connections
    connections = {i: set() for i in pts}
    for edge in keypoint_scheme:
        i, j = edge
        connections[i].add(j)
        connections[j].add(i)
    for v in pts:
        connected = connections[v]
        connected = sorted(list(connected))
        ept0 = keypoints[:, v, :]
        for i in range(len(connected)):
            for j in range(i + 1, len(connected)):
                # add cosine for the given pair of vertices
                idx1 = connected[i]
                idx2 = connected[j]
                ept1 = keypoints[:, idx1, :]
                ept2 = keypoints[:, idx2, :]

                v1 = ept1 - ept0
                v2 = ept2 - ept0

                c = np.sum(v1 * v2, axis=1) / np.sqrt(
                    1e-12 + np.sum(v1 * v1, axis=1) * np.sum(v2 * v2, axis=1)
                )
                cosines.append(c)
    cosines = np.stack(cosines, axis=1)
    return cosines


def extract_connection_derivative_angles(
    keypoints: np.ndarray,
    vects: np.ndarray,  # derivatives of keypoints
    keypoint_scheme: KeypointScheme,
) -> np.ndarray:
    """
    Takes keypoints, their derivatives, and connection scheme.
    Returns angles between connections and derivatives of adjacent connections.

    For each connection [i,j], find all other connections that share endpoint i or j,
    and compute the angle between [i,j] and the derivative of those adjacent connections.
    """
    connections = KeypointConnections[keypoint_scheme]
    angles = []

    # Create a dictionary to track which connections share endpoints
    adjacent_connections = {}
    for idx, (i, j) in enumerate(connections):
        if i not in adjacent_connections:
            adjacent_connections[i] = []
        if j not in adjacent_connections:
            adjacent_connections[j] = []
        # (connection_idx, other_endpoint)
        adjacent_connections[i].append((idx, j))
        # (connection_idx, other_endpoint)
        adjacent_connections[j].append((idx, i))

    # For each connection, find all adjacent connections and compute angles
    for idx, (i, j) in enumerate(connections):
        # Get vector for current connection
        conn_vector = keypoints[:, j, :] - keypoints[:, i, :]

        # Find adjacent connections at endpoint i
        for adj_idx, other_endpoint in adjacent_connections[i]:
            if adj_idx != idx:  # Don't compare connection with itself
                # Get derivative vector for adjacent connection
                adj_vector = vects[:, other_endpoint, :] - vects[:, i, :]

                # Compute angle between connection and derivative of adjacent connection
                # Using the dot product formula: cos(θ) = (a · b) / (|a| * |b|)
                dot_product = np.sum(conn_vector * adj_vector, axis=1)
                magnitude_a = np.sqrt(np.sum(conn_vector * conn_vector, axis=1) + 1e-12)
                magnitude_b = np.sqrt(np.sum(adj_vector * adj_vector, axis=1) + 1e-12)
                cosine = dot_product / (magnitude_a * magnitude_b)
                # Clip to ensure valid arccos input
                cosine = np.clip(cosine, -1.0, 1.0)
                angle = np.arccos(cosine)
                angles.append(angle)

        # Find adjacent connections at endpoint j
        for adj_idx, other_endpoint in adjacent_connections[j]:
            if adj_idx != idx:  # Don't compare connection with itself
                # Get derivative vector for adjacent connection
                adj_vector = vects[:, other_endpoint, :] - vects[:, j, :]

                # Compute angle between connection and derivative of adjacent
                # connection
                dot_product = np.sum(conn_vector * adj_vector, axis=1)
                magnitude_a = np.sqrt(np.sum(conn_vector * conn_vector, axis=1) + 1e-12)
                magnitude_b = np.sqrt(np.sum(adj_vector * adj_vector, axis=1) + 1e-12)
                cosine = dot_product / (magnitude_a * magnitude_b)
                # Clip to ensure valid arccos input
                cosine = np.clip(cosine, -1.0, 1.0)
                angle = np.arccos(cosine)
                angles.append(angle)

    # Stack all angles into a single array
    angles = np.stack(angles, axis=1)
    return angles
