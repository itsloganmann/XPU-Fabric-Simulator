"""Routing algorithms: ECMP (hash-based) and DLB (dynamic load balancing)."""

import hashlib


def ecmp_route(flow_id: int, src_gpu: int, dst_gpu: int, num_uplinks: int) -> int:
    """Select an uplink using Equal-Cost Multi-Path hashing.

    Hashes the flow's 5-tuple-like key to deterministically pick a path.
    Identical flows always take the same path, which can cause collisions.
    """
    key = f"{flow_id}:{src_gpu}:{dst_gpu}"
    digest = hashlib.md5(key.encode()).hexdigest()
    return int(digest, 16) % num_uplinks


def dlb_route(queue_depths: dict[int, int]) -> int:
    """Select the uplink with the shortest queue depth.

    Dynamic Load Balancing sprays packets across paths based on
    real-time congestion, avoiding the hash-collision problem of ECMP.
    """
    return min(queue_depths, key=queue_depths.get)
