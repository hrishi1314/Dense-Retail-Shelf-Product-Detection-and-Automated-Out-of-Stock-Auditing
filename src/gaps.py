"""Shelf-gap analysis: turn product detections into out-of-stock gaps.

Pipeline
--------
1. Cluster detections into shelf rows using their vertical centres.
2. Inside each row, sort products left to right and measure the empty
   horizontal space between neighbours (and between the row's outer products
   and the shelf edges).
3. Any space wider than ``gap_factor`` x the typical product width is flagged
   as an out-of-stock gap, with an estimate of how many facings are missing.

Pure NumPy, so it can be tested without a model or a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Gap:
    row: int
    x1: int
    y1: int
    x2: int
    y2: int
    est_missing: int
    kind: str  # "between" | "left_edge" | "right_edge"

    def as_dict(self) -> dict:
        return asdict(self)


def cluster_rows(boxes: np.ndarray, row_tol: float = 0.6) -> list[np.ndarray]:
    """Group boxes into shelf rows.

    Boxes are sorted by vertical centre; a new row starts whenever the jump in
    centre-y to the previous box exceeds ``row_tol`` x the median box height.
    Returns a list of index arrays (indices into ``boxes``), top row first.
    """
    if len(boxes) == 0:
        return []
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    med_h = float(np.median(boxes[:, 3] - boxes[:, 1]))
    order = np.argsort(cy)
    rows: list[list[int]] = [[int(order[0])]]
    for prev, cur in zip(order[:-1], order[1:]):
        if cy[cur] - cy[prev] > row_tol * med_h:
            rows.append([])
        rows[-1].append(int(cur))
    return [np.array(r) for r in rows]


def find_gaps(
    boxes: np.ndarray,
    img_w: int,
    row_tol: float = 0.6,
    gap_factor: float = 0.6,
    min_row_items: int = 3,
    check_edges: bool = True,
) -> tuple[list[Gap], list[np.ndarray]]:
    """Find out-of-stock gaps.

    Args:
        boxes: (N, 4) array of xyxy product boxes in pixels.
        img_w: image width, used as the right shelf edge.
        row_tol: row-clustering sensitivity (see ``cluster_rows``).
        gap_factor: a gap must exceed this x the row's median product width.
        min_row_items: rows with fewer products are ignored (likely noise).
        check_edges: also flag empty space before the first / after the last
            product, measured against the shelf extent of all rows.

    Returns:
        (gaps, rows) where ``rows`` are the index arrays of the kept rows.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    rows = [r for r in cluster_rows(boxes, row_tol) if len(r) >= min_row_items]
    if not rows:
        return [], []

    shelf_left = float(min(boxes[r, 0].min() for r in rows))
    shelf_right = float(max(boxes[r, 2].max() for r in rows))
    shelf_right = min(shelf_right, float(img_w))

    gaps: list[Gap] = []
    for row_id, idx in enumerate(rows):
        rb = boxes[idx][np.argsort(boxes[idx][:, 0])]
        med_w = float(np.median(rb[:, 2] - rb[:, 0]))
        y1 = int(np.median(rb[:, 1]))
        y2 = int(np.median(rb[:, 3]))
        min_gap = gap_factor * med_w

        def add(x_from: float, x_to: float, kind: str) -> None:
            width = x_to - x_from
            if width > min_gap:
                gaps.append(
                    Gap(row_id, int(x_from), y1, int(x_to), y2,
                        max(1, int(round(width / med_w))), kind)
                )

        # Gaps between neighbouring products. Running max of x2 handles
        # overlapping boxes (dense shelves) without producing false gaps.
        right_so_far = rb[0, 2]
        for nxt in rb[1:]:
            add(right_so_far, nxt[0], "between")
            right_so_far = max(right_so_far, nxt[2])

        if check_edges:
            add(shelf_left, rb[0, 0], "left_edge")
            add(rb[:, 2].max(), shelf_right, "right_edge")

    gaps.sort(key=lambda g: (g.row, g.x1))
    return gaps, rows
