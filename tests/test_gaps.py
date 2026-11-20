import numpy as np
from src.gaps import find_gaps, cluster_rows


def row(y, xs, w=50, h=80):
    return [[x, y, x + w, y + h] for x in xs]


def test_clusters_two_rows():
    boxes = np.array(row(10, [0, 50, 100, 150]) + row(200, [0, 50, 100, 150]))
    assert len(cluster_rows(boxes)) == 2


def test_full_shelf_has_no_gaps():
    boxes = np.array(row(10, [0, 50, 100, 150, 200]))
    gaps, _ = find_gaps(boxes, img_w=250)
    assert gaps == []


def test_missing_product_between():
    # product at x=100 missing -> 50px hole between 50..100 and 150
    boxes = np.array(row(10, [0, 50, 150, 200, 250]))
    gaps, _ = find_gaps(boxes, img_w=300)
    assert len(gaps) == 1
    g = gaps[0]
    assert g.kind == "between" and (g.x1, g.x2) == (100, 150) and g.est_missing == 1


def test_two_missing_estimated():
    boxes = np.array(row(10, [0, 50, 200, 250]))
    gaps, _ = find_gaps(boxes, img_w=300, min_row_items=3)
    assert gaps[0].est_missing == 2


def test_overlapping_boxes_no_false_gap():
    boxes = np.array(row(10, [0, 40, 80, 120, 160]))  # overlap by 10px
    gaps, _ = find_gaps(boxes, img_w=210)
    assert gaps == []


def test_edge_gap_uses_other_rows_extent():
    full = row(10, [0, 50, 100, 150, 200])
    short = row(200, [0, 50, 100])          # right part of lower shelf empty
    gaps, _ = find_gaps(np.array(full + short), img_w=250)
    assert any(g.kind == "right_edge" and g.row == 1 for g in gaps)


def test_empty_input():
    assert find_gaps(np.zeros((0, 4)), 100) == ([], [])
