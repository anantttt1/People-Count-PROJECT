import numpy as np

from backend.app.vision.tracker import ByteTracker, iou_matrix


def det(x, y, w=40, h=100, score=0.9):
    return [x, y, x + w, y + h, score]


def run(tracker, frames, fps=10.0):
    out = []
    for i, dets in enumerate(frames):
        arr = np.array(dets, dtype=float).reshape(-1, 5)
        out.append({t.track_id: t.tlbr.copy() for t in tracker.update(arr, now=i / fps)})
    return out


def test_iou_matrix_basic():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0]])


def test_single_person_keeps_one_id():
    tr = ByteTracker()
    out = run(tr, [[det(10 + 6 * i, 50)] for i in range(30)])
    ids = set().union(*[set(o) for o in out])
    assert ids == {1}
    assert all(len(o) == 1 for o in out)


def test_new_track_needs_two_frames_after_first_frame():
    tr = ByteTracker()
    run(tr, [[det(10, 50)]] * 3)
    # A second person appears at frame 4: not reported on its first frame, reported on the next.
    o1 = {t.track_id for t in tr.update(np.array([det(10, 50), det(300, 50)]), now=0.3)}
    o2 = {t.track_id for t in tr.update(np.array([det(10, 50), det(302, 50)]), now=0.4)}
    assert o1 == {1}
    assert o2 == {1, 2}


def test_single_frame_false_positive_is_never_reported():
    tr = ByteTracker()
    frames = [[det(10, 50)] for _ in range(10)]
    frames[5] = [det(10, 50), det(400, 300)]
    out = run(tr, frames)
    assert set().union(*[set(o) for o in out]) == {1}


def test_short_occlusion_keeps_id():
    tr = ByteTracker(max_lost_seconds=2.0)
    frames = [[det(10 + 8 * i, 50)] for i in range(10)]
    frames += [[] for _ in range(6)]                    # hidden for 0.6 s
    frames += [[det(10 + 8 * i, 50)] for i in range(16, 26)]
    out = run(tr, frames)
    ids = set().union(*[set(o) for o in out])
    assert ids == {1}
    assert out[-1] and 1 in out[-1]


def test_long_disappearance_gets_new_id():
    tr = ByteTracker(max_lost_seconds=1.0)
    frames = [[det(10, 50)] for _ in range(5)] + [[] for _ in range(20)] + [[det(10, 50)] for _ in range(5)]
    out = run(tr, frames)
    assert set().union(*[set(o) for o in out]) == {1, 2}


def test_low_score_detection_keeps_partially_occluded_person():
    tr = ByteTracker(high_thresh=0.5, low_thresh=0.1)
    frames = [[det(10 + 5 * i, 50, score=0.9)] for i in range(5)]
    frames += [[det(10 + 5 * i, 50, score=0.2)] for i in range(5, 12)]   # score drops (occlusion)
    frames += [[det(10 + 5 * i, 50, score=0.9)] for i in range(12, 15)]
    out = run(tr, frames)
    assert all(set(o) == {1} for o in out)


def test_low_score_detections_never_create_tracks():
    tr = ByteTracker(high_thresh=0.5, low_thresh=0.1)
    out = run(tr, [[det(10, 50, score=0.3)] for _ in range(10)])
    assert all(o == {} for o in out)


def test_two_people_passing_each_other_keep_ids():
    tr = ByteTracker()
    frames = []
    for i in range(40):
        a = det(20 + 10 * i, 60)            # walks right
        b = det(420 - 10 * i, 90)           # walks left, slightly lower (different depth)
        frames.append([a, b])
    out = run(tr, frames)
    first = out[2]
    last = out[-1]
    # identify by x position: at the start id with the smaller x walks right
    right_id = min(first, key=lambda k: first[k][0])
    left_id = max(first, key=lambda k: first[k][0])
    assert set(last) == {right_id, left_id}
    assert last[right_id][0] > last[left_id][0]   # the right-walker is now on the right


def test_reset_clears_ids():
    tr = ByteTracker()
    run(tr, [[det(10, 50)]] * 3)
    tr.reset()
    out = run(tr, [[det(10, 50)]] * 3)
    assert set(out[-1]) == {1}
