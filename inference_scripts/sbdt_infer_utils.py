# sbdt_infer_utils.py
from __future__ import annotations
from dataclasses import dataclass
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple, Callable

import cv2
import numpy as np
import torch


def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR frame to RGB."""
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def motion_predict_if_consecutive(
    hist: deque[Tuple[int, np.ndarray]],
) -> Optional[np.ndarray]:
    """
    Implements Eq.(4) from WASB, but only if we have positions at t-2,t-1,t (consecutive frame indices).
    hist stores (frame_idx, pos[x,y]) for detections.
    """
    if len(hist) < 3:
        return None

    (t2, p2), (t1, p1), (t0, p0) = hist[-3], hist[-2], hist[-1]
    if not (t2 + 1 == t1 and t1 + 1 == t0):
        return None

    a_t = p0 - 2 * p1 + p2
    v_t = p0 - p1 + a_t
    p_hat = p0 + v_t + 0.5 * a_t
    return p_hat


def extract_blob_candidates_coh(
    heatmap: np.ndarray,
    thr: float = 0.5,
    connectivity: int = 8,
) -> List[Tuple[float, float, float]]:
    """
    Blob-based candidate extraction + CoH (Center of Heatmap values).
    heatmap: float32, shape [H,W], values in [0,1]
    Returns list of (cx, cy, conf), where conf is sum of heatmap values in blob.
    """
    hm = heatmap.astype(np.float32)
    mask = (hm > thr).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)

    cands: List[Tuple[float, float, float]] = []
    for lbl_id in range(1, num_labels):  # skip background
        m = (labels == lbl_id)
        w = hm[m]
        s = float(w.sum())
        if s <= 0:
            continue
        ys, xs = np.where(m)
        cx = float((xs * w).sum() / s)
        cy = float((ys * w).sum() / s)
        cands.append((cx, cy, s))
    return cands


def select_with_gating_maxconf(
    candidates: List[Tuple[float, float, float]],
    pred_xy: Optional[np.ndarray],
    gate_px: Optional[float],
) -> Tuple[int, float, float, float]:
    """
    WASB-style online tracking selection:
      - if pred exists: filter by distance <= gate_px
      - choose max-confidence among remaining
      - if none remain: no detection
    """
    if pred_xy is not None and gate_px is not None:
        px, py = float(pred_xy[0]), float(pred_xy[1])
        kept = []
        for (cx, cy, conf) in candidates:
            if np.hypot(cx - px, cy - py) <= gate_px:
                kept.append((cx, cy, conf))
        candidates = kept

    if not candidates:
        return 0, 0.0, 0.0, 0.0

    cx, cy, conf = max(candidates, key=lambda t: t[2])
    return 1, float(cx), float(cy), float(conf)


def apply_heatmap_overlay(frame: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """
    Apply heatmap overlay to frame following deepball pattern.
    Normalizes heatmap, applies JET colormap, and blends with frame (60% frame, 40% heatmap).
    """
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_normalized = heatmap_normalized.astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    overlayed_frame = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
    return overlayed_frame


@dataclass
class TrackingConfig:
    thr: float = 0.5
    gate_px: float = 80.0        # in ORIGINAL output-frame pixels
    use_gating: bool = True      # turn off if you want "pure detector"
    connectivity: int = 8


class MIMOOverSampler:
    """
    Proper Step=1 oversampling for N=3:
      - sliding window stride=1
      - aggregate candidates per frame across windows
      - flush frame k when we've processed the window ending at k+2 (i.e., when current end t => flush t-2)
    """
    def __init__(self, N: int = 3):
        self.N = N
        self.win = deque(maxlen=N)  # stores (idx, frame_bgr)
        self.frame_store: Dict[int, np.ndarray] = {}
        self.cand_store: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)
        self.heatmap_store: Dict[int, np.ndarray] = {}  # store latest heatmap per frame

    def add_frame(self, idx: int, frame_bgr: np.ndarray) -> bool:
        self.frame_store[idx] = frame_bgr
        self.win.append((idx, frame_bgr))
        return len(self.win) == self.N

    def get_window(self) -> Tuple[List[int], List[np.ndarray]]:
        idxs = [it[0] for it in self.win]
        frames = [it[1] for it in self.win]
        return idxs, frames

    def add_window_candidates(self, idxs: List[int], cand_lists: List[List[Tuple[float, float, float]]]) -> None:
        for i, fidx in enumerate(idxs):
            self.cand_store[fidx].extend(cand_lists[i])

    def add_window_heatmaps(self, idxs: List[int], heatmaps: List[np.ndarray]) -> None:
        """Store the latest heatmap for each frame index."""
        for i, fidx in enumerate(idxs):
            self.heatmap_store[fidx] = heatmaps[i]

    def flush_ready_idx(self, end_idx: int) -> Optional[int]:
        ready = end_idx - (self.N - 1)  # for N=3 => end-2
        if ready in self.frame_store and ready in self.cand_store:
            return ready
        # still allow flush even if no candidates (frame exists but cand list empty)
        if ready in self.frame_store and ready not in self.cand_store:
            self.cand_store[ready] = []
            return ready
        return None

    def pop_ready(self, idx: int) -> Tuple[np.ndarray, List[Tuple[float, float, float]], Optional[np.ndarray]]:
        frame = self.frame_store.pop(idx)
        cands = self.cand_store.pop(idx, [])
        heatmap = self.heatmap_store.pop(idx, None)
        return frame, cands, heatmap

    def flush_remaining_sorted(self) -> List[int]:
        # flush all remaining indices in ascending order
        return sorted(self.frame_store.keys())


@torch.inference_mode()
def run_mimo_heatmap_inference(
    *,
    model: torch.nn.Module,
    cap: cv2.VideoCapture,
    writer: cv2.VideoWriter,
    transform_fn: Callable[[np.ndarray], torch.Tensor],
    forward_fn: Callable[[torch.Tensor], torch.Tensor],  # returns [B,N,h,w] logits
    cfg: TrackingConfig,
    device: torch.device,
    step: int = 3,           # 3 = non-overlap, 1 = oversampling
    batch_windows: int = 8,
    draw: bool = True,
    overlay: bool = False,
    show: bool = False,       # if True, display frames in real-time
) -> List[List[float]]:
    """
    Generic runner for TrackNetV2 / ResTrackNetV2 / MonoTrack / WASB-style models:
      - Input windows: N=3 frames -> [B,9,h,w]
      - Output: logits [B,3,h,w]
      - Postprocess: sigmoid -> resize to original -> blob candidates -> (optional) gating -> maxconf
    Returns coords rows: [frame_idx, detected, x, y, conf]
    """
    N = 3
    coords: List[List[float]] = []

    hist = deque(maxlen=3)  # stores (frame_idx, pos_xy) for detections

    if step not in (1, 3):
        raise ValueError("step must be 1 or 3 for this implementation")

    # Read original dims once
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if step == 3:
        # Non-overlapping windows: process [0,1,2], [3,4,5], ...
        buf_frames: List[np.ndarray] = []
        buf_idxs: List[int] = []
        idx = 0
        should_break = False

        while True:
            ret, frame_bgr = cap.read()
            if not ret or should_break:
                break

            buf_frames.append(frame_bgr)
            buf_idxs.append(idx)
            idx += 1

            if len(buf_frames) < N:
                continue

            # build tensor [1,9,h,w]
            frames_t = [transform_fn(f) for f in buf_frames]          # each [3,h,w]
            inp = torch.cat(frames_t, dim=0).unsqueeze(0).to(device)  # [1,9,h,w]

            logits = forward_fn(inp)                                  # [1,3,h,w]
            probs = torch.sigmoid(logits[0]).detach().cpu().numpy()   # [3,h,w]

            for i in range(N):
                hm = cv2.resize(probs[i].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

                cands = extract_blob_candidates_coh(hm, thr=cfg.thr, connectivity=cfg.connectivity)

                pred = motion_predict_if_consecutive(hist) if cfg.use_gating else None
                gate = cfg.gate_px if cfg.use_gating else None
                det, cx, cy, conf = select_with_gating_maxconf(cands, pred, gate)

                frame_out = buf_frames[i].copy()
                if overlay:
                    frame_out = apply_heatmap_overlay(frame_out, hm)
                if draw and det:
                    cv2.circle(frame_out, (int(cx), int(cy)), 10, (0, 255, 0), 2)

                writer.write(frame_out)
                coords.append([buf_idxs[i], det, cx, cy, conf])

                if det:
                    hist.append((buf_idxs[i], np.array([cx, cy], dtype=np.float32)))

                if show:
                    cv2.imshow("Frame", frame_out)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        should_break = True
                        break

            # clear for next non-overlap
            buf_frames, buf_idxs = [], []

        return coords

    # step == 1: proper oversampling
    overs = MIMOOverSampler(N=N)
    idx = 0
    should_break = False

    pending_inp: List[torch.Tensor] = []
    pending_idxs: List[List[int]] = []
    pending_end: List[int] = []

    def process_batch():
        nonlocal pending_inp, pending_idxs, pending_end, coords, hist, should_break
        if not pending_inp:
            return

        batch = torch.cat(pending_inp, dim=0).to(device)            # [B,9,h,w]
        logits = forward_fn(batch)                                   # [B,3,h,w]
        probs = torch.sigmoid(logits).detach().cpu().numpy()         # [B,3,h,w]

        for b in range(probs.shape[0]):
            if should_break:
                break
            idxs3 = pending_idxs[b]
            end_idx = pending_end[b]

            cand_lists: List[List[Tuple[float, float, float]]] = []
            heatmap_list: List[np.ndarray] = []
            for i in range(N):
                hm = cv2.resize(probs[b, i].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
                cands = extract_blob_candidates_coh(hm, thr=cfg.thr, connectivity=cfg.connectivity)
                cand_lists.append(cands)
                heatmap_list.append(hm)

            overs.add_window_candidates(idxs3, cand_lists)
            if overlay:
                overs.add_window_heatmaps(idxs3, heatmap_list)

            ready = overs.flush_ready_idx(end_idx)
            if ready is not None:
                frame_out, cands, hm_ready = overs.pop_ready(ready)

                pred = motion_predict_if_consecutive(hist) if cfg.use_gating else None
                gate = cfg.gate_px if cfg.use_gating else None
                det, cx, cy, conf = select_with_gating_maxconf(cands, pred, gate)

                if overlay and hm_ready is not None:
                    frame_out = apply_heatmap_overlay(frame_out, hm_ready)
                if draw and det:
                    cv2.circle(frame_out, (int(cx), int(cy)), 10, (0, 255, 0), 2)

                writer.write(frame_out)
                coords.append([ready, det, cx, cy, conf])

                if det:
                    hist.append((ready, np.array([cx, cy], dtype=np.float32)))

                if show:
                    cv2.imshow("Frame", frame_out)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        should_break = True
                        break

        pending_inp, pending_idxs, pending_end = [], [], []

    while True:
        if should_break:
            break
        ret, frame_bgr = cap.read()
        if not ret:
            break

        full = overs.add_frame(idx, frame_bgr)
        if full:
            idxs3, frames3 = overs.get_window()  # consecutive indices

            frames_t = [transform_fn(f) for f in frames3]
            inp = torch.cat(frames_t, dim=0).unsqueeze(0)            # [1,9,h,w]

            pending_inp.append(inp)
            pending_idxs.append(idxs3)
            pending_end.append(idxs3[-1])

            if len(pending_inp) >= batch_windows:
                process_batch()

        idx += 1

    # flush remaining batch
    if not should_break:
        process_batch()

    # flush remaining frames (last 1-2 frames) with partial candidate aggregation
    if not should_break:
        for ridx in overs.flush_remaining_sorted():
            frame_out, cands, hm_ready = overs.pop_ready(ridx)
            pred = motion_predict_if_consecutive(hist) if cfg.use_gating else None
            gate = cfg.gate_px if cfg.use_gating else None
            det, cx, cy, conf = select_with_gating_maxconf(cands, pred, gate)
            if overlay and hm_ready is not None:
                frame_out = apply_heatmap_overlay(frame_out, hm_ready)
            if draw and det:
                cv2.circle(frame_out, (int(cx), int(cy)), 10, (0, 255, 0), 2)
            writer.write(frame_out)
            coords.append([ridx, det, cx, cy, conf])
            if det:
                hist.append((ridx, np.array([cx, cy], dtype=np.float32)))
            
            if show:
                cv2.imshow("Frame", frame_out)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    if show:
        cv2.destroyAllWindows()
    
    coords.sort(key=lambda r: r[0])
    return coords

