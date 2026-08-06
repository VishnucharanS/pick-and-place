"""
Robotic depalletizing vision pipeline.

Pipeline stages:
  1. One-time manual calibration: user draws the pallet ROI polygon and the
     system extracts the pallet base plane via RANSAC plane fitting.
  2. Per-frame: point cloud is restricted to the ROI, points above the pallet
     plane are isolated, clustered (DBSCAN), and iteratively plane-fit
     (RANSAC) to reject tilted/vertical noise (tripod legs, box side-walls).
  3. The highest-scoring flat cluster is turned into a 2D mask/bbox/centroid
     via morphology, then used to prompt SAM 2 for the final pick mask.

Requires: opencv-python, numpy, open3d, torch, and a local sam2 install.
"""

import glob
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import torch

# ==========================================
# CONFIGURATION
# ==========================================
RGB_DIR = "percipio_captures1/rgb_images/"
DEPTH_DIR = "percipio_captures1/depth_npy/"

SAM2_CHECKPOINT = "sam2/checkpoints/sam2.1_hiera_base_plus.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# Geometry thresholds (kept centralized instead of scattered as magic numbers)
PALLET_PLANE_DIST_THRESHOLD = 0.02      # RANSAC inlier distance for pallet plane (m)
FLOOR_GAP = 0.04                        # min height above pallet to count as "box" (m)

# NOTE on CLUSTER_EPS / SUBPLANE_DIST_THRESHOLD: tightening these reduces false
# merges between adjacent same-height boxes, but risks over-segmenting a single
# large box (e.g. one with a visible seam/logo band) into multiple clusters.
# We lean slightly tighter than before and rely on the seam-detection +
# rectangularity check below to catch the merge cases these thresholds miss,
# rather than chasing the merge problem with distance thresholds alone.
CLUSTER_EPS = 0.05                      # DBSCAN neighborhood radius (m) - was 0.08
CLUSTER_MIN_POINTS = 10
TOP_N_CLUSTERS = 3                      # only RANSAC-refine the N clusters closest to the camera
CLUSTER_MIN_SIZE = 20                   # ignore clusters smaller than this before ranking
SUBPLANE_DIST_THRESHOLD = 0.015         # RANSAC inlier distance for per-cluster planes (m) - was 0.02
MAX_SUBPLANES_PER_CLUSTER = 5
MIN_SUBPLANE_INLIERS = 15
MAX_TILT_DEG = 25                       # reject planes tilted more than this from pallet normal
HEIGHT_NORM_CAP = 1.5                   # meters; used to normalize the height score term
AREA_NORM_CAP = 200                     # points; used to normalize the area score term
SCORE_WEIGHTS = (0.90, 0.05, 0.05)      # (height, flatness, area)

# Rectangularity sanity check: a single box's 2D projection should be a fairly
# solid rectangle. A low fill ratio (contour area / bbox area) is a signal
# that the bbox spans a gap or seam between two separate boxes.
RECTANGULARITY_MIN_FILL = 0.55

# RGB seam detection (Canny + Hough) used to split a suspicious bbox in 2D
# before it's ever handed to SAM.
SEAM_CANNY_LOW = 50
SEAM_CANNY_HIGH = 150
SEAM_MIN_LINE_FRACTION = 0.65           # seam line must span at least this fraction of the crop dimension
SEAM_ANGLE_TOL_DEG = 10                 # how close to purely vertical/horizontal a seam line must be

SAM_UNDER_SEGMENT_RATIO = 0.20          # if SAM mask area < this fraction of geom bbox, retry with box prompt
SAM_OVER_SEGMENT_RATIO = 1.6            # if SAM mask area > this multiple of geom bbox, likely merged two boxes
AUTO_MASK_MIN_AREA_FRACTION = 0.05      # ignore tiny automatic-mask proposals relative to the crop area
BBOX_PAD_RATIO = 0.15

WINDOW_SCALE = 0.5
CALIBRATION_WINDOW_SCALE = 0.6


# ==========================================
# CALIBRATION UI
# ==========================================
class PalletROISelector:
    """Interactive 4-point polygon picker for the pallet region."""

    def __init__(self):
        self.corners: list[tuple[int, int]] = []

    def _on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.corners) < 4:
            self.corners.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and self.corners:
            self.corners.pop()

    def select(self, image: np.ndarray) -> np.ndarray:
        """Show the image, let the user click 4 corners, return the ROI mask."""
        h, w = image.shape[:2]
        window = "Calibration - click 4 pallet corners, ENTER to confirm, C to clear"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, int(w * CALIBRATION_WINDOW_SCALE), int(h * CALIBRATION_WINDOW_SCALE))
        cv2.setMouseCallback(window, self._on_click)

        while True:
            preview = image.copy()
            for i, pt in enumerate(self.corners):
                cv2.circle(preview, pt, 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(preview, self.corners[i - 1], pt, (0, 255, 0), 2)
            if len(self.corners) == 4:
                cv2.line(preview, self.corners[3], self.corners[0], (0, 255, 0), 2)
                cv2.putText(preview, "Press ENTER to confirm", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow(window, preview)
            key = cv2.waitKey(15) & 0xFF
            if key == 13 and len(self.corners) == 4:
                break
            if key in (ord('c'), ord('C')):
                self.corners = []

        cv2.destroyWindow(window)

        mask = np.zeros((h, w), dtype=np.uint8)
        poly = np.array(self.corners, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [poly], 255)
        return mask, poly


# ==========================================
# SEGMENTATION MODEL INTERFACE
# ==========================================
class BaseSegmenter(ABC):
    @abstractmethod
    def load_model(self): ...

    @abstractmethod
    def predict(self, rgb_image, box_prompt, point_prompts): ...


class SAM2Segmenter(BaseSegmenter):
    def __init__(self, checkpoint_path: str, config_path: str):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.predictor = None
        self._sam2_model = None
        self.auto_generator = None  # lazily built - only needed for the disambiguation fallback

    def load_model(self):
        print("[Model] Loading SAM 2 predictor...")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self._sam2_model = build_sam2(self.config_path, self.checkpoint_path)
        self.predictor = SAM2ImagePredictor(self._sam2_model)
        print("[Model] SAM 2 ready.")

    def predict(self, rgb_image: np.ndarray, box_prompt: Optional[list], point_prompts: list):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(rgb_image)
            labels = np.ones(len(point_prompts), dtype=np.int32)
            box_input = np.array(box_prompt) if box_prompt is not None else None

            masks, scores, _ = self.predictor.predict(
                point_coords=np.array(point_prompts),
                point_labels=labels,
                box=box_input,
                multimask_output=False,
            )
        return masks[0], scores[0]

    def _ensure_auto_generator(self):
        if self.auto_generator is None:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            print("[Model] Lazily initializing SAM 2 automatic mask generator...")
            self.auto_generator = SAM2AutomaticMaskGenerator(self._sam2_model)

    def predict_auto_best(self, rgb_image: np.ndarray, target_point_local: tuple, expected_area: float):
        """Runs SAM's automatic mask generator over a small crop and returns
        the proposal that both contains the geometric pick point and best
        matches the expected (RANSAC-derived) area. Used as a disambiguation
        fallback when point/box prompting produces an implausible mask size
        (e.g. two boxes merged into one mask, or a mask smaller than a single
        box's true area)."""
        self._ensure_auto_generator()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            proposals = self.auto_generator.generate(rgb_image)

        if not proposals:
            return None

        crop_area = rgb_image.shape[0] * rgb_image.shape[1]
        tx, ty = int(target_point_local[0]), int(target_point_local[1])
        h, w = rgb_image.shape[:2]
        if not (0 <= tx < w and 0 <= ty < h):
            return None

        best_mask, best_score = None, -1.0
        for proposal in proposals:
            seg = proposal["segmentation"]
            area = proposal["area"]
            if area < AUTO_MASK_MIN_AREA_FRACTION * crop_area:
                continue
            if not seg[ty, tx]:
                continue  # must actually cover the geometric pick point
            area_ratio = (min(area, expected_area) / max(area, expected_area)) if expected_area > 0 else 0.0
            if area_ratio > best_score:
                best_score = area_ratio
                best_mask = seg
        return best_mask


# ==========================================
# GEOMETRY HELPERS
# ==========================================
@dataclass
class PickTarget:
    geom_rot_rect: tuple
    geom_centroid: tuple
    sam_bbox: tuple
    sam_pad: tuple
    tape_void_mask: np.ndarray
    core_mask: np.ndarray           # full-image binary mask of the isolated box region
    fill_ratio: float               # contour area / bbox area - low value hints at a merged bbox


class Hybrid3DLocator:
    """Finds the highest, flattest, tilt-consistent surface in a point cloud
    and turns it into a 2D prompt/bbox for SAM 2."""

    def __init__(self, pad_ratio: float = BBOX_PAD_RATIO):
        self.pad_ratio = pad_ratio

    @staticmethod
    def project_to_2d(x, y, z, img_w, img_h):
        """Pinhole projection back to pixel coordinates."""
        fx = fy = max(img_h, img_w) * 0.8
        cx, cy = img_w / 2.0, img_h / 2.0

        u = (x * fx / z) + cx
        v = (y * fy / z) + cy

        u = np.clip(np.round(u).astype(np.int32), 0, img_w - 1)
        v = np.clip(np.round(v).astype(np.int32), 0, img_h - 1)
        return u, v

    @staticmethod
    def align_plane_to_camera(plane_eq):
        """Ensures the plane normal points toward the camera (+Z convention)."""
        a, b, c, d = plane_eq
        return (-a, -b, -c, -d) if c > 0 else (a, b, c, d)

    @staticmethod
    def signed_height_above_plane(x, y, z, plane_eq):
        a, b, c, d = plane_eq
        num = a * x + b * y + c * z + d
        den = math.sqrt(a ** 2 + b ** 2 + c ** 2)
        return num / den

    def _score_subplane(self, inlier_count, total_in_cluster, avg_height):
        norm_height = min(avg_height / HEIGHT_NORM_CAP, 1.0)
        flatness = inlier_count / total_in_cluster
        norm_area = min(inlier_count / AREA_NORM_CAP, 1.0)
        w_h, w_f, w_a = SCORE_WEIGHTS
        return w_h * norm_height + w_f * flatness + w_a * norm_area

    def find_target(self, pcd, pallet_plane, roi_mask, img_shape) -> Optional[PickTarget]:
        img_h, img_w = img_shape
        pts = np.asarray(pcd.points)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        # 1. Restrict to the calibrated pallet ROI
        u, v = self.project_to_2d(x, y, z, img_w, img_h)
        in_roi = roi_mask[v, u] == 255
        x, y, z, u, v = x[in_roi], y[in_roi], z[in_roi], u[in_roi], v[in_roi]
        if len(z) < 10:
            return None

        # 2. Keep only points clearly above the pallet plane
        heights = self.signed_height_above_plane(x, y, z, pallet_plane)
        print(f"  [Debug] Height range above pallet: {heights.min():.3f} .. {heights.max():.3f} m")

        above = heights > FLOOR_GAP
        bx, by, bz, bu, bv = x[above], y[above], z[above], u[above], v[above]
        b_heights = heights[above]
        if len(bz) < 10:
            return None

        # 3. Cluster into candidate objects
        box_pcd = o3d.geometry.PointCloud()
        box_pcd.points = o3d.utility.Vector3dVector(np.column_stack((bx, by, bz)))
        labels = np.array(box_pcd.cluster_dbscan(
            eps=CLUSTER_EPS, min_points=CLUSTER_MIN_POINTS, print_progress=False))
        if labels.size == 0 or labels.max() < 0:
            return None

        pallet_normal = np.array(pallet_plane[:3])
        pallet_normal /= np.linalg.norm(pallet_normal)

        # 3b. Cheap pre-ranking: pick only the top-N clusters closest to the
        # camera (= highest above the pallet plane) before running any RANSAC
        # on them. This is a single percentile lookup per cluster, so it's
        # far cheaper than the iterative plane-fitting we'd otherwise run on
        # every cluster in the scene, including ones near the pallet floor
        # that could never out-score a top-layer box anyway.
        candidate_labels = []
        for lbl in np.unique(labels[labels >= 0]):
            cluster_idx = np.where(labels == lbl)[0]
            if len(cluster_idx) < CLUSTER_MIN_SIZE:
                continue
            proximity = np.percentile(b_heights[cluster_idx], 90)
            candidate_labels.append((lbl, proximity))

        if not candidate_labels:
            return None

        candidate_labels.sort(key=lambda item: item[1], reverse=True)
        top_labels = [lbl for lbl, _ in candidate_labels[:TOP_N_CLUSTERS]]
        print(f"  [Debug] {len(candidate_labels)} clusters found -> "
              f"refining top {len(top_labels)} closest to camera: {top_labels}")

        best_score = -1.0
        best_uv = None

        print("\n  --- Target Scoring Matrix ---")
        for lbl in top_labels:
            cluster_idx = np.where(labels == lbl)[0]

            remaining_pcd = box_pcd.select_by_index(cluster_idx)
            remaining_idx = cluster_idx.copy()
            plane_count = 0

            # Iteratively peel planes out of the cluster to separate stacked
            # / touching surfaces at different heights and reject tilted noise.
            while len(remaining_pcd.points) >= 15 and plane_count < MAX_SUBPLANES_PER_CLUSTER:
                plane_model, inliers = remaining_pcd.segment_plane(
                    distance_threshold=SUBPLANE_DIST_THRESHOLD, ransac_n=3, num_iterations=500)
                if len(inliers) < MIN_SUBPLANE_INLIERS:
                    break

                cluster_normal = np.array(plane_model[:3])
                cluster_normal /= np.linalg.norm(cluster_normal)
                tilt_deg = math.degrees(math.acos(
                    np.clip(abs(np.dot(pallet_normal, cluster_normal)), 0.0, 1.0)))

                if tilt_deg > MAX_TILT_DEG:
                    # Tilted surface (side-wall, tripod leg, debris) - discard and keep searching
                    remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
                    remaining_idx = np.delete(remaining_idx, inliers)
                    continue

                plane_global_idx = remaining_idx[inliers]
                avg_height = np.percentile(b_heights[plane_global_idx], 90)
                score = self._score_subplane(len(inliers), len(remaining_idx), avg_height)

                print(f"  [Cluster {lbl}-P{plane_count}] pts={len(inliers):>3} "
                      f"height={avg_height:.3f}m tilt={tilt_deg:>4.1f}deg score={score:.3f}")

                if score > best_score:
                    best_score = score
                    best_uv = (bu[plane_global_idx], bv[plane_global_idx])

                remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
                remaining_idx = np.delete(remaining_idx, inliers)
                plane_count += 1

        if best_uv is None:
            return None
        print(f"  => Winner score: {best_score:.3f}")

        return self._build_pick_target(*best_uv, img_shape)

    def _build_pick_target(self, inlier_u, inlier_v, img_shape) -> Optional[PickTarget]:
        """Turn the winning cluster's projected points into a clean 2D mask,
        bbox, and a tape-safe pick centroid."""
        raw_dots = np.zeros(img_shape, dtype=np.uint8)
        raw_dots[inlier_v, inlier_u] = 255

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        solid_mask = cv2.morphologyEx(raw_dots, cv2.MORPH_CLOSE, close_kernel)

        # Erode to snap apart flush-touching boxes, then re-inflate just the
        # winning connected core so we recover its true boundary.
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        eroded_mask = cv2.erode(solid_mask, erode_kernel, iterations=1)
        contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Erosion wiped out a small box entirely - fall back to the unsplit mask
            contours, _ = cv2.findContours(solid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            best_cnt = max(contours, key=cv2.contourArea)
        else:
            best_core_cnt = max(contours, key=cv2.contourArea)
            core_mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.drawContours(core_mask, [best_core_cnt], -1, 255, -1)
            dilated_core = cv2.dilate(core_mask, erode_kernel, iterations=1)
            isolated_mask = cv2.bitwise_and(solid_mask, dilated_core)

            final_contours, _ = cv2.findContours(isolated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not final_contours:
                return None
            best_cnt = max(final_contours, key=cv2.contourArea)

        bx, by, bw, bh = cv2.boundingRect(best_cnt)
        if bw < 15 or bh < 15:
            return None

        geom_rot_rect = cv2.minAreaRect(best_cnt)
        fill_ratio = cv2.contourArea(best_cnt) / (bw * bh) if bw * bh > 0 else 0.0

        # A light close on the raw (unhealed) dots preserves gaps caused by
        # tape/reflective surfaces so we never place the pick point on tape.
        tiny_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        tape_void_mask = cv2.morphologyEx(raw_dots, cv2.MORPH_CLOSE, tiny_close)

        core_mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(core_mask, [best_cnt], -1, 255, -1)
        safe_prompt_mask = cv2.bitwise_and(core_mask, tape_void_mask)

        dist_map = cv2.distanceTransform(safe_prompt_mask, cv2.DIST_L2, 5)
        _, _, _, geom_centroid = cv2.minMaxLoc(dist_map)

        pad_w, pad_h = int(bw * self.pad_ratio), int(bh * self.pad_ratio)

        return PickTarget(
            geom_rot_rect=geom_rot_rect,
            geom_centroid=geom_centroid,
            sam_bbox=(bx, by, bw, bh),
            sam_pad=(pad_w, pad_h),
            tape_void_mask=tape_void_mask,
            core_mask=core_mask,
            fill_ratio=fill_ratio,
        )


# ==========================================
# RGB SEAM SPLITTING
# ==========================================
def detect_seam_line(gray_crop: np.ndarray) -> Optional[tuple]:
    """Looks for a single strong, near-full-length straight line (Canny +
    HoughLinesP) running mostly vertically or horizontally through the crop.
    A depth-derived bbox with a low fill ratio often contains exactly this:
    a real seam between two adjacent boxes that RANSAC's plane fit couldn't
    tell apart because both surfaces are coplanar. Returns (axis, position)
    in crop-local pixel coordinates, or None if no convincing seam is found."""
    h, w = gray_crop.shape[:2]
    if h < 20 or w < 20:
        return None

    edges = cv2.Canny(gray_crop, SEAM_CANNY_LOW, SEAM_CANNY_HIGH)
    min_len = int(min(h, w) * SEAM_MIN_LINE_FRACTION)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                             minLineLength=min_len, maxLineGap=10)
    if lines is None:
        return None

    best, best_len = None, 0
    for (x1, y1, x2, y2) in lines[:, 0]:
        length = math.hypot(x2 - x1, y2 - y1)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180

        is_vertical = abs(angle - 90) < SEAM_ANGLE_TOL_DEG and length >= h * SEAM_MIN_LINE_FRACTION
        is_horizontal = (angle < SEAM_ANGLE_TOL_DEG or angle > 180 - SEAM_ANGLE_TOL_DEG) \
            and length >= w * SEAM_MIN_LINE_FRACTION

        if is_vertical and length > best_len:
            best_len, best = length, ("vertical", (x1 + x2) // 2)
        elif is_horizontal and length > best_len:
            best_len, best = length, ("horizontal", (y1 + y2) // 2)

    return best


def refine_target_with_seam_split(img_bgr: np.ndarray, target: PickTarget) -> PickTarget:
    """If the target's 2D footprint looks under-filled (a rectangle that
    isn't actually mostly solid), search its crop for a straight seam line
    and, if found, shrink the target down to just the half containing the
    geometric pick point. This catches the case where RANSAC's plane fit
    merged two coplanar, closely-spaced box tops into one bbox."""
    if target.fill_ratio >= RECTANGULARITY_MIN_FILL:
        return target

    x, y, w, h = target.sam_bbox
    pad_w, pad_h = target.sam_pad
    img_h, img_w = img_bgr.shape[:2]
    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
    x2, y2 = min(img_w, x + w + pad_w), min(img_h, y + h + pad_h)

    gray_crop = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    seam = detect_seam_line(gray_crop)
    if seam is None:
        return target

    axis, pos = seam
    gc_x, gc_y = target.geom_centroid
    crop_h, crop_w = gray_crop.shape

    half_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    if axis == "vertical":
        half_mask[:, :pos] = 255 if (gc_x - x1) < pos else 0
        half_mask[:, pos:] = 255 if (gc_x - x1) >= pos else 0
    else:
        half_mask[:pos, :] = 255 if (gc_y - y1) < pos else 0
        half_mask[pos:, :] = 255 if (gc_y - y1) >= pos else 0

    full_half_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    full_half_mask[y1:y2, x1:x2] = half_mask

    restricted_core = cv2.bitwise_and(target.core_mask, full_half_mask)
    contours, _ = cv2.findContours(restricted_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return target

    best_cnt = max(contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(best_cnt)
    if bw < 15 or bh < 15:
        return target

    new_rot_rect = cv2.minAreaRect(best_cnt)
    new_fill_ratio = cv2.contourArea(best_cnt) / (bw * bh) if bw * bh > 0 else 0.0

    safe_mask = cv2.bitwise_and(restricted_core, target.tape_void_mask)
    dist_map = cv2.distanceTransform(safe_mask, cv2.DIST_L2, 5)
    _, _, _, new_centroid = cv2.minMaxLoc(dist_map)

    new_pad_w, new_pad_h = int(bw * BBOX_PAD_RATIO), int(bh * BBOX_PAD_RATIO)

    print(f"    -> Seam detected ({axis} @ crop-px {pos}, fill_ratio was {target.fill_ratio:.2f}); "
          f"splitting bbox to isolate a single box.")

    return PickTarget(
        geom_rot_rect=new_rot_rect,
        geom_centroid=new_centroid,
        sam_bbox=(bx, by, bw, bh),
        sam_pad=(new_pad_w, new_pad_h),
        tape_void_mask=target.tape_void_mask,
        core_mask=restricted_core,
        fill_ratio=new_fill_ratio,
    )


# ==========================================
# VISUALIZATION HELPERS
# ==========================================
def draw_geometry_overlay(img_bgr, roi_poly, target: PickTarget):
    overlay = img_bgr.copy()
    cv2.polylines(overlay, [roi_poly], True, (0, 255, 0), 2)
    box_pts = np.int32(cv2.boxPoints(target.geom_rot_rect))
    cv2.drawContours(overlay, [box_pts], 0, (0, 255, 255), 2)
    cv2.circle(overlay, target.geom_centroid, 6, (255, 255, 255), -1)
    return overlay


def run_sam_on_target(segmenter: SAM2Segmenter, img_bgr, target: PickTarget, img_w, img_h):
    """Runs SAM 2 on the padded crop around the target, in three escalating
    tiers:
      1. Point-only prompt at the geometric centroid (lets SAM trace real
         RGB edges freely).
      2. If under-segmented (mask much smaller than the geometric bbox),
         retry with an added box prompt spanning the full bbox.
      3. If the result is still implausible in either direction - too small
         (missed most of the box) or too large (likely merged two boxes into
         one mask) - fall back to SAM's automatic mask generator and pick
         whichever proposal both contains the pick point and best matches
         the RANSAC-derived expected area.
    """
    x, y, w, h = target.sam_bbox
    pad_w, pad_h = target.sam_pad
    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
    x2, y2 = min(img_w, x + w + pad_w), min(img_h, y + h + pad_h)

    cropped_rgb = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    gc_x, gc_y = target.geom_centroid
    local_point = (gc_x - x1, gc_y - y1)
    local_points = [list(local_point)]

    local_mask, _ = segmenter.predict(cropped_rgb, None, local_points)

    geom_area = w * h
    sam_area = np.sum(local_mask)
    ratio = sam_area / geom_area if geom_area else 0
    if ratio < SAM_UNDER_SEGMENT_RATIO:
        print(f"    -> SAM point-only under-segmented (ratio={ratio:.2f}); retrying with box prompt.")
        local_box = [x - x1, y - y1, x + w - x1, y + h - y1]
        local_mask, _ = segmenter.predict(cropped_rgb, local_box, local_points)
        sam_area = np.sum(local_mask)
        ratio = sam_area / geom_area if geom_area else 0
    else:
        print(f"    -> SAM point-only succeeded (ratio={ratio:.2f}).")

    if ratio < SAM_UNDER_SEGMENT_RATIO or ratio > SAM_OVER_SEGMENT_RATIO:
        reason = "under-segmented" if ratio < SAM_UNDER_SEGMENT_RATIO else "over-segmented (likely merged boxes)"
        print(f"    -> Still {reason} (ratio={ratio:.2f}); trying automatic-mask disambiguation.")
        auto_mask = segmenter.predict_auto_best(cropped_rgb, local_point, geom_area)
        if auto_mask is not None:
            auto_area = np.sum(auto_mask)
            auto_ratio = auto_area / geom_area if geom_area else 0
            if SAM_UNDER_SEGMENT_RATIO <= auto_ratio <= SAM_OVER_SEGMENT_RATIO:
                print(f"    -> Automatic mask matching improved the result (ratio={auto_ratio:.2f}).")
                local_mask = auto_mask
            else:
                print(f"    -> Automatic mask matching didn't help (ratio={auto_ratio:.2f}); keeping prior mask.")
        else:
            print("    -> No automatic-mask proposal contained the pick point; keeping prior mask.")

    global_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    global_mask[y1:y2, x1:x2] = (local_mask * 255).astype(np.uint8)
    return global_mask


def draw_sam_overlay(img_bgr, global_sam_mask, tape_void_mask):
    """Draws the SAM contour and the tape-safe pick centroid. Returns the
    rendered image (falls back to a plain copy if no contour is found)."""
    sam_img = img_bgr.copy()
    contours, _ = cv2.findContours(global_sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return sam_img

    sam_cnt = max(contours, key=cv2.contourArea)
    sam_rot_rect = cv2.minAreaRect(sam_cnt)
    box_pts = np.int32(cv2.boxPoints(sam_rot_rect))
    cv2.drawContours(sam_img, [box_pts], 0, (255, 0, 255), 2)

    # Punch tape voids into the SAM mask so the pick point never lands on tape/reflection.
    safe_mask = cv2.bitwise_and(global_sam_mask, tape_void_mask)
    dist_map = cv2.distanceTransform(safe_mask, cv2.DIST_L2, 5)
    _, _, _, safe_centroid = cv2.minMaxLoc(dist_map)

    cv2.circle(sam_img, safe_centroid, 8, (255, 255, 255), -1)
    cv2.circle(sam_img, safe_centroid, 4, (0, 0, 255), -1)
    cv2.putText(sam_img, "Safe Pick Center", (safe_centroid[0] + 15, safe_centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    overlay = sam_img.copy()
    overlay[safe_mask == 255] = [0, 255, 0]
    return cv2.addWeighted(sam_img, 0.6, overlay, 0.4, 0)


# ==========================================
# MAIN PIPELINE
# ==========================================
def calibrate_pallet_plane(locator: Hybrid3DLocator, rgb_path: str, roi_mask: np.ndarray, img_w: int, img_h: int):
    timestamp = os.path.splitext(os.path.basename(rgb_path))[0].replace("rgb_", "")
    pcd_path = os.path.join(DEPTH_DIR, f"depth_{timestamp}.pcd")

    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points)
    px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]

    pu, pv = locator.project_to_2d(px, py, pz, img_w, img_h)
    in_roi = roi_mask[pv, pu] == 255

    roi_pcd = o3d.geometry.PointCloud()
    roi_pcd.points = o3d.utility.Vector3dVector(pts[in_roi])

    plane, _ = roi_pcd.segment_plane(
        distance_threshold=PALLET_PLANE_DIST_THRESHOLD, ransac_n=3, num_iterations=1000)
    return locator.align_plane_to_camera(plane)


def main():
    segmenter = SAM2Segmenter(SAM2_CHECKPOINT, SAM2_CONFIG)
    segmenter.load_model()
    locator = Hybrid3DLocator()

    rgb_files = sorted(glob.glob(os.path.join(RGB_DIR, "*.png")))
    if len(rgb_files) < 3:
        print("Error: need at least 3 images (1 calibration + 2 to process).")
        return

    # --- Calibration ---
    calibration_path = rgb_files[1]
    calib_img = cv2.imread(calibration_path)
    img_h, img_w = calib_img.shape[:2]

    print("\n[Calibration] Draw the pallet region on the empty-pallet image...")
    roi_mask, roi_poly = PalletROISelector().select(calib_img)

    print("[Calibration] Extracting pallet base plane via RANSAC...")
    pallet_plane = calibrate_pallet_plane(locator, calibration_path, roi_mask, img_w, img_h)

    # --- Processing loop ---
    win_geom, win_sam = "1. RANSAC 3D Geometry", "2. AI Safe Pick Extraction"
    for win in (win_geom, win_sam):
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, int(img_w * WINDOW_SCALE), int(img_h * WINDOW_SCALE))

    for idx in range(2, len(rgb_files)):
        rgb_path = rgb_files[idx]
        timestamp = os.path.splitext(os.path.basename(rgb_path))[0].replace("rgb_", "")
        pcd_path = os.path.join(DEPTH_DIR, f"depth_{timestamp}.pcd")

        print(f"\n==============================================")
        print(f"Processing [{idx + 1}/{len(rgb_files)}]: rgb_{timestamp}.png")

        img_bgr = cv2.imread(rgb_path)
        pcd = o3d.io.read_point_cloud(pcd_path)

        target = locator.find_target(pcd, pallet_plane, roi_mask, (img_h, img_w))
        if target is None:
            print(" -> No valid geometric target found.")
            cv2.imshow(win_geom, img_bgr)
            cv2.imshow(win_sam, img_bgr)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
            continue

        target = refine_target_with_seam_split(img_bgr, target)
        geom_img = draw_geometry_overlay(img_bgr, roi_poly, target)
        global_sam_mask = run_sam_on_target(segmenter, img_bgr, target, img_w, img_h)
        sam_final = draw_sam_overlay(img_bgr, global_sam_mask, target.tape_void_mask)

        cv2.imshow(win_geom, geom_img)
        cv2.imshow(win_sam, sam_final)

        print(" -> Press any key for next image, 'q' to quit.")
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()