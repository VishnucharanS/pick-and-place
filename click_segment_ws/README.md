# Click-to-Segment: Multi-Method Point Cloud Segmentation

A click-to-segment pipeline for point clouds: click an object (in 2D or directly in 3D), and have it segmented out of a cluttered tabletop scene. Built and benchmarked across **four segmentation approaches on the same data**, with rigorous, automated metrics and explicit failure-case analysis — can be extended into localization, motion planning, and grasp planning.

---

## Dataset
 
[YCB-Video](https://rse-lab.cs.washington.edu/projects/posecnn/), accessed via the [BOP benchmark's](https://bop.felk.cvut.cz/) curated test subset (`ycbv_test_bop19`). Chosen over the full YCB-V dataset because the BOP subset is smaller, removes redundant/erroneous-ground-truth frames, and avoids known data-leakage issues present in the original synthetic training split. Chosen over outdoor LiDAR datasets (e.g. KITTI) to stay consistent with the project's longer-term grasp-planning direction - RGB-D-derived point cloud domain is a better fit than outdoor LiDAR scans for that goal.
 
12 scenes (`test/000048`–`000059`), 75 frames per scene, totaling 900 evaluation images, each with RGB, depth, amodal and visible-only ground-truth masks, per-frame camera intrinsics/extrinsics, and object pose annotations.

---

## 🛠️ System Architecture & Methods

click_segment_ws/
├── src/
│   ├── click_segment_core/      # pure Python, zero ROS dependencies — all math/CV/ML logic
├── scripts/                     # throwaway test/demo scripts exercising click_segment_core directly
└── data/ycbv/                   # dataset

## Methods
 
### 1. Geometric Clustering (classical baseline)
RANSAC dominant-plane removal (strips the tabletop) followed by DBSCAN spatial clustering. A click is mapped to whichever DBSCAN cluster contains it.
 
**Strength:** no learned model, fast, fully interpretable.
**Weakness:** relies purely on Euclidean proximity — cannot separate objects that are physically touching, since DBSCAN has no concept of object identity beyond spatial contiguity.
 
### 2. SAM2 (2D foundation model → 3D projection)
The 3D click is projected into the corresponding 2D pixel via the camera intrinsics. SAM2 runs point-prompted segmentation on the RGB image (selecting the **largest-area mask** among its three candidate outputs, to favor whole-object segmentation over the sub-part masks SAM tends to produce on a single click). The resulting 2D mask is back-projected into 3D using the depth image. Checkpoint used: "sam2.1_hiera_base_plus.pt"
 
**Strength:** best performer by a clear margin (see Results). Leverages dense 2D visual/texture features that no untextured 3D geometric method has access to, giving crisp object-boundary localization even when objects are touching.
**Weakness:** purely 2D-grounded — no inherent 3D structural awareness; relies entirely on depth-image quality for the back-projection step.
 
### 3. PointNet++ (frozen pretrained backbone)
A pretrained PointNet++ part-segmentation backbone ([yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch), trained on ShapeNetPart), used frozen, on a click-radius-cropped local region of the scene.
 
**Result: underperforms relative to SAM2 and (on isolated objects) Geometric Clustering.** Two specific causes were diagnosed, not assumed:
1. **Crop truncation** — the fixed-radius spherical crop around the click point visually clips the extent of larger objects before the model ever sees them, confirmed by direct visualization of the cropped region.
2. **No meaningful category conditioning** — the model is category-conditioned (ShapeNetPart's 16 classes: airplane, bag, cap, car, chair, etc.), none of which correspond to YCB-V's bottles/boxes/tools. A full sweep across all 16 category indices showed no semantically meaningful choice rescues performance — variation across indices is noise, not signal.
**Deliberate scope decision:** Fine-Tuning could have made the model much better, but due to space and GPU constraints I limited this model with pretrained weights.
 
### 4. Hybrid (SAM2 + DBSCAN-based denoising)
An attempt to combine SAM2's clean boundaries with DBSCAN's full 3D extent. Two versions were built and tested:
 
- **v1 (denoise only):** filters SAM2's projected 3D points, keeping only those with at least one neighboring point in the full scene cloud within a small radius — removing sparse/floating noise from the SAM2 mask's back-projection.
- **v2 (DBSCAN backfill, tested and reverted):** additionally backfilled points from the clicked object's DBSCAN cluster that were near an existing SAM2 point, to recover real boundary points SAM2 missed. This improved results on isolated objects but caused a specific, diagnosed failure: at occlusion boundaries, a DBSCAN cluster can already span two real objects (since occlusion-split fragments of one object are sometimes contiguous, in 3D, with a different neighboring object). Backfilling from such a cluster reintroduced contamination from the *wrong* object — directly undermining the reason SAM2 was being used in the first place. Hence, **v2 was reverted**; v1 (denoise-only) is the shipped hybrid method.
This is reported as a genuine, evidence-backed finding about the limits of geometry-based backfill near occlusion boundaries, not a discarded experiment.

## Results
 
Automated benchmark across all 12 dataset scenes. For each ground-truth object in each frame, a click point is generated automatically from the visible-mask centroid (back-projected to 3D via depth + intrinsics), ensuring every method is evaluated on an identical input per object — not a manually chosen, cherry-picked click.
 
| Scene             | Geometric | PointNet++ |  SAM | Hybrid |
|-------------------|----------:|-----------:|-----:|-------:|
| 000048            |      15.9 |       37.4 | 40.2 |   36.4 |
| 000049            |      29.9 |       38.3 | 66.7 |   61.3 |
| 000050            |      73.2 |       47.5 | 80.6 |   72.1 |
| 000051            |      57.4 |       45.1 | 75.4 |   69.7 |
| 000052            |      41.1 |       39.2 | 71.4 |   66.5 |
| 000053            |      56.0 |       54.6 | 73.5 |   69.4 |
| 000054            |      45.3 |       45.1 | 75.6 |   73.9 |
| 000055            |      54.3 |       56.2 | 67.2 |   63.8 |
| 000056            |      60.4 |       41.3 | 81.2 |   72.0 |
| 000057            |      67.4 |       49.3 | 69.7 |   65.8 |
| 000058            |      55.9 |       48.2 | 85.7 |   82.9 |
| 000059            |      42.0 |       35.5 | 70.9 |   60.4 |
| **Overall Mean**  |  **50.0** |   **44.8** | **71.0** | **65.6** |
 
*(Metric: IoU % against `mask_visib` ground truth, averaged per scene across all annotated objects.)*
 
**Headline finding:** SAM2 leads on every single scene, often by a wide margin — confirming that 2D foundation-model features generalize far better to real, cluttered RGB-D scenes than either a geometry-only method or a frozen 3D backbone trained on clean CAD shapes. The denoise-only Hybrid method consistently trails SAM2 slightly (denoising can occasionally strip a few correct boundary points along with noise) but remains the second-strongest method overall. Geometric Clustering and PointNet++ trade places depending on scene clutter density — Geometric Clustering does reasonably well on scenes with fewer touching objects (e.g. 000050, 000057) but degrades sharply on cluttered scenes (e.g. 000048, 000049), consistent with its core limitation.

## Failure-Case Analysis
 
- **Touching/occluded objects** (e.g. scene 000054, frame 1134): Geometric Clustering's DBSCAN step merges multiple physically-touching objects into a single cluster, splitting composition roughly evenly across 3–4 ground-truth objects with no single IoU exceeding ~40%. SAM2 and the Hybrid method both degrade more gracefully here (typically 55–70% IoU) since SAM2's 2D boundary cues remain meaningful even when 3D geometry is ambiguous.
- **Occlusion-split fragments of a single object**: confirmed during Hybrid v2 testing — two visually-separated 3D fragments of the same physical object (split by an occluding neighbor) can fall into a DBSCAN cluster that also touches the occluding object's geometry, making naive cluster-based backfill unsafe.
- **PointNet++ on real depth scans**: consistently weak even on isolated objects, isolating the gap to scene-specific fine-tuning rather than architectural choice, per the diagnosis above.

## How to Install and Run

```bash
# 1. Clone submodules and navigate to workspace root
cd ~/Projects/point_cloud_segmentation/click_segment_ws

# 2. Install the core algorithmic library package in editable mode
pip install -e src/click_segment_core

# 3. Execute individual interactive visualizer testing tools (Press 'Q' to select/close)
python scripts/test_load_frame.py
python scripts/test_sam.py
python scripts/test_pointnet.py
python scripts/test_hybrid.py

"shift+leftclick to select points in o3d"

# 4. Trigger the full background 900-frame evaluation batch run
python scripts/compute_dataset_metrics.py
```