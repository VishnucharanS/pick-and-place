import torch
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "sam2/checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

def project_points_to_pixels(points_3d, cam_K, image_shape):
    # Extract focal lengths and principal points from the 3x3 camera matrix
    u = ((cam_K[0, 0] * points_3d[:, 0] / points_3d[:, 2]) + cam_K[0, 2]).astype(int)
    v = ((cam_K[1, 1] * points_3d[:, 1] / points_3d[:, 2]) + cam_K[1, 2]).astype(int)
    valid_indices = (u >= 0) & (u < image_shape[1]) & (v >= 0) & (v < image_shape[0])
    pixel_coords = np.column_stack((u[valid_indices], v[valid_indices]))
    return pixel_coords

def test_sam2_predictor(scene_path, frame_id, input_points):
    rgb = cv2.imread(f"{scene_path}/rgb/{frame_id:06d}.png", cv2.IMREAD_COLOR) 
    rgb_converted = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb_converted)
    combined_mask = np.zeros(rgb.shape[:2], dtype=bool)
    individual_masks = []
    for idx, pt in enumerate(input_points):
        single_point = np.array([[int(pt[0]), int(pt[1])]], dtype=np.float32)
        single_label = np.array([1], dtype=np.int32)
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, logits = predictor.predict(
                point_coords=single_point, 
                point_labels=single_label,
                multimask_output=True
            )
        print(f"Point {idx + 1}: {single_point}, Score: {scores[np.argmax(scores)]:.4f}")
        if hasattr(masks, 'cpu'):
            masks_np = masks.cpu().numpy()
        else:
            masks_np = np.array(masks)
        masks_np = np.squeeze(masks_np)
        mask_areas = [masks_np[m].sum() for m in range(3)]
        largest_mask_idx = np.argmax(mask_areas)
        best_mask_bool = masks_np[largest_mask_idx].astype(bool)
        combined_mask = combined_mask | best_mask_bool
        individual_masks.append(best_mask_bool)

    display_img = rgb.copy()
    for pt in input_points:
        cv2.circle(display_img, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

    final_mask_visual = (combined_mask * 255).astype(np.uint8)

    cv2.imshow("Input Image with Picked Points", display_img)
    cv2.imshow("SAM2 Combined Predicted Mask", final_mask_visual)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

    return combined_mask, individual_masks

def project_mask_to_points(mask, cam_K, depth):
    mask_indices = np.argwhere(mask)
    v = mask_indices[:, 0]
    u = mask_indices[:, 1]
    z = depth[v, u]  
    valid = z > 0
    u, v, z = u[valid], v[valid], z[valid]
    x = (u - cam_K[0, 2]) * z / cam_K[0, 0]
    y = (v - cam_K[1, 2]) * z / cam_K[1, 1]
    points_3d = np.column_stack((x, y, z))
    return points_3d

