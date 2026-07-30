"""
Mask2Former panoptic model load + inference, adapted from the user's
playground notebook. Used by 02_svi_segmentation.

Functions to implement:
- load_model(checkpoint) -> (processor, model)
- infer_panoptic(image, processor, model, stuff_class_ids) -> (segmentation_map, segments_info)
- save_result(point_id, segmentation_map, segments_info, out_dir)  # per-image checkpoint
"""
