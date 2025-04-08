import torch
import math

"""
get_ground_truth_se2 fn returns SE(2) parameters for a given text prompt. 
Our goal in this parameters to extract geometrical/rotational behaviours between poses. 
"""
def get_ground_truth_se2(text_prompt):
    """
    Maps text prompts like "left eye, input: OD, target: Nasal" to SE(2) parameters (θ, dx, dy).
    SE(2) parameters are predefined based on anatomical relationships in fundus imaging.
    
    Returns:
        se2_params (torch.Tensor): [θ (radians), dx, dy]
    """
    # Parse text prompt
    input_fov, target_fov, eye = parse_fov_and_eye(text_prompt)
    
    # Base SE(2) parameters for left eye (θ in radians, dx normalized to [-1,1])
    se2_lookup = {
        # OD ↔ Nasal/Temporal
        ("OD", "Nasal"):    (math.pi/6,  0.1, 0.0),  # 30° rotation, rightward translation
        ("OD", "Temporal"): (-math.pi/6, -0.1, 0.0),  # -30° rotation, leftward translation
        ("Nasal", "OD"):    (-math.pi/6, -0.1, 0.0),
        ("Temporal", "OD"): (math.pi/6,  0.1, 0.0),
        
        # Nasal ↔ Temporal
        ("Nasal", "Temporal"): (math.pi, 0.2, 0.0),   # 180° rotation, large leftward translation
        ("Temporal", "Nasal"): (math.pi, 0.2, 0.0),   # Same as above (mirror)
        
        # Same-view (no transformation)
        ("OD", "OD"):        (0.0, 0.0, 0.0),
        ("Nasal", "Nasal"):  (0.0, 0.0, 0.0),
        ("Temporal", "Temporal"): (0.0, 0.0, 0.0)
    }
    
    # Get base parameters
    key = (input_fov, target_fov)
    if key not in se2_lookup:
        raise ValueError(f"Unsupported FOV pair: {key}")
    θ, dx, dy = se2_lookup[key]
    
    # Adjust for right eye: flip rotation and translation direction
    if eye == "right":
        θ *= -1
        dx *= -1
    
    return torch.tensor([θ, dx, dy], dtype=torch.float32)
