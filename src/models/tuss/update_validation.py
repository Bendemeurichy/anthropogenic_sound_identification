import re

with open('src/models/tuss/train.py', 'r') as f:
    content = f.read()

old_func = """def validate_prompts_against_checkpoint(
    config_prompts: list[str],
    bg_prompt: str,
    checkpoint_path: str | None,
) -> tuple[list[str], list[str]]:
    \"\"\"Validate config prompts against an existing checkpoint.
    
    Args:
        config_prompts: COI prompts from config
        bg_prompt: Background prompt from config
        checkpoint_path: Path to checkpoint to resume from
        
    Returns:
        Tuple of:
            - existing_prompts: List of prompts that already exist in checkpoint
            - new_prompts: List of prompts that are new (will be injected)
    \"\"\"
    if not checkpoint_path:
        return [], config_prompts + [bg_prompt]
    
    checkpoint_prompts = get_prompts_from_checkpoint(checkpoint_path)
    
    if not checkpoint_prompts:
        # Checkpoint doesn't have prompts or couldn't be read
        return [], config_prompts + [bg_prompt]
    
    all_config_prompts = set(config_prompts + [bg_prompt])
    existing = sorted(all_config_prompts & checkpoint_prompts)
    new = sorted(all_config_prompts - checkpoint_prompts)
    frozen = sorted(checkpoint_prompts - all_config_prompts)"""

new_func = """def validate_prompts_against_checkpoint(
    config_prompts: list[str],
    bg_prompt: str,
    checkpoint_path: str | None,
) -> tuple[list[str], list[str]]:
    \"\"\"Validate config prompts against an existing checkpoint.
    
    Args:
        config_prompts: COI prompts from config
        bg_prompt: Background prompt from config
        checkpoint_path: Path to checkpoint to resume from
        
    Returns:
        Tuple of:
            - existing_prompts: List of prompts that already exist in checkpoint
            - new_prompts: List of prompts that are new (will be injected)
    \"\"\"
    if not checkpoint_path:
        return [], config_prompts + [bg_prompt]
    
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        old_coi_prompts = ckpt.get("coi_prompts", [])
        old_bg_prompt = ckpt.get("bg_prompt", bg_prompt)
    except Exception as e:
        print(f"⚠ Warning: Could not read prompts from checkpoint: {e}")
        return [], config_prompts + [bg_prompt]

    if old_coi_prompts:
        # Strict validation: Check prefix matches
        if len(config_prompts) < len(old_coi_prompts):
            raise ValueError(f"Config has fewer COI prompts ({len(config_prompts)}) than checkpoint ({len(old_coi_prompts)}).")
        for i, p in enumerate(old_coi_prompts):
            if config_prompts[i] != p:
                raise ValueError(f"Prompt order mismatch! Checkpoint prompt {i} is '{p}', but config prompt {i} is '{config_prompts[i]}'. You must append new prompts at the end.")
        
        if bg_prompt != old_bg_prompt:
            raise ValueError(f"Background prompt mismatch! Checkpoint uses '{old_bg_prompt}', config uses '{bg_prompt}'.")
    
    checkpoint_prompts = set(old_coi_prompts + [old_bg_prompt]) if old_coi_prompts else get_prompts_from_checkpoint(checkpoint_path)
    
    if not checkpoint_prompts:
        return [], config_prompts + [bg_prompt]
    
    all_config_prompts = set(config_prompts + [bg_prompt])
    existing = sorted(all_config_prompts & checkpoint_prompts)
    new = sorted(all_config_prompts - checkpoint_prompts)
    frozen = sorted(checkpoint_prompts - all_config_prompts)"""

if old_func in content:
    content = content.replace(old_func, new_func)
    with open('src/models/tuss/train.py', 'w') as f:
        f.write(content)
    print("Replaced!")
else:
    print("Not found.")
