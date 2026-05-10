#!/usr/bin/env python3
"""
Example: Using the Orchestration Layer for Sound Separation

This script demonstrates the simplified API for managing sound separation
projects, designed for biologists and end-users.

The orchestration layer provides:
1. Simple project management (create, load, save)
2. Easy sound class definition (just name + labels)
3. Training presets (quick, balanced, thorough)
4. Integrated inference with mask recycling

Run this script to see example usage patterns.
"""

import sys
from pathlib import Path

# Add src to path for imports
_SCRIPT_DIR = Path(__file__).parent.resolve()
_SRC_DIR = _SCRIPT_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def example_create_project():
    """Example: Create a new project and add sound classes."""
    from orchestration import Project, SoundClass
    
    print("=" * 60)
    print("EXAMPLE: Creating a New Project")
    print("=" * 60)
    
    # Create a new project
    project = Project.create(
        name="wildlife_monitoring",
        base_dir="./example_projects",
        description="Separate bird calls from anthropogenic noise"
    )
    print(f"Created project: {project}")
    print(f"Project directory: {project.project_dir}")
    
    # Add sound classes - just specify name and labels!
    project.add_class(SoundClass(
        name="bird",
        labels=["bird", "Bird", "birdsong", "bird_call", "bird_chirp"],
        description="Bird vocalizations including songs and calls"
    ))
    
    project.add_class(SoundClass(
        name="airplane",
        labels=["airplane", "aircraft", "jet", "propeller"],
        description="Aircraft engine noise"
    ))
    
    project.add_class(SoundClass(
        name="vehicle",
        labels=["car", "truck", "motorcycle", "engine"],
        description="Ground vehicle noise"
    ))
    
    print(f"\nSound classes: {project.list_classes()}")
    print(f"Project saved to: {project.project_dir / 'project.yaml'}")
    
    return project


def example_configure_project(project):
    """Example: Configure project settings."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Configuring Project Settings")
    print("=" * 60)
    
    # Configure data settings
    project.settings.data_path = "data/metadata.csv"  # Your data CSV
    project.settings.sample_rate = 48000  # Must match your audio files
    project.settings.segment_length = 4.0  # 4 second segments
    
    # Choose a training preset
    # Options: "quick", "balanced", "thorough", "small_dataset", "large_dataset"
    project.settings.preset = "balanced"
    
    # Optional: Override specific preset values
    project.settings.epochs = 150  # Override balanced preset's 100 epochs
    
    # Configure inference settings
    project.settings.enable_mask_recycling = True  # Speed up inference
    project.settings.similarity_threshold = 0.85  # Cache reuse threshold
    
    project.save()
    
    print(f"Data path: {project.settings.data_path}")
    print(f"Sample rate: {project.settings.sample_rate} Hz")
    print(f"Training preset: {project.settings.preset}")
    print(f"Epochs: {project.settings.epochs}")
    
    return project


def example_view_presets():
    """Example: View available training presets."""
    from orchestration import get_preset
    from orchestration.presets import list_presets, PRESETS
    
    print("\n" + "=" * 60)
    print("EXAMPLE: Available Training Presets")
    print("=" * 60)
    
    for name, description in list_presets().items():
        preset = get_preset(name)
        print(f"\n{name.upper()}")
        print(f"  {description}")
        print(f"  - Epochs: {preset.epochs}")
        print(f"  - Patience: {preset.patience}")
        print(f"  - Freeze backbone: {preset.freeze_backbone}")
        print(f"  - Learning rate: {preset.lr}")


def example_generate_config(project):
    """Example: Generate TUSS training config from project."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Generating Training Config")
    print("=" * 60)
    
    # Generate the full TUSS training config
    config = project.generate_training_config()
    
    print("Generated config structure:")
    print(f"  data.target_classes: {config['data']['target_classes']}")
    print(f"  model.coi_prompts: {config['model']['coi_prompts']}")
    print(f"  model.freeze_backbone: {config['model']['freeze_backbone']}")
    print(f"  training.num_epochs: {config['training']['num_epochs']}")
    print(f"  training.lr: {config['training']['lr']}")
    
    # Write config file (this happens automatically during train())
    config_path = project.write_training_config()
    print(f"\nConfig written to: {config_path}")
    
    return config_path


def example_validate_project(project):
    """Example: Validate project configuration."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Validating Project")
    print("=" * 60)
    
    errors = project.validate()
    
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Project configuration is valid!")
    
    return len(errors) == 0


def example_load_project():
    """Example: Load an existing project."""
    from orchestration import Project
    
    print("\n" + "=" * 60)
    print("EXAMPLE: Loading Existing Project")
    print("=" * 60)
    
    # Load from project directory
    project = Project.load("./example_projects/wildlife_monitoring")
    
    print(f"Loaded: {project}")
    print(f"Classes: {project.list_classes()}")
    print(f"Is trained: {project.is_trained}")
    
    return project


def example_training_workflow():
    """Example: Complete training workflow (dry run)."""
    from orchestration import Project, SoundClass
    
    print("\n" + "=" * 60)
    print("EXAMPLE: Training Workflow (Dry Run)")
    print("=" * 60)
    
    # Quick setup
    project = Project.create("quick_test", base_dir="./example_projects")
    project.add_class(SoundClass(name="bird", labels=["bird", "birdsong"]))
    
    # In a real scenario, you'd point to your actual metadata CSV
    # project.settings.data_path = "path/to/your/metadata.csv"
    
    # For this example, we'll just generate the config without validation
    print("\nGenerating training config (skipping validation for demo)...")
    config_path = project.write_training_config()
    print(f"Config written to: {config_path}")
    
    print(f"\nTo actually train:")
    print("  1. Set project.settings.data_path to your metadata CSV")
    print("  2. Call project.train(preset='quick')")
    print("\nThis would:")
    print("  1. Load metadata from data_path")
    print("  2. Create TUSS model with injected prompts")
    print("  3. Train for the specified number of epochs")
    print("  4. Save checkpoints to project_dir/checkpoints/")
    
    return project


def example_inference_workflow():
    """Example: Inference workflow (requires trained model)."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Inference Workflow")
    print("=" * 60)
    
    print("""
After training, run inference like this:

    # Load trained project
    project = Project.load("./projects/wildlife_monitoring")
    
    # Separate a single file
    sources = project.separate("recording.wav")
    
    # Access separated audio by class name
    bird_audio = sources["bird"]
    airplane_audio = sources["airplane"]
    background = sources["background"]
    
    # Or save to files
    sources = project.separate(
        "recording.wav",
        output_dir="./separated/"
    )
    # Creates: separated/recording_bird.wav, recording_airplane.wav, etc.
    
    # For batch processing, get the pipeline directly
    pipeline = project.get_pipeline()
    
    for audio_file in audio_files:
        waveform, sr = torchaudio.load(audio_file)
        sources = pipeline.separate_waveform(waveform)
        # ... process sources ...
    
    # Check mask recycling efficiency
    stats = pipeline.get_stats()
    print(f"Cache hit rate: {stats['hit_rate']:.1%}")
    """)


def example_project_yaml():
    """Show what a project.yaml file looks like."""
    print("\n" + "=" * 60)
    print("EXAMPLE: project.yaml File Format")
    print("=" * 60)
    
    example_yaml = '''
# project.yaml - Simplified project configuration
# This is much simpler than training_config.yaml!

settings:
  name: wildlife_monitoring
  description: Separate bird calls from anthropogenic noise
  
  # Data settings
  data_path: data/metadata.csv
  sample_rate: 48000
  segment_length: 4.0
  
  # Training preset (quick, balanced, thorough, small_dataset, large_dataset)
  preset: balanced
  
  # Optional overrides
  epochs: 150        # Override preset default
  patience: null     # Use preset default
  freeze_backbone: null  # Use preset default
  
  # Inference settings
  enable_mask_recycling: true
  similarity_threshold: 0.85

# Sound classes to separate
classes:
  - name: bird
    labels: [bird, Bird, birdsong, bird_call]
    description: Bird vocalizations
    
  - name: airplane
    labels: [airplane, aircraft, jet]
    description: Aircraft noise
    
  - name: vehicle
    labels: [car, truck, motorcycle]
    description: Ground vehicle noise
'''
    print(example_yaml)


def main():
    """Run all examples."""
    print("\n" + "#" * 60)
    print("# ORCHESTRATION LAYER EXAMPLES")
    print("# Simplified Sound Separation Pipeline Management")
    print("#" * 60)
    
    # Create and configure a project
    project = example_create_project()
    project = example_configure_project(project)
    
    # View available presets
    example_view_presets()
    
    # Generate training config
    example_generate_config(project)
    
    # Validate
    example_validate_project(project)
    
    # Show how to load existing project
    example_load_project()
    
    # Show training workflow
    example_training_workflow()
    
    # Show inference workflow
    example_inference_workflow()
    
    # Show YAML format
    example_project_yaml()
    
    print("\n" + "=" * 60)
    print("DONE - See ./example_projects/ for generated files")
    print("=" * 60)


if __name__ == "__main__":
    main()
