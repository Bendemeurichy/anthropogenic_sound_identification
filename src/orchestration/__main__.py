"""
CLI entry point for the orchestration module.

Usage:
    python -m orchestration <command> [options]

Commands:
    create      Create a new project
    add-class   Add a sound class to a project
    remove-class  Remove a sound class from a project
    list        List available training presets
    info        Show project information
    train       Train the separation model
    separate    Separate audio into component sources

Run with --help on any command for detailed options.
"""

import sys
from pathlib import Path

# Ensure src/ is on sys.path so imports work regardless of working directory.
_SCRIPT_DIR = Path(__file__).parent.resolve()
_SRC_DIR = _SCRIPT_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def cmd_create(args):
    """Create a new project."""
    import argparse
    parser = argparse.ArgumentParser(prog="orchestration create", description="Create a new project")
    parser.add_argument("name", help="Project name")
    parser.add_argument("base_dir", nargs="?", default="./projects", help="Base directory for projects")
    parser.add_argument("--description", default="", help="Project description")
    opts = parser.parse_args(args)

    from orchestration import Project
    project = Project.create(opts.name, base_dir=opts.base_dir, description=opts.description)
    print(f"Created project: {project}")
    print(f"Directory: {project.project_dir}")


def cmd_add_class(args):
    """Add a sound class to a project."""
    import argparse
    parser = argparse.ArgumentParser(prog="orchestration add-class", description="Add a sound class")
    parser.add_argument("project", help="Path to project directory or project.yaml")
    parser.add_argument("name", help="Class name (e.g. 'bird')")
    parser.add_argument("labels", help="Comma-separated label strings (e.g. 'bird,birdsong,bird_call')")
    parser.add_argument("--description", default="", help="Class description")
    opts = parser.parse_args(args)

    from orchestration import Project, SoundClass
    project = Project.load(opts.project)
    labels = [label.strip() for label in opts.labels.split(",") if label.strip()]
    cls = SoundClass(name=opts.name, labels=labels, description=opts.description)
    project.add_class(cls)
    print(f"Added class '{opts.name}' with labels: {labels}")
    print(f"Project now has classes: {project.list_classes()}")


def cmd_remove_class(args):
    """Remove a sound class from a project."""
    import argparse
    parser = argparse.ArgumentParser(prog="orchestration remove-class", description="Remove a sound class")
    parser.add_argument("project", help="Path to project directory or project.yaml")
    parser.add_argument("name", help="Class name to remove")
    opts = parser.parse_args(args)

    from orchestration import Project
    project = Project.load(opts.project)
    if project.remove_class(opts.name):
        print(f"Removed class '{opts.name}'")
    else:
        print(f"Class '{opts.name}' not found.")
        sys.exit(1)
    print(f"Remaining classes: {project.list_classes()}")


def cmd_list(args):
    """List available training presets."""
    from orchestration import list_presets
    presets = list_presets()
    for name, desc in presets.items():
        print(f"  {name:20s} {desc}")


def cmd_info(args):
    """Show project information."""
    import argparse
    parser = argparse.ArgumentParser(prog="orchestration info", description="Show project info")
    parser.add_argument("project", help="Path to project directory or project.yaml")
    opts = parser.parse_args(args)

    from orchestration import Project
    project = Project.load(opts.project)
    print(project)
    print(f"Directory: {project.project_dir}")
    print(f"Preset: {project.settings.preset}")
    print(f"Data path: {project.settings.data_path}")
    print(f"Sample rate: {project.settings.sample_rate} Hz")
    print(f"Segment length: {project.settings.segment_length}s")
    print(f"SNR range: {project.settings.snr_range}")
    print(f"Freeze backbone: {project.settings.freeze_backbone}")
    print(f"Trained: {project.is_trained}")
    if project.checkpoint_path:
        print(f"Checkpoint: {project.checkpoint_path}")


def cmd_train(args):
    """Train the separation model."""
    import argparse
    parser = argparse.ArgumentParser(prog="orchestration train", description="Train the model")
    parser.add_argument("project", help="Path to project directory or project.yaml")
    parser.add_argument("--data-path", help="Path to metadata CSV file")
    parser.add_argument("--preset", help="Training preset name (quick, balanced, thorough, small_dataset, large_dataset)")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--device", default=None, help="Device (cuda, cuda:0, cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Generate config only, don't train")
    parser.add_argument("--sample-rate", type=int, help="Audio sample rate")
    parser.add_argument("--webdataset", action="store_true", help="Use WebDataset mode")
    parser.add_argument("--webdataset-path", help="Path to WebDataset tar shards")
    opts = parser.parse_args(args)

    from orchestration import Project
    project = Project.load(opts.project)

    if opts.data_path:
        project.settings.data_path = opts.data_path
    if opts.sample_rate:
        project.settings.sample_rate = opts.sample_rate
    if opts.webdataset:
        project.settings.use_webdataset = True
        if opts.webdataset_path:
            project.settings.webdataset_path = opts.webdataset_path

    project.save()

    print(f"Training project: {project.settings.name}")
    print(f"  Classes: {project.list_classes()}")
    print(f"  Preset: {opts.preset or project.settings.preset}")

    errors = project.validate()
    if errors:
        print("\nValidation errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    try:
        checkpoint_path = project.train(
            preset=opts.preset,
            epochs=opts.epochs,
            device=opts.device,
            dry_run=opts.dry_run,
        )
        if checkpoint_path:
            print(f"\nTraining complete. Checkpoint: {checkpoint_path}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_separate(args):
    """Separate audio into component sources."""
    import argparse
    parser = argparse.ArgumentParser(prog="orchestration separate", description="Separate audio")
    parser.add_argument("project", help="Path to project directory or project.yaml")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--classes", help="Comma-separated prompt names (default: all)")
    parser.add_argument("--no-recycling", action="store_true", help="Disable mask recycling")
    parser.add_argument("--device", default=None, help="Device (cuda, cuda:0, cpu)")
    opts = parser.parse_args(args)

    from orchestration import Project
    project = Project.load(opts.project)

    if not project.is_trained:
        print("Error: Project has no trained model. Run 'train' first.")
        sys.exit(1)

    classes = None
    if opts.classes:
        classes = [c.strip() for c in opts.classes.split(",") if c.strip()]

    print(f"Separating: {opts.audio}")
    print(f"  Using checkpoint: {project.checkpoint_path}")

    project.settings.enable_mask_recycling = not opts.no_recycling

    sources = project.separate(
        audio_path=opts.audio,
        output_dir=opts.output,
        classes=classes,
    )

    print(f"\nOutput sources: {list(sources.keys())}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print()
        print("Available commands:")
        for cmd in sorted(COMMANDS):
            print(f"  {cmd}")
        print()
        print("Example workflow:")
        print("  python -m orchestration create my_project ./projects")
        print("  python -m orchestration add-class my_project bird \"bird,birdsong\"")
        print("  python -m orchestration train my_project --data-path data.csv")
        print("  python -m orchestration separate my_project recording.wav --output out/")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd in COMMANDS:
        COMMANDS[cmd](sys.argv[2:])
    else:
        print(f"Unknown command: {cmd}")
        print(f"Available commands: {', '.join(sorted(COMMANDS))}")
        sys.exit(1)


COMMANDS = {
    "create": cmd_create,
    "add-class": cmd_add_class,
    "remove-class": cmd_remove_class,
    "list": cmd_list,
    "info": cmd_info,
    "train": cmd_train,
    "separate": cmd_separate,
}

if __name__ == "__main__":
    main()
