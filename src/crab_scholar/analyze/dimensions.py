"""Pluggable analysis dimension loader.

Loads analysis dimensions from YAML files. Ships with 5 built-in
dimensions focused on LLM evaluation. Users can add/override dimensions
by placing YAML files in a custom prompts directory.
"""

import logging
from importlib import resources
from pathlib import Path

import yaml

from crab_scholar.models import AnalysisDimension

logger = logging.getLogger(__name__)

# Built-in prompt YAMLs are in crab_scholar/prompts/
_BUNDLED_PROMPTS_PACKAGE = "crab_scholar.prompts"

# Default dimension names (shipped with the package)
DEFAULT_DIMENSIONS = [
    "analysis",
    "dataset_crafting",
    "evaluation_method",
    "metrics",
    "statistical_tests",
]


def load_dimensions(
    prompts_dir: Path | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[AnalysisDimension]:
    """Load analysis dimensions from bundled defaults + optional user overrides.

    Args:
        prompts_dir: Optional directory with custom YAML prompt files.
            Files with the same name as a built-in dimension override it.
            New files add new dimensions.
        include: If set, only load dimensions with these names.
        exclude: If set, skip dimensions with these names.

    Returns:
        List of AnalysisDimension objects, sorted by name.
    """
    dimensions: dict[str, AnalysisDimension] = {}

    # Step 1: Load bundled defaults
    for name in DEFAULT_DIMENSIONS:
        try:
            dim = _load_bundled(name)
            if dim:
                dimensions[dim.name] = dim
        except Exception as e:
            logger.warning(f"Failed to load bundled dimension '{name}': {e}")

    # Step 2: Overlay user-provided dimensions
    if prompts_dir and prompts_dir.is_dir():
        for yaml_file in sorted(prompts_dir.glob("*.yaml")):
            try:
                dim = _load_from_path(yaml_file)
                if dim:
                    if dim.name in dimensions:
                        logger.info(f"User override: replacing dimension '{dim.name}'")
                    else:
                        logger.info(f"User custom dimension: '{dim.name}'")
                    dimensions[dim.name] = dim
            except Exception as e:
                logger.warning(f"Failed to load dimension from {yaml_file}: {e}")

        # Also support .yml extension
        for yaml_file in sorted(prompts_dir.glob("*.yml")):
            try:
                dim = _load_from_path(yaml_file)
                if dim:
                    if dim.name in dimensions:
                        logger.info(f"User override: replacing dimension '{dim.name}'")
                    else:
                        logger.info(f"User custom dimension: '{dim.name}'")
                    dimensions[dim.name] = dim
            except Exception as e:
                logger.warning(f"Failed to load dimension from {yaml_file}: {e}")

    # Step 3: Apply include/exclude filters
    if include:
        include_set = set(include)
        dimensions = {k: v for k, v in dimensions.items() if k in include_set}
    if exclude:
        exclude_set = set(exclude)
        dimensions = {k: v for k, v in dimensions.items() if k not in exclude_set}

    result = sorted(dimensions.values(), key=lambda d: d.name)
    logger.info(f"Loaded {len(result)} analysis dimensions: {[d.name for d in result]}")
    return result


def list_available_dimensions(prompts_dir: Path | None = None) -> list[dict]:
    """List all available dimensions with metadata.

    Returns:
        List of dicts with name, display_name, description, source (bundled/custom).
    """
    dims = load_dimensions(prompts_dir)
    result = []
    for dim in dims:
        source = "bundled" if dim.name in DEFAULT_DIMENSIONS else "custom"
        result.append({
            "name": dim.name,
            "display_name": dim.display_name,
            "description": dim.description,
            "source": source,
        })
    return result


def _load_bundled(name: str) -> AnalysisDimension | None:
    """Load a bundled dimension YAML from the package resources."""
    try:
        ref = resources.files(_BUNDLED_PROMPTS_PACKAGE).joinpath(f"{name}.yaml")
        content = ref.read_text(encoding="utf-8")
        return _parse_dimension_yaml(content, source=f"bundled:{name}")
    except Exception as e:
        logger.debug(f"Could not load bundled dimension '{name}': {e}")
        return None


def _load_from_path(path: Path) -> AnalysisDimension | None:
    """Load a dimension from a YAML file path."""
    content = path.read_text(encoding="utf-8")
    return _parse_dimension_yaml(content, source=str(path))


def _parse_dimension_yaml(content: str, source: str = "") -> AnalysisDimension:
    """Parse a YAML string into an AnalysisDimension."""
    data = yaml.safe_load(content)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML dict, got {type(data).__name__}")

    name = data.get("name", "")
    if not name:
        raise ValueError(f"Dimension YAML missing 'name' field ({source})")

    return AnalysisDimension(
        name=name,
        display_name=data.get("display_name", name.replace("_", " ").title()),
        description=data.get("description", ""),
        system_message=data.get("system_message", ""),
        extraction_prompt=data.get("extraction_prompt", ""),
    )
