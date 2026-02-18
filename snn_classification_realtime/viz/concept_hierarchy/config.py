"""Configuration for concept hierarchy visualization."""

from dataclasses import dataclass


@dataclass
class ConceptHierarchyConfig:
    """Configuration for concept hierarchy plots."""

    json_file: str
    output_dir: str = "concept_hierarchy_output"
