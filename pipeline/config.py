from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Literal

# --- Step 1: Network Construction ---
class LayerConfig(BaseModel):
    layer_type: Literal['conv']
    filters: int = Field(default=0, description="Use 0 for no multiple filters (or single filter)")
    kernel_size: int
    stride: int
    connectivity: float = 1.0
    shortcut_connections: bool = False
    inhibitory_signals: bool = False

class NetworkConfig(BaseModel):
    layers: List[LayerConfig]
    inhibitory_signals: bool = False # Global override or initial setting

# --- Step 2: Activity Recording ---
class SimulationConfig(BaseModel):
    ticks_per_image: int = 20
    images_per_label: int = 500
    tick_time: float = 0.0
    fresh_run_per_label: bool = True
    fresh_run_per_image: bool = True
    export_state: bool = False
    dataset_name_base: str = "mnist_experiment"

# --- Step 3: Data Preparation ---
class PreparationConfig(BaseModel):
    feature_types: List[str] = ["avg_S", "firings"]
    train_split: float = 0.8
    scaler: str = "none"
    scale_eps: float = 1e-8
    max_ticks: Optional[int] = None
    max_samples: Optional[int] = None

# --- Step 4: Training ---
class TrainingConfig(BaseModel):
    epochs: int = 20
    test_every: int = 1
    learning_rate: float = 0.0005
    batch_size: int = 32
    device: str = "cpu"

# --- Step 5: Evaluation ---
class EvaluationConfig(BaseModel):
    window_size: int = 80
    eval_samples: int = 1000
    think_longer: bool = True
    max_thinking_multiplier: int = 4
    device: str = "cpu"
    dataset_name: str = "mnist"

# --- Visualization ---
class VisualizationParam(BaseModel):
    enabled: bool = False
    params: Dict[str, Any] = {}

class VisualizationsConfig(BaseModel):
    neuron_visualizer: VisualizationParam = VisualizationParam()
    activity_3d: VisualizationParam = VisualizationParam()
    activity_dataset: VisualizationParam = VisualizationParam()
    network_activity: VisualizationParam = VisualizationParam()
    concept_hierarchy: VisualizationParam = VisualizationParam()
    synaptic_analysis: VisualizationParam = VisualizationParam()
    cluster_activity: VisualizationParam = VisualizationParam()
    cluster_neurons: VisualizationParam = VisualizationParam()

# --- Main Job Config ---
class JobConfig(BaseModel):
    name: str = "experiment"
    network: NetworkConfig
    simulation: SimulationConfig
    preparation: PreparationConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    visualizations: VisualizationsConfig = VisualizationsConfig()
