import os
import sys
import argparse
import yaml
import logging
import datetime
import subprocess
import json
from typing import Optional

# Ensure local imports resolve
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import JobConfig
from pipeline.step1_build_network import build_network
from pipeline.step2_record_activity import run_recording
from pipeline.step3_prepare_data import process_data
from pipeline.step4_train_model import train_model
from pipeline.step5_evaluate import evaluate_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PipelineRunner")

def setup_job_directory(job_name: str, explicit_dir: Optional[str] = None) -> str:
    """Create a unique directory for the job."""
    if explicit_dir:
        job_dir = explicit_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        job_dir_name = f"{job_name}_{timestamp}"
        job_dir = os.path.join("experiments", job_dir_name)

    os.makedirs(job_dir, exist_ok=True)
    return job_dir

def save_config(config_path: str, output_dir: str):
    """Save a copy of the configuration file to the job directory."""
    with open(config_path, 'r') as f:
        content = f.read()
    with open(os.path.join(output_dir, "config.yaml"), 'w') as f:
        f.write(content)

def run_step_visualization(config: JobConfig, step_name: str, input_path: str, output_dir: str):
    """Run visualization scripts relevant to the step."""
    viz_config = config.visualizations

    if step_name == "step2_activity":
        # 1. Visualize Network Activity
        if viz_config.network_activity.enabled:
            logger.info("Running network activity visualization...")
            cmd = [
                sys.executable,
                "pipeline/visualization/visualize_network_activity.py",
                "--input-file", input_path,
                "--output-dir", os.path.join(output_dir, "viz_network_activity"),
                "--num-classes", "10"
            ]
            for k, v in viz_config.network_activity.params.items():
                arg_name = f"--{k.replace('_', '-')}"
                if isinstance(v, list):
                    cmd.append(arg_name)
                    cmd.extend([str(item) for item in v])
                else:
                    cmd.extend([arg_name, str(v)])
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Visualization failed: {e}")

        # 2. Activity 3D
        if viz_config.activity_3d.enabled:
             logger.info("Running 3D activity visualization...")
             cmd = [
                sys.executable,
                "pipeline/visualization/visualize_activity_3d.py",
                "--input-file", os.path.join(input_path, "activity_dataset.h5"),
                "--output-dir", os.path.join(output_dir, "viz_activity_3d")
             ]
             try:
                subprocess.run(cmd, check=True)
             except subprocess.CalledProcessError as e:
                logger.error(f"3D Visualization failed: {e}")

        # 3. Cluster Neurons
        if viz_config.cluster_neurons.enabled:
            logger.info("Running neuron clustering...")
            cmd = [
                sys.executable,
                "pipeline/visualization/cluster_neurons.py",
                "--input-file", input_path,
                "--output-dir", os.path.join(output_dir, "viz_neuron_clustering")
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Neuron clustering failed: {e}")

        # 4. Cluster Activity
        if viz_config.cluster_activity.enabled:
            logger.info("Running activity clustering...")
            cmd = [
                sys.executable,
                "pipeline/visualization/cluster_activity_data.py",
                "--input-file", input_path,
                "--output-dir", os.path.join(output_dir, "viz_activity_clustering")
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Activity clustering failed: {e}")

        # 5. Activity Dataset Plots
        if viz_config.activity_dataset.enabled:
            logger.info("Running activity dataset plots...")
            cmd = [
                sys.executable,
                "pipeline/visualization/visualize_activity_dataset.py",
                input_path,
                "--out-dir", os.path.join(output_dir, "viz_activity_dataset")
            ]
            for k, v in viz_config.activity_dataset.params.items():
                arg_name = f"--{k.replace('_', '-')}"
                cmd.extend([arg_name, str(v)])
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Activity dataset plots failed: {e}")

        # 6. Synaptic Analysis (Needs network_state)
        if viz_config.synaptic_analysis.enabled:
            network_state_dir = os.path.join(input_path, "network_state")
            if os.path.exists(network_state_dir):
                logger.info("Running synaptic connections analysis...")
                cmd = [
                    sys.executable,
                    "pipeline/visualization/synaptic_connections_analysis.py",
                    network_state_dir,
                    "--output", os.path.join(output_dir, "viz_synaptic_analysis")
                ]
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Synaptic analysis failed: {e}")
            else:
                logger.warning("Skipping synaptic analysis: 'network_state' not found (enable export_state in simulation config).")

    if step_name == "step5_evaluation":
        # 7. Concept Hierarchy
        if viz_config.concept_hierarchy.enabled:
            # Look for evaluation_summary.json in input_path (which is step5_dir)
            json_file = os.path.join(input_path, "evaluation_summary.json")
            if os.path.exists(json_file):
                logger.info("Running concept hierarchy visualization...")
                cmd = [
                    sys.executable,
                    "pipeline/visualization/plot_concept_hierarchy.py",
                    "--json-file", json_file,
                    "--output-dir", os.path.join(output_dir, "viz_concept_hierarchy")
                ]
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Concept hierarchy visualization failed: {e}")

def run_pipeline(config_path: str, job_dir_arg: Optional[str] = None):
    logger.info(f"Loading configuration from {config_path}")

    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
        try:
            job_config = JobConfig(**cfg_dict)
        except Exception as e:
            logger.error(f"Invalid configuration: {e}")
            sys.exit(1)

    # 1. Setup Job Directory
    job_dir = setup_job_directory(job_config.name, job_dir_arg)
    logger.info(f"Job directory created: {job_dir}")

    # Save config
    save_config(config_path, job_dir)

    # Add file handler
    file_handler = logging.FileHandler(os.path.join(job_dir, "pipeline.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    try:
        # --- Step 1: Build Network ---
        logger.info(">>> Starting Step 1: Build Network")
        step1_dir = os.path.join(job_dir, "step1_network")
        build_network(job_config.network, step1_dir)
        network_json_path = os.path.join(step1_dir, "network.json")
        logger.info("<<< Step 1 Complete")

        # --- Step 2: Record Activity ---
        logger.info(">>> Starting Step 2: Record Activity")
        step2_dir = os.path.join(job_dir, "step2_activity")
        run_recording(job_config.simulation, network_json_path, step2_dir)
        logger.info("<<< Step 2 Complete")

        run_step_visualization(job_config, "step2_activity", step2_dir, job_dir)

        # --- Step 3: Prepare Data ---
        logger.info(">>> Starting Step 3: Prepare Data")
        step3_dir = os.path.join(job_dir, "step3_data")
        process_data(job_config.preparation, step2_dir, step3_dir)
        logger.info("<<< Step 3 Complete")

        # --- Step 4: Train Model ---
        logger.info(">>> Starting Step 4: Train Model")
        step4_dir = os.path.join(job_dir, "step4_model")
        train_model(job_config.training, step3_dir, step4_dir)
        logger.info("<<< Step 4 Complete")

        # --- Step 5: Evaluate Model ---
        logger.info(">>> Starting Step 5: Evaluate Model")
        step5_dir = os.path.join(job_dir, "step5_evaluation")
        evaluate_model(job_config.evaluation, step4_dir, network_json_path, step5_dir)
        logger.info("<<< Step 5 Complete")

        run_step_visualization(job_config, "step5_evaluation", step5_dir, job_dir)

        logger.info(f"Pipeline finished successfully. All results in {job_dir}")

    except Exception as e:
        logger.exception("Pipeline failed")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SNN experimentation pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to the job configuration YAML file.")
    parser.add_argument("--job-dir", type=str, help="Explicit output directory for the job.")
    args = parser.parse_args()

    run_pipeline(args.config, args.job_dir)
