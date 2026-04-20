# Neural Network CLI Documentation

A powerful command-line interface for creating, managing, and simulating neural networks with rich visualization and interactive features.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation & Setup](#installation--setup)
- [Getting Started](#getting-started)
- [Command Reference](#command-reference)
  - [Core Operations](#core-operations)
  - [Network Management](#network-management)
  - [Neuron Operations](#neuron-operations)
  - [Synapse Operations](#synapse-operations)
  - [Connection Management](#connection-management)
  - [Signal Operations](#signal-operations)
  - [Visualization & Analysis](#visualization--analysis)
  - [System Commands](#system-commands)
- [Context System](#context-system)
- [Advanced Features](#advanced-features)
- [Examples & Workflows](#examples--workflows)
- [Troubleshooting](#troubleshooting)

## Overview

The Neural Network CLI is an interactive command-line tool designed for building, simulating, and analyzing neural networks. It provides a comprehensive set of commands for managing neurons, synapses, connections, and running simulations with real-time visualization capabilities.

### Key Concepts

- **Neurons**: The basic computational units with membrane potential, firing rates, and synaptic inputs
- **Synapses**: Input connections to neurons that receive signals from other neurons or external sources
- **Terminals**: Output points on neurons that send signals to other neurons' synapses
- **Connections**: Links between neuron terminals and synapses
- **External Inputs**: Synapses that receive signals from outside the network
- **Traveling Events**: Signals moving through the network with realistic propagation delays

## Features

- 🎯 **Interactive CLI** with autocomplete and command history
- 🧠 **Comprehensive Neuron Management** - Create, configure, and monitor neurons
- 🔗 **Flexible Connection System** - Manual and automatic network topology creation
- 📊 **Real-time Visualization** - Network plots with traveling signals and activity states
- 🎛️ **Context System** - Focus operations on specific neurons/synapses
- ⚡ **Signal Simulation** - Send signals and observe network dynamics
- 📁 **Network Persistence** - Import/export network configurations
- 🕐 **Timing Control** - Manual stepping or autonomous time progression
- 📈 **Performance Monitoring** - Command timing and network statistics

## Installation & Setup

### Prerequisites

From the **repository root** (parent of `cli/`):

```bash
uv sync --extra cli --extra web
```

For the full visualization stack (UMAP, static Plotly export, debugger), use `uv sync --all-extras` instead.

**Included in those extras:** `numpy`, `matplotlib`, `networkx`, `scipy`, `loguru`, `prompt-toolkit`, `rich`, `websockets`, `flask`, `flask-cors`, `flask-socketio`, and related pins from `uv.lock`.

### Running the CLI

```bash
cd cli/
python neuron_cli.py
```

Or make it executable:

```bash
chmod +x neuron_cli.py
./neuron_cli.py
```

## Getting Started

### Basic Workflow

1. **Start the CLI** and see the welcome screen
2. **Create neurons** with `add_neuron`
3. **Connect neurons** with `add_connection` or `auto_connect`
4. **Send signals** with `signal`
5. **Run simulation** with `tick` or `start`
6. **Visualize results** with `plot`

### Quick Example

```
> add_neuron 4 2 3           # Create 3 neurons with 4 synapses, 2 terminals each
> auto_connect 1 1           # Auto-connect neurons, leaving 1 free synapse/terminal
> signal 12345 0 1.5         # Send signal strength 1.5 to neuron 12345, synapse 0
> nticks 10                  # Run 10 simulation steps
> plot                       # Visualize the network
```

## Command Reference

### Core Operations

#### `tick`
Execute a single simulation time step.

```
> tick
✓ Tick 1 completed
Total activity: 2.3
```

#### `nticks [N]`
Execute multiple simulation steps. Default is 10 if not specified.

```
> nticks 50                  # Run 50 time steps
> nticks                     # Prompted for number (default: 10)
✓ Completed 50/50 ticks
Neural activity in 23 ticks
```

#### `start`
Begin autonomous time progression at specified rate.

```
> start
Tick rate (tps) [1.0]: 2.5   # 2.5 ticks per second
✓ Time flow started at 2.5 tps
```

#### `stop`
Stop autonomous time progression.

```
> stop
✓ Time flow stopped
```

### Network Management

#### `import [filepath]`
Load a network from JSON configuration file.

```
> import networks/my_network.json
> import                     # Uses default: networks/network.json
✓ Network imported from networks/my_network.json
```

#### `export`
Save current network to JSON file.

```
> export
Export file path [network.json]: my_experiment.json
✓ Network exported to networks/my_experiment.json
```

### Neuron Operations

#### `add_neuron [synapses] [terminals] [count]`
Create new neurons with specified topology.

- `synapses`: Number of input synapses (default: 4)
- `terminals`: Number of output terminals (default: 1)  
- `count`: Number of neurons to create (default: 1)

```
> add_neuron 6 2 5           # 5 neurons, 6 synapses, 2 terminals each
> add_neuron 4               # 1 neuron, 4 synapses, 1 terminal
> add_neuron                 # Prompted for parameters
✓ Created 5/5 neurons successfully
Created neuron IDs: [12345, 67890, 13579, ...]
```

#### `get_neuron [id]`
Display detailed information about a specific neuron.

```
> get_neuron 12345           # Get info for neuron 12345
> get_neuron                 # Uses context neuron or prompts

╭─ Basic Information ─╮
│ Neuron ID        │ 12345     │
│ Membrane Potential│ 0.234567  │
│ Firing Rate      │ 0.000000  │
│ Output           │ 0.000000  │
│ Synapse Count    │ 4         │
│ Terminal Count   │ 2         │
╰──────────────────────────────╯

Show detailed neuron parameters? [y/N]: y
```

#### `del_neuron`
Delete a neuron from the network.

```
> del_neuron
Neuron ID: 12345
Delete neuron 12345? [y/N]: y
✓ Neuron 12345 deleted
```

### Synapse Operations

#### `add_synapse`
Add a synapse to an existing neuron.

```
> add_synapse
Neuron ID: 12345
Synapse ID: 3
Distance to hillock [5]: 7
✓ Synapse 3 added to neuron 12345
```

#### `get_synapse`
Display synapse information (uses context if available).

```
> get_synapse
Neuron ID: 12345            # Or uses context
Synapse ID: 2               # Or uses context

╭─ Synapse 2 Information ─╮
│ Distance to Hillock │ 5     │
│ Potential          │ 0.123 │
╰─────────────────────────────╯
```

#### `del_synapse`
Remove a synapse from a neuron.

```
> del_synapse
Neuron ID: 12345
Synapse ID: 2
Delete synapse 2 from neuron 12345? [y/N]: y
✓ Synapse 2 deleted from neuron 12345
```

### Connection Management

#### `add_connection [src_neuron] [src_terminal] [tgt_neuron] [tgt_synapse]`
Create a direct connection between two neurons.

```
> add_connection 100 0 200 1    # Connect N100:T0 → N200:S1
> add_connection                # Prompted for all parameters
✓ Connected N100:T0 → N200:S1
```

#### `auto_connect [min_synapses] [min_terminals]`
Automatically create connections while preserving minimum free synapses/terminals.

```
> auto_connect 1 1           # Leave 1 free synapse and 1 free terminal per neuron
> auto_connect 0 2           # Leave 0 free synapses, 2 free terminals
> auto_connect              # Prompted for parameters (default: 1, 1)

Auto-connecting neurons...
✓ Created 25 connections
Free synapses: 15, Free terminals: 12
Synaptic density: 67.3%, Graph density: 23.1%
```

#### `clear_connections`
Remove all connections, converting them to external inputs.

```
> clear_connections
Clear all 25 connections? [y/N]: y
✓ All connections cleared
```

#### `list_external`
Display all external input synapses.

```
> list_external

╭─ External Input Synapses ─╮
│ Neuron ID │ Synapse ID │ Distance │ Info Signal │ Mod Signals │ Synapse Potential │
├───────────┼────────────┼──────────┼─────────────┼─────────────┼───────────────────┤
│ 12345     │ 0          │ 5        │ 0.000       │ [0.0, 0.0]  │ 0.000000         │
│ 12345     │ 1          │ 3        │ 0.000       │ [0.0, 0.0]  │ 0.000000         │
╰─────────────────────────────────────────────────────────────────────────────────╯

Total external inputs: 15
Use 'signal' command to send signals to these synapses
```

#### `list_free_outputs`
Show available presynaptic terminals for connections.

```
> list_free_outputs

╭─ Free Presynaptic Terminals ─╮
│ Neuron ID │ Terminal ID │
├───────────┼─────────────┤
│ 12345     │ 1           │
│ 67890     │ 0           │
│ 67890     │ 1           │
╰─────────────────────────────╯

Total free terminals: 12
```

### Signal Operations

#### `signal [neuron] [synapse] [strength] [repeat]`
Send signal(s) to a specific synapse.

```
> signal 12345 0 1.5 1       # Single signal, strength 1.5
> signal 12345 0 1.0 100     # 100 signals of strength 1.0
> signal 1.5                 # Uses context neuron/synapse
> signal                     # Prompted for all parameters

✓ Signal sent to neuron 12345, synapse 0
```

#### `batch_signal`
Send multiple signals in one operation.

```
> batch_signal
Number of signals to send [3]: 3

Signal 1:
  Target neuron ID: 12345
  Target synapse ID: 0  
  Signal strength [1.5]: 2.0

Signal 2:
  Target neuron ID: 67890
  Target synapse ID: 1
  Signal strength [1.5]: 1.8

Signal 3:
  Target neuron ID: 13579
  Target synapse ID: 0
  Signal strength [1.5]: 1.2

Send these signals? [y/N]: y
✓ Successfully sent 3/3 signals
```

### Visualization & Analysis

#### `plot`
Generate comprehensive network visualization with activity states and traveling events.

```
> plot
Generating enhanced network plot...
✓ Enhanced network plot saved as: plots/network_plot_tick_42.png
Showing 8 traveling events and 15 external inputs
```

**Plot Features:**
- 🔴 Red nodes: Firing neurons (output > 0)
- 🟠 Orange nodes: High potential neurons (> 0.5)
- 🟡 Yellow nodes: Active neurons (> 0)
- 🔵 Blue nodes: Inactive neurons
- 🟢 Green squares: External input synapses
- 🟠 Orange dots: Presynaptic events traveling
- 🟣 Purple dots: Retrograde events traveling

#### `state`
Display comprehensive network state information.

```
> state

╭─ Core State ─╮
│ Current Tick │ 42        │
│ Is Running   │ False     │
│ Tick Rate    │ 1.00 tps  │
╰─────────────────────────╯

╭─ Network State ─╮
│ Neurons     │ 20  │
│ Connections │ 25  │
╰──────────────────╯
```

#### `status`
Show current network status (same as automatic status display).

```
> status
● Running at 2.5 TPS | Tick: 42 | Neurons: 20 | Active: 3
Connections: 25 | External: 15 | Free: 35 | Synaptic Density: 67.30% | Graph Density: 23.10%
Activity: Avg Potential: 0.234 | Avg Rate: 0.012 | Peak: 0.891
```

### System Commands

#### `context, ctx`
Display current context settings.

```
> context

╭─ Current Context ─╮
│ Type    │ ID    │ Status   │
├─────────┼───────┼──────────┤
│ Neuron  │ 12345 │ ✓ Valid  │
│ Synapse │ 2     │ ✓ Valid  │
╰─────────────────────────────╯

Commands will use this neuron ID when not specified
Commands will use this synapse ID when not specified
```

#### `set_neuron [id]`
Set neuron context for focused operations.

```
> set_neuron 12345           # Set neuron 12345 as context
> set_neuron                 # Prompted for neuron ID
✓ Set neuron context to 12345
```

#### `set_synapse [id]`
Set synapse context (requires neuron context).

```
> set_synapse 2              # Set synapse 2 as context
> set_synapse                # Prompted for synapse ID
✓ Set synapse context to 2
```

#### `clear_context`
Clear all context settings.

```
> clear_context
✓ Context cleared
```

#### `log_level`
Configure logging verbosity.

```
> log_level
Available log levels:
  1. TRACE
  2. DEBUG  
  3. INFO
  4. SUCCESS
  5. WARNING
  6. ERROR
  7. CRITICAL

Select log level (1-7) or enter level name [INFO]: DEBUG
✓ Log level set to: DEBUG
```

#### `toggle_timing`
Enable/disable command execution timing display.

```
> toggle_timing
✓ Command timing is now enabled

> tick
✓ Tick 43 completed
Command execution time: 0.0234 seconds
```

#### `clear`
Clear the terminal screen.

#### `help, h`
Display comprehensive help information with all commands and examples.

#### `exit, quit`
Exit the application gracefully.

## Context System

The context system allows you to focus operations on specific neurons and synapses, eliminating the need to repeatedly specify IDs.

### Setting Context

```bash
> set_neuron 12345           # Set neuron context
> set_synapse 2              # Set synapse context (requires neuron context)
```

### Using Context

Once context is set, commands will use these values when parameters are omitted:

```bash
> signal 1.5                 # Uses context neuron/synapse
> get_neuron                 # Uses context neuron
> get_synapse                # Uses context neuron/synapse
```

### Context Display

Context information appears in the status display:

```
● Stopped | Tick: 42 | Neurons: 20 | Active: 0
Context: N:12345 | S:2
```

## Advanced Features

### Autocomplete & History

- **TAB**: Autocomplete commands
- **↑/↓**: Navigate command history
- **History**: Saved to `.neuron_cli_history`

### Parameter Flexibility

Commands accept parameters in multiple ways:

```bash
> signal 12345 0 1.5         # All parameters provided
> signal 12345 0             # Prompted for remaining parameters
> signal                     # Prompted for all parameters
> signal 1.5                 # Uses context neuron/synapse
```

### Batch Operations

Create multiple neurons efficiently:

```bash
> add_neuron 4 2 100         # Create 100 neurons at once
```

Send multiple signals:

```bash
> signal 12345 0 1.0 50      # Send 50 signals
```

### Network Statistics

The CLI automatically calculates and displays:

- **Synaptic Density**: Percentage of synapses that are connected
- **Graph Density**: Percentage of possible connections that exist
- **Activity Metrics**: Average potential, firing rates, peak activity

### File Organization

```
project/
├── cli/
│   ├── neuron_cli.py        # Main CLI application
│   ├── networks/            # Network configuration files (initially empty)
│   └── .neuron_cli_history  # Command history (created on first run)
├── plots/                   # Generated visualizations (created by plot command)
│   └── network_plot_tick_*.png
├── neuron/                  # Core neural network modules
├── tests/                   # Test files
└── pyproject.toml / uv.lock  # Python dependencies (uv)
```

**Note:** The `plots/` directory and `.neuron_cli_history` file are created automatically when first needed.

## Examples & Workflows

### Creating a Simple Network

```bash
# Start CLI
> python neuron_cli.py

# Create 5 neurons with 4 synapses, 2 terminals each
> add_neuron 4 2 5

# Auto-connect neurons (leave 1 free synapse, 1 free terminal)
> auto_connect 1 1

# View external inputs
> list_external

# Send a signal to stimulate the network
> signal 12345 0 2.0

# Run simulation for 20 steps
> nticks 20

# Visualize the network
> plot

# Check network status
> status
```

### Using Context for Focused Analysis

```bash
# Set context to a specific neuron
> set_neuron 12345

# Set synapse context
> set_synapse 0

# View current context
> context

# Send signals using context (no need to specify IDs)
> signal 1.5
> signal 2.0
> signal 1.8

# Get detailed neuron information
> get_neuron

# Run simulation and monitor
> nticks 10
> plot
```

### Network Analysis Workflow

```bash
# Import existing network
> import networks/experiment_1.json

# Check network status
> status

# View network structure
> plot

# Send test signals
> batch_signal

# Run extended simulation
> start
# (let it run for a while)
> stop

# Analyze results
> state
> plot

# Export results
> export experiment_1_results.json
```

### Debugging Network Issues

```bash
# Enable detailed logging
> log_level DEBUG

# Enable timing to identify slow operations
> toggle_timing

# Check for free terminals
> list_free_outputs

# Check external inputs
> list_external

# Step through simulation slowly
> tick
> status
> tick
> status

# Clear problematic connections
> clear_connections

# Rebuild network
> auto_connect 0 1
```

## Troubleshooting

### Common Issues

**No network loaded**
```
Error: No network loaded
Solution: Create neurons with 'add_neuron' or import with 'import'
```

**Import failed**
```
Error: Import failed
Solutions:
- Check file path exists (networks/ directory starts empty)
- Verify JSON format
- Check file permissions
- Create a test network first with add_neuron, then export it
```

**Connection failed**
```
Error: Failed to create connection
Solutions:
- Verify source neuron has free terminals: 'list_free_outputs'
- Check target synapse exists: 'get_neuron [target_id]'
- Ensure synapse isn't already connected: 'list_external'
```

**Plot generation fails**
```
Error: Import error: No module named 'matplotlib'
Solution: from the repo root run `uv sync` (core deps include `matplotlib` and `networkx`).
```

### Performance Tips

1. **Large Networks**: Use `auto_connect` instead of manual connections
2. **Batch Operations**: Create multiple neurons at once with `add_neuron`
3. **Timing**: Enable timing to identify slow commands
4. **Logging**: Use appropriate log levels (INFO for normal use, DEBUG for troubleshooting)

### Network Design Guidelines

1. **Balanced Topology**: Leave some free synapses and terminals for flexibility
2. **Realistic Parameters**: Default neuron parameters are biologically inspired
3. **Signal Strength**: Start with signals around 1.0-2.0 strength
4. **Simulation Length**: Run 10-50 ticks to see meaningful dynamics

### Keyboard Shortcuts

- **Ctrl+C**: Interrupt current command (doesn't exit)
- **Ctrl+D** or **EOF**: Exit CLI
- **TAB**: Autocomplete commands
- **↑/↓**: Navigate command history
- **Ctrl+L**: Clear screen (alternative to `clear`)

---

## Support

For questions, issues, or contributions, please refer to the project documentation or contact the development team.

**Happy neural network building! 🧠⚡**
