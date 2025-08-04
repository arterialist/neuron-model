#!/usr/bin/env python3
"""
Simple Neural Network CLI Tool with Rich Styling
Clean command-line interface without complex TUI features
"""

import os
import signal
import sys
import traceback
import time
import functools

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm

# Prompt toolkit for autocomplete and history
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import clear

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Network imports
from neuron.nn_core import NNCore
from neuron.neuron import NeuronParameters


def timed_command(func):
    """Decorator to print command execution time if timing is enabled."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "timing_enabled"):
            return func(self, *args, **kwargs)
        if self.timing_enabled:
            start = time.perf_counter()
            result = func(self, *args, **kwargs)
            elapsed = time.perf_counter() - start
            self.console.print(
                f"[dim]Command execution time: {elapsed:.4f} seconds[/dim]"
            )
            return result
        else:
            return func(self, *args, **kwargs)

    return wrapper


class NeuralNetworkCLI:
    """
    Simple command-line interface for Neural Network operations
    """

    def __init__(self):
        self.console = Console()
        self.nn_core = NNCore()
        self.running = True
        self.timing_enabled = False  # Timing toggle

        # Context for focused operations
        self.context_neuron_id = None
        self.context_synapse_id = None

        # Command mapping
        self.commands = {
            "help": self.cmd_help,
            "h": self.cmd_help,
            "tick": self.cmd_tick,
            "nticks": self.cmd_n_ticks,
            "start": self.cmd_start_time,
            "stop": self.cmd_stop_time,
            "import": self.cmd_import_network,
            "export": self.cmd_export_network,
            "add_neuron": self.cmd_add_neuron,
            "get_neuron": self.cmd_get_neuron,
            "del_neuron": self.cmd_delete_neuron,
            "add_synapse": self.cmd_add_synapse,
            "get_synapse": self.cmd_get_synapse,
            "del_synapse": self.cmd_delete_synapse,
            "add_connection": self.cmd_add_connection,
            "auto_connect": self.cmd_auto_connect,
            "clear_connections": self.cmd_clear_connections,
            "list_external": self.cmd_list_external,
            "signal": self.cmd_send_signal,
            "batch_signal": self.cmd_batch_signals,
            "plot": self.cmd_plot_network,
            "state": self.cmd_detailed_state,
            "status": self.cmd_status,
            "log_level": self.cmd_set_log_level,
            "context": self.cmd_show_context,
            "ctx": self.cmd_show_context,
            "set_neuron": self.cmd_set_neuron_context,
            "set_synapse": self.cmd_set_synapse_context,
            "clear_context": self.cmd_clear_context,
            "clear": self.cmd_clear,
            "exit": self.cmd_exit,
            "quit": self.cmd_exit,
            "toggle_timing": self.cmd_toggle_timing,
            "list_free_outputs": self.cmd_list_free_outputs,
        }

        # Set up autocomplete and history
        self.setup_prompt()

    def setup_prompt(self):
        """Set up autocomplete and history for the prompt"""
        # Create completer with all commands
        command_list = list(self.commands.keys())
        self.completer = WordCompleter(
            command_list,
            ignore_case=True,
            sentence=False,  # Complete individual words, not full sentences
        )

        # Set up history (saves to .neuron_cli_history in current directory)
        try:
            self.history = FileHistory(".neuron_cli_history")
        except Exception:
            # Fallback to in-memory history if file history fails
            from prompt_toolkit.history import InMemoryHistory

            self.history = InMemoryHistory()

        # Create prompt session
        self.session = PromptSession(
            history=self.history,
            completer=self.completer,
        )

    def print_header(self):
        """Print application header"""
        header = Panel(
            "[bold cyan]Neural Network CLI[/bold cyan]\n"
            "Type 'help' for available commands, 'exit' to quit\n"
            "[dim]Use TAB for autocomplete, ↑↓ arrows for history[/dim]",
            style="blue",
            expand=False,
        )
        self.console.print(header)
        self.console.print()

    def print_status(self):
        """Print current network status"""
        state = self.nn_core.get_network_state()

        if "error" in state:
            self.console.print("[red]●[/red] No network loaded", style="dim")
            return

        core_state = state["core_state"]
        network_state = state["network"]

        # Basic counts
        num_neurons = len(network_state.get("neurons", {}))
        num_connections = len(network_state.get("connections", []))
        num_external_inputs = len(network_state.get("external_inputs", {}))

        # Calculate network statistics
        total_synapses = 0
        active_neurons = []
        neuron_details = []

        for neuron_id, neuron_data in network_state.get("neurons", {}).items():
            total_synapses += len(neuron_data.get("synapses", []))
            if neuron_data.get("output", 0) > 0:
                active_neurons.append(neuron_id)

            neuron_details.append(
                {
                    "id": neuron_id,
                    "potential": neuron_data.get("membrane_potential", 0),
                    "firing_rate": neuron_data.get("firing_rate", 0),
                    "output": neuron_data.get("output", 0),
                }
            )

        free_synapses = total_synapses - num_connections - num_external_inputs

        # Get densities from backend
        synaptic_density = network_state.get("synaptic_density", 0.0)
        graph_density = network_state.get("graph_density", 0.0)

        # Status indicator with more detail
        if core_state["is_running"]:
            status_text = (
                f"[green]●[/green] Running at {core_state['tick_rate']:.1f} TPS"
            )
        else:
            status_text = "[yellow]●[/yellow] Stopped"

        # Main status line
        self.console.print(
            f"{status_text} | "
            f"Tick: [cyan]{core_state['current_tick']}[/cyan] | "
            f"Neurons: [cyan]{num_neurons}[/cyan] | "
            f"Active: [bright_red]{len(active_neurons)}[/bright_red]"
        )

        # Network topology line
        self.console.print(
            f"[dim]Connections:[/dim] [magenta]{num_connections}[/magenta] | "
            f"[dim]External:[/dim] [blue]{num_external_inputs}[/blue] | "
            f"[dim]Free:[/dim] [green]{free_synapses}[/green] | "
            f"[dim]Synaptic Density:[/dim] [yellow]{synaptic_density:.2%}[/yellow] | "
            f"[dim]Graph Density:[/dim] [yellow]{graph_density:.2%}[/yellow]"
        )

        # Timing information
        if core_state["is_running"]:
            import time

            time_since_tick = time.time() - core_state.get("last_tick_time", 0)
            self.console.print(
                f"[dim]Last Tick:[/dim] [cyan]{time_since_tick:.1f}s ago[/cyan]"
            )

        # Activity summary
        if neuron_details:
            avg_potential = sum(n["potential"] for n in neuron_details) / len(
                neuron_details
            )
            avg_firing_rate = sum(n["firing_rate"] for n in neuron_details) / len(
                neuron_details
            )
            max_potential = max(n["potential"] for n in neuron_details)

            activity_color = (
                "red"
                if len(active_neurons) > 0
                else "yellow" if avg_potential > 0.1 else "green"
            )

            self.console.print(
                f"[dim]Activity:[/dim] [{activity_color}]Avg Potential: {avg_potential:.3f}[/{activity_color}] | "
                f"[dim]Avg Rate:[/dim] [cyan]{avg_firing_rate:.3f}[/cyan] | "
                f"[dim]Peak:[/dim] [red]{max_potential:.3f}[/red]"
            )

        # Context information (if any set)
        if self.context_neuron_id is not None or self.context_synapse_id is not None:
            context_info = []
            if self.context_neuron_id is not None:
                context_info.append(f"N:{self.context_neuron_id}")
            if self.context_synapse_id is not None:
                context_info.append(f"S:{self.context_synapse_id}")
            context_str = " | ".join(context_info)
            self.console.print(f"[dim]Context: {context_str}[/dim]")

    def run(self):
        """Main command loop"""
        self.print_header()

        while self.running:
            try:
                # Show status
                self.print_status()

                # Get command with autocomplete and history
                try:
                    user_input = self.session.prompt("\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    self.console.print("\n[yellow]Exiting...[/yellow]")
                    self.cmd_exit()
                    break

                if user_input == "":
                    continue

                # Parse command and arguments
                parts = user_input.split()
                command = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []

                if command in self.commands:
                    try:
                        # Pass arguments to command if the command supports them
                        import inspect

                        sig = inspect.signature(self.commands[command])
                        if "params" in sig.parameters:
                            self.commands[command](params=args)
                        else:
                            self.commands[command]()
                    except KeyboardInterrupt:
                        self.console.print("\n[yellow]Command interrupted[/yellow]")
                    except Exception as e:
                        self.console.print(f"[red]Error executing command: {e}[/red]")
                else:
                    self.console.print(f"[red]Unknown command: {command}[/red]")
                    self.console.print(
                        "Type 'help' for available commands or use TAB for autocomplete"
                    )

                self.console.print()  # Add spacing

            except Exception as e:
                self.console.print(f"[red]Unexpected error: {e}[/red]")
                self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
                self.cmd_exit()

    @timed_command
    def cmd_help(self):
        """Show help information"""
        commands_info = [
            ("help, h", "Show this help message"),
            ("", ""),
            ("== Core Operations ==", ""),
            ("tick", "Execute single tick"),
            ("nticks [N]", "Execute N ticks (default: 10)"),
            ("start", "Start automatic time progression"),
            ("stop", "Stop automatic time progression"),
            ("", ""),
            ("== Network Management ==", ""),
            ("import", "Import network from JSON file"),
            ("export", "Export network to JSON file"),
            ("", ""),
            ("== Neuron Operations ==", ""),
            (
                "add_neuron [SYNAPSES] [TERMINALS] [COUNT]",
                "Add neurons (defaults: 4, 1, 1)",
            ),
            ("get_neuron [ID]", "Get detailed neuron information"),
            ("del_neuron", "Delete a neuron"),
            ("", ""),
            ("== Synapse Operations ==", ""),
            ("add_synapse", "Add synapse to a neuron"),
            ("get_synapse", "Get synapse information"),
            ("del_synapse", "Delete a synapse"),
            ("", ""),
            ("== Connection Management ==", ""),
            ("add_connection [SRC_N] [SRC_T] [TGT_N] [TGT_S]", "Connect neurons"),
            (
                "auto_connect [MIN_SYN] [MIN_TERM]",
                "Auto-connect neurons (defaults: 1, 1)",
            ),
            ("clear_connections", "Clear all connections in the network"),
            ("list_external", "List external input synapses"),
            ("", ""),
            ("== Signal Operations ==", ""),
            ("signal [NEURON] [SYNAPSE] [STRENGTH] [REPEAT]", "Send signal to synapse"),
            ("batch_signal", "Send multiple signals"),
            ("", ""),
            ("== Visualization & Analysis ==", ""),
            ("plot", "Generate network visualization"),
            ("state", "Show detailed network state"),
            ("status", "Show enhanced network status"),
            ("", ""),
            ("== System ==", ""),
            ("context, ctx", "Show current context"),
            ("set_neuron [ID]", "Set neuron context"),
            ("set_synapse [ID]", "Set synapse context"),
            ("clear_context", "Clear all context"),
            ("log_level", "Set logging level"),
            ("clear", "Clear screen"),
            ("exit, quit", "Exit the application"),
            ("toggle_timing", "Toggle command execution timing display on/off"),
            ("list_free_outputs", "List free outputs"),
        ]

        # Create help table
        help_table = Table(title="Neural Network CLI Commands", show_header=False)
        help_table.add_column("Command", style="cyan", min_width=35)
        help_table.add_column("Description", style="white")

        for cmd, desc in commands_info:
            if cmd == "":
                help_table.add_row("", "")
            elif cmd.startswith("=="):
                help_table.add_row(f"[bold yellow]{cmd}[/bold yellow]", "")
            else:
                help_table.add_row(cmd, desc)

        self.console.print(help_table)

        # Parameter usage info
        self.console.print("\n[bold cyan]Parameter Usage:[/bold cyan]")
        self.console.print(
            "[dim]Commands with [BRACKETS] accept optional parameters.[/dim]"
        )
        self.console.print("[dim]Examples:[/dim]")
        self.console.print(
            "  [green]signal 12345 2 1.5[/green]     - Send signal to neuron 12345, synapse 2, strength 1.5"
        )
        self.console.print("  [green]nticks 50[/green]            - Execute 50 ticks")
        self.console.print(
            "  [green]add_neuron 6 2 5[/green]     - Add 5 neurons with 6 synapses, 2 terminals each"
        )
        self.console.print(
            "  [green]get_neuron 12345[/green]     - Get info for neuron 12345"
        )
        self.console.print(
            "  [green]add_connection 100 0 200 0[/green] - Connect neuron 100 terminal 0 to neuron 200 synapse 0"
        )
        self.console.print(
            "  [green]auto_connect 0 2[/green]     - Auto-connect with 0 min free synapses, 2 min free terminals"
        )
        self.console.print(
            "  [green]set_neuron 12345[/green]     - Set neuron 12345 as context"
        )
        self.console.print(
            "  [green]set_synapse 2[/green]        - Set synapse 2 as context"
        )
        self.console.print(
            "\n[dim]Missing parameters will be prompted interactively.[/dim]"
        )

        # Context usage examples
        self.console.print("\n[bold cyan]Context Usage:[/bold cyan]")
        self.console.print("[dim]Set context to avoid typing IDs repeatedly:[/dim]")
        self.console.print(
            "  [green]set_neuron 12345[/green]       - Set neuron 12345 as context"
        )
        self.console.print(
            "  [green]set_synapse 2[/green]         - Set synapse 2 as context"
        )
        self.console.print(
            "  [green]signal 1.5[/green]            - Send signal using context neuron/synapse"
        )
        self.console.print(
            "  [green]get_neuron[/green]            - Get info for context neuron"
        )
        self.console.print(
            "  [green]context[/green]               - Show current context"
        )
        self.console.print("  [green]clear_context[/green]         - Clear all context")

    @timed_command
    def cmd_tick(self):
        """Execute single tick"""
        result = self.nn_core.do_tick()
        if "error" in result:
            self.console.print(f"[red]Tick error: {result['error']}[/red]")
        else:
            self.console.print(
                f"[green]Tick {result.get('tick', 'N/A')} completed[/green]"
            )
            if result.get("total_activity", 0) > 0:
                self.console.print(
                    f"[yellow]Total activity: {result['total_activity']}[/yellow]"
                )

    @timed_command
    def cmd_n_ticks(self, params=None):
        """Run multiple ticks"""
        try:
            # Extract n_ticks from parameters if provided
            n_ticks = None
            if params and len(params) >= 1:
                try:
                    n_ticks = int(params[0])
                    if n_ticks <= 0:
                        self.console.print(
                            "[red]Number of ticks must be positive[/red]"
                        )
                        return
                except ValueError:
                    self.console.print(
                        f"[red]Invalid number of ticks: {params[0]}[/red]"
                    )
                    return

            # Prompt if not provided
            if n_ticks is None:
                n_ticks = IntPrompt.ask("Number of ticks", default=10)

            if n_ticks <= 0:
                self.console.print("[red]Number of ticks must be positive[/red]")
                return

            # Use do_n_ticks method
            results = self.nn_core.do_n_ticks(n_ticks)

            # Count successful ticks and activity
            successful_ticks = sum(1 for r in results if "error" not in r)
            active_ticks = sum(1 for r in results if r.get("total_activity", 0) > 0)

            self.console.print(
                f"[green]✓[/green] Completed {successful_ticks}/{n_ticks} ticks"
            )
            if active_ticks > 0:
                self.console.print(
                    f"[cyan]Neural activity in {active_ticks} ticks[/cyan]"
                )

        except ValueError as e:
            self.console.print(f"[red]Invalid input: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error during ticks: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

    @timed_command
    def cmd_start_time(self):
        """Start autonomous time flow"""
        try:
            rate = FloatPrompt.ask("Tick rate (tps)", default=1.0)
            if self.nn_core.start_time_flow(rate):
                self.console.print(f"[green]Time flow started at {rate} tps[/green]")
            else:
                self.console.print("[yellow]Time flow already running[/yellow]")
        except ValueError:
            self.console.print("[red]Invalid rate[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error setting tick duration: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

    @timed_command
    def cmd_stop_time(self):
        """Stop autonomous time flow"""
        if self.nn_core.stop_time_flow():
            self.console.print("[green]Time flow stopped[/green]")
        else:
            self.console.print("[yellow]Time flow not running[/yellow]")

    @timed_command
    def cmd_import_network(self, params=None):
        """Import network"""
        try:
            # Get filepath from params or prompt
            filepath = None
            if params and len(params) >= 1:
                filepath = params[0]
            else:
                # Only prompt if input is interactive (to avoid EOF errors when piping)
                if hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
                    filepath = Prompt.ask(
                        "Config file path", default="networks/network.json"
                    )
                else:
                    filepath = "networks/network.json"  # Default when piping

            if self.nn_core.import_network(filepath):
                self.console.print(f"[green]Network imported from {filepath}[/green]")
            else:
                self.console.print("[red]Import failed[/red]")
        except ValueError as e:
            self.console.print(f"[red]Invalid input: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error importing network: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

    @timed_command
    def cmd_export_network(self):
        """Export network"""
        try:
            filepath = Prompt.ask("Export file path", default="network.json")
            os.makedirs("networks", exist_ok=True)
            filepath = f"networks/{filepath}"
            if self.nn_core.export_network(filepath):
                self.console.print(f"[green]Network exported to {filepath}[/green]")
            else:
                self.console.print("[red]Export failed[/red]")
        except ValueError as e:
            self.console.print(f"[red]Invalid input: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error exporting network: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

    @timed_command
    def cmd_add_neuron(self, params=None):
        """Add one or more new neurons"""
        try:
            # Extract parameters if provided
            num_synapses = None
            num_terminals = None
            num_neurons = None

            if params:
                if len(params) >= 1:
                    try:
                        num_synapses = int(params[0])
                        if num_synapses < 0:
                            self.console.print(
                                "[red]Number of synapses cannot be negative[/red]"
                            )
                            return
                    except ValueError:
                        self.console.print(
                            f"[red]Invalid number of synapses: {params[0]}[/red]"
                        )
                        return

                if len(params) >= 2:
                    try:
                        num_terminals = int(params[1])
                        if num_terminals < 0:
                            self.console.print(
                                "[red]Number of terminals cannot be negative[/red]"
                            )
                            return
                    except ValueError:
                        self.console.print(
                            f"[red]Invalid number of terminals: {params[1]}[/red]"
                        )
                        return

                if len(params) >= 3:
                    try:
                        num_neurons = int(params[2])
                        if num_neurons <= 0:
                            self.console.print(
                                "[red]Number of neurons must be positive[/red]"
                            )
                            return
                    except ValueError:
                        self.console.print(
                            f"[red]Invalid number of neurons: {params[2]}[/red]"
                        )
                        return

            # Prompt for missing parameters
            if num_synapses is None:
                num_synapses = IntPrompt.ask("Number of synapses", default=4)
            if num_terminals is None:
                num_terminals = IntPrompt.ask("Number of terminals", default=1)
            if num_neurons is None:
                num_neurons = IntPrompt.ask("Number of neurons to create", default=1)

            if num_synapses < 0:
                self.console.print("[red]Number of synapses cannot be negative[/red]")
                return
            if num_terminals < 0:
                self.console.print("[red]Number of terminals cannot be negative[/red]")
                return
            if num_neurons <= 0:
                self.console.print("[red]Number of neurons must be positive[/red]")
                return

            # Get existing neurons to avoid conflicts
            import random
            import numpy as np

            existing_neurons = set()
            state = self.nn_core.get_network_state()
            if "network" in state:
                existing_neurons = set(state["network"]["neurons"].keys())

            # Create multiple neurons
            success_count = 0
            created_neurons = []

            for i in range(num_neurons):
                # Generate a unique neuron ID
                max_attempts = 1000
                neuron_id = None
                for _ in range(max_attempts):
                    candidate_id = random.randint(1, 2**32 - 1)
                    if candidate_id not in existing_neurons:
                        neuron_id = candidate_id
                        existing_neurons.add(
                            neuron_id
                        )  # Add to set to avoid duplicates in batch
                        break

                if neuron_id is None:
                    self.console.print(
                        f"[red]Failed to generate unique neuron ID for neuron {i+1}[/red]"
                    )
                    continue

                # Create neuron parameters using NeuronParameters dataclass
                neuron_params = NeuronParameters(
                    num_neuromodulators=2,
                    num_inputs=num_synapses,
                    r_base=np.random.uniform(1.0, 1.2),
                    b_base=np.random.uniform(1.2, 1.4),
                    c=10,
                    lambda_param=20,
                    p=1.0,
                    gamma=np.array([0.99, 0.995]),
                    beta_avg=0.999,
                    w_r=np.array([-0.2, 0.05]),
                    w_b=np.array([-0.2, 0.05]),
                    w_tref=np.array([-20.0, 10.0]),
                    delta_decay=0.95,
                    eta_post=0.01,
                    eta_retro=0.01,
                )

                # Add neuron to network
                neuron_success = self.nn_core.add_neuron(neuron_id, neuron_params)
                if not neuron_success:
                    self.console.print(
                        f"[yellow]Failed to add neuron {neuron_id}[/yellow]"
                    )
                    continue

                # Add synapses to the neuron
                synapse_failures = 0
                for syn_id in range(num_synapses):
                    distance = random.randint(2, 7)  # Random distance from hillock
                    if not self.nn_core.add_synapse(neuron_id, syn_id, distance):
                        synapse_failures += 1

                # Add terminals (presynaptic points)
                neuron = (
                    self.nn_core.neural_net.network.neurons[neuron_id]
                    if self.nn_core.neural_net
                    else None
                )
                if neuron:
                    for term_id in range(num_terminals):
                        distance = random.randint(2, 7)
                        neuron.add_axon_terminal(term_id, distance)

                # Add synapses as external inputs by default
                for syn_id in range(num_synapses):
                    synapse_key = (neuron_id, syn_id)
                    if self.nn_core.neural_net:
                        self.nn_core.neural_net.network.external_inputs[synapse_key] = {
                            "info": 0.0,
                            "mod": np.array([0.0, 0.0]),
                        }

                success_count += 1
                created_neurons.append(neuron_id)

                # Show progress for multiple neurons
                if num_neurons > 1 and (i + 1) % max(1, num_neurons // 10) == 0:
                    progress = (i + 1) / num_neurons * 100
                    self.console.print(
                        f"[dim]Progress: {progress:.0f}% ({i + 1}/{num_neurons})[/dim]"
                    )

                if synapse_failures > 0:
                    self.console.print(
                        f"[yellow]Neuron {neuron_id}: {synapse_failures}/{num_synapses} synapses failed[/yellow]"
                    )

            # Report results
            if num_neurons == 1:
                if success_count > 0:
                    self.console.print(
                        f"[green]✓[/green] Added neuron {created_neurons[0]} with {num_synapses} synapses and {num_terminals} terminals"
                    )
                else:
                    self.console.print("[red]✗[/red] Failed to add neuron")
            else:
                self.console.print(
                    f"[green]✓[/green] Created {success_count}/{num_neurons} neurons successfully"
                )
                if success_count < num_neurons:
                    failed_count = num_neurons - success_count
                    self.console.print(
                        f"[yellow]⚠[/yellow] {failed_count} neurons failed"
                    )

                if created_neurons:
                    self.console.print(
                        f"[cyan]Created neuron IDs: {created_neurons[:10]}{'...' if len(created_neurons) > 10 else ''}[/cyan]"
                    )

        except ValueError as e:
            self.console.print(f"[red]Invalid input: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error adding neuron: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

    @timed_command
    def cmd_get_neuron(self, params=None):
        """Get detailed neuron information"""
        try:
            # Extract neuron_id from parameters if provided
            neuron_id = None
            if params and len(params) >= 1:
                try:
                    neuron_id = int(params[0])
                except ValueError:
                    self.console.print(f"[red]Invalid neuron ID: {params[0]}[/red]")
                    return

            # Use context if no parameter provided
            if neuron_id is None:
                if self.context_neuron_id is not None:
                    neuron_id = self.context_neuron_id
                    self.console.print(f"[dim]Using context neuron: {neuron_id}[/dim]")
                else:
                    neuron_id = IntPrompt.ask("Neuron ID")

            # Get neuron data
            neuron_data = self.nn_core.get_neuron(neuron_id)
            if not neuron_data:
                self.console.print(f"[red]Neuron {neuron_id} not found[/red]")
                return

            # Show basic information
            self.console.print(
                f"\n[bold cyan]Neuron {neuron_id} Information[/bold cyan]"
            )

            basic_table = Table(title="Basic Information", show_header=True)
            basic_table.add_column("Property", style="cyan", no_wrap=True)
            basic_table.add_column("Value", style="white")

            basic_table.add_row("Neuron ID", str(neuron_id))
            basic_table.add_row(
                "Membrane Potential", f"{neuron_data.get('membrane_potential', 0):.6f}"
            )
            basic_table.add_row(
                "Firing Rate", f"{neuron_data.get('firing_rate', 0):.6f}"
            )
            basic_table.add_row("Output", f"{neuron_data.get('output', 0):.6f}")

            synapses = neuron_data.get("synapses", [])
            terminals = neuron_data.get("terminals", [])
            basic_table.add_row("Synapse Count", str(len(synapses)))
            basic_table.add_row("Terminal Count", str(len(terminals)))

            self.console.print(basic_table)

            # Show synapses if they exist
            if synapses:
                synapses_table = Table(title="Synapses", show_header=True)
                synapses_table.add_column("ID", style="yellow", no_wrap=True)
                synapses_table.add_column("Distance", style="green", no_wrap=True)
                synapses_table.add_column("Connected", style="blue", no_wrap=True)
                synapses_table.add_column("Input Type", style="magenta")
                synapses_table.add_column("Potential", style="red")

                # Get network state to check connections
                state = self.nn_core.get_network_state()
                connections = state.get("network", {}).get("connections", [])
                external_inputs = state.get("network", {}).get("external_inputs", {})

                for synapse in synapses:
                    # Handle case where synapse might be just an ID (int) or a full dict
                    if isinstance(synapse, dict):
                        syn_id = synapse.get("id", "N/A")
                        distance = synapse.get("distance_to_hillock", "N/A")
                        potential = f"{synapse.get('potential', 0):.6f}"
                    else:
                        # synapse is just an integer ID
                        syn_id = synapse
                        # Get detailed synapse info
                        synapse_data = self.nn_core.get_synapse(neuron_id, syn_id)
                        if synapse_data:
                            distance = synapse_data.get("distance_to_hillock", "N/A")
                            potential = f"{synapse_data.get('potential', 0):.6f}"
                        else:
                            distance = "N/A"
                            potential = "N/A"

                    # Check connection status
                    connected = "No"
                    input_type = "Free"

                    # Check if connected to another neuron
                    for conn in connections:
                        if (
                            len(conn) >= 4
                            and conn[2] == neuron_id
                            and conn[3] == syn_id
                        ):
                            connected = "Yes"
                            input_type = f"Neuron {conn[0]}:{conn[1]}"
                            break

                    # Check if external input
                    ext_key = f"{neuron_id}_{syn_id}"
                    if ext_key in external_inputs:
                        connected = "Yes"
                        input_type = "External"

                    synapses_table.add_row(
                        str(syn_id), str(distance), connected, input_type, potential
                    )

                self.console.print(synapses_table)

            # Show connection topology
            if connections:
                # Incoming connections
                incoming = [c for c in connections if len(c) >= 4 and c[2] == neuron_id]
                outgoing = [c for c in connections if len(c) >= 4 and c[0] == neuron_id]

                if incoming or outgoing:
                    self.console.print(f"\n[bold]Connection Topology[/bold]")

                    if incoming:
                        self.console.print(f"[cyan]Incoming ({len(incoming)}):[/cyan]")
                        for conn in incoming[:10]:  # Limit to first 10
                            source, source_term, _, synapse = conn
                            self.console.print(
                                f"  N{source}:{source_term} → S{synapse}"
                            )
                        if len(incoming) > 10:
                            self.console.print(f"  ... and {len(incoming) - 10} more")

                    if outgoing:
                        self.console.print(
                            f"[magenta]Outgoing ({len(outgoing)}):[/magenta]"
                        )
                        for conn in outgoing[:10]:  # Limit to first 10
                            _, source_term, target, synapse = conn
                            self.console.print(
                                f"  T{source_term} → N{target}:S{synapse}"
                            )
                        if len(outgoing) > 10:
                            self.console.print(f"  ... and {len(outgoing) - 10} more")

            # Ask if user wants detailed parameters
            if Confirm.ask("\nShow detailed neuron parameters?", default=False):
                params_table = Table(title="Neuron Parameters", show_header=True)
                params_table.add_column("Parameter", style="cyan", no_wrap=True)
                params_table.add_column("Value", style="white")

                for key, value in neuron_data.items():
                    if key not in ["synapses", "terminals"]:
                        # Format arrays nicely
                        if isinstance(value, np.ndarray):
                            if value.size <= 10:
                                value_str = str(value.tolist())
                            else:
                                value_str = f"Array({value.shape}) [{value.flat[0]:.3f}, ..., {value.flat[-1]:.3f}]"
                        else:
                            value_str = str(value)

                        params_table.add_row(key, value_str)

                self.console.print(params_table)

        except ValueError as e:
            self.console.print(f"[red]Invalid input: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error getting neuron: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

    @timed_command
    def cmd_delete_neuron(self):
        """Delete neuron"""
        try:
            neuron_id = IntPrompt.ask("Neuron ID")
            if Confirm.ask(f"Delete neuron {neuron_id}?"):
                if self.nn_core.delete_neuron(neuron_id):
                    self.console.print(f"[green]Neuron {neuron_id} deleted[/green]")
                else:
                    self.console.print("[red]Failed to delete neuron[/red]")
        except ValueError:
            self.console.print("[red]Invalid neuron ID[/red]")

    @timed_command
    def cmd_add_synapse(self):
        """Add synapse"""
        try:
            neuron_id = IntPrompt.ask("Neuron ID")
            synapse_id = IntPrompt.ask("Synapse ID")
            distance = IntPrompt.ask("Distance to hillock", default=5)

            if self.nn_core.add_synapse(neuron_id, synapse_id, distance):
                self.console.print(
                    f"[green]Synapse {synapse_id} added to neuron {neuron_id}[/green]"
                )
            else:
                self.console.print("[red]Failed to add synapse[/red]")
        except ValueError:
            self.console.print("[red]Invalid input[/red]")

    @timed_command
    def cmd_get_synapse(self):
        """Get synapse information"""
        try:
            # Use context or prompt for neuron ID
            neuron_id = None
            if self.context_neuron_id is not None:
                neuron_id = self.context_neuron_id
                self.console.print(f"[dim]Using context neuron: {neuron_id}[/dim]")
            else:
                neuron_id = IntPrompt.ask("Neuron ID")

            # Use context or prompt for synapse ID
            synapse_id = None
            if self.context_synapse_id is not None:
                synapse_id = self.context_synapse_id
                self.console.print(f"[dim]Using context synapse: {synapse_id}[/dim]")
            else:
                synapse_id = IntPrompt.ask("Synapse ID")

            # Get synapse data
            synapse_data = self.nn_core.get_synapse(neuron_id, synapse_id)
            if synapse_data:
                table = Table(title=f"Synapse {synapse_id} Information")
                table.add_column("Property", style="cyan")
                table.add_column("Value", style="white")

                table.add_row(
                    "Distance to Hillock", str(synapse_data["distance_to_hillock"])
                )
                table.add_row("Potential", f"{synapse_data['potential']:.3f}")

                self.console.print(table)
            else:
                self.console.print("[red]Synapse not found[/red]")

        except ValueError as e:
            self.console.print(f"[red]Invalid input: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error getting synapse: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

    @timed_command
    def cmd_delete_synapse(self):
        """Delete synapse"""
        try:
            neuron_id = IntPrompt.ask("Neuron ID")
            synapse_id = IntPrompt.ask("Synapse ID")
            if Confirm.ask(f"Delete synapse {synapse_id} from neuron {neuron_id}?"):
                if self.nn_core.delete_synapse(neuron_id, synapse_id):
                    self.console.print(
                        f"[green]Synapse {synapse_id} deleted from neuron {neuron_id}[/green]"
                    )
                else:
                    self.console.print("[red]Failed to delete synapse[/red]")
        except ValueError:
            self.console.print("[red]Invalid input[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error during auto-connect: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

    @timed_command
    def cmd_add_connection(self, params=None):
        """Add a connection between two neurons"""
        try:
            # Extract parameters if provided
            source_neuron_id = None
            source_terminal_id = None
            target_neuron_id = None
            target_synapse_id = None

            if params:
                if len(params) >= 1:
                    try:
                        source_neuron_id = int(params[0])
                    except ValueError:
                        self.console.print(
                            f"[red]Invalid source neuron ID: {params[0]}[/red]"
                        )
                        return
                if len(params) >= 2:
                    try:
                        source_terminal_id = int(params[1])
                    except ValueError:
                        self.console.print(
                            f"[red]Invalid source terminal ID: {params[1]}[/red]"
                        )
                        return
                if len(params) >= 3:
                    try:
                        target_neuron_id = int(params[2])
                    except ValueError:
                        self.console.print(
                            f"[red]Invalid target neuron ID: {params[2]}[/red]"
                        )
                        return
                if len(params) >= 4:
                    try:
                        target_synapse_id = int(params[3])
                    except ValueError:
                        self.console.print(
                            f"[red]Invalid target synapse ID: {params[3]}[/red]"
                        )
                        return

            # Prompt for missing parameters
            if source_neuron_id is None:
                if self.context_neuron_id is not None:
                    source_neuron_id = self.context_neuron_id
                    self.console.print(
                        f"[dim]Using context neuron as source: {source_neuron_id}[/dim]"
                    )
                else:
                    source_neuron_id = IntPrompt.ask("Source neuron ID")

            if source_terminal_id is None:
                source_terminal_id = IntPrompt.ask("Source terminal ID")

            if target_neuron_id is None:
                target_neuron_id = IntPrompt.ask("Target neuron ID")
            if target_synapse_id is None:
                target_synapse_id = IntPrompt.ask("Target synapse ID")

            success = self.nn_core.add_connection(
                source_neuron_id,
                source_terminal_id,
                target_neuron_id,
                target_synapse_id,
            )

            if success:
                self.console.print(
                    f"[green]✓[/green] Connected N{source_neuron_id}:T{source_terminal_id} → N{target_neuron_id}:S{target_synapse_id}"
                )
            else:
                self.console.print(f"[red]✗[/red] Failed to create connection")

        except ValueError as e:
            self.console.print(f"[red]Invalid input: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error adding connection: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

    @timed_command
    def cmd_auto_connect(self, params=None):
        """Automatically connect neurons"""
        try:
            # Extract parameters if provided
            min_free_synapses = None
            min_free_terminals = None

            if params:
                if len(params) >= 1:
                    try:
                        min_free_synapses = int(params[0])
                        if min_free_synapses < 0:
                            self.console.print(
                                "[red]Minimum free synapses cannot be negative[/red]"
                            )
                            return
                    except ValueError:
                        self.console.print(
                            f"[red]Invalid minimum free synapses: {params[0]}[/red]"
                        )
                        return

                if len(params) >= 2:
                    try:
                        min_free_terminals = int(params[1])
                        if min_free_terminals < 0:
                            self.console.print(
                                "[red]Minimum free terminals cannot be negative[/red]"
                            )
                            return
                    except ValueError:
                        self.console.print(
                            f"[red]Invalid minimum free terminals: {params[1]}[/red]"
                        )
                        return

            # Prompt for missing parameters
            if min_free_synapses is None:
                min_free_synapses = IntPrompt.ask(
                    "Minimum free synapses per neuron", default=1
                )
            if min_free_terminals is None:
                min_free_terminals = IntPrompt.ask(
                    "Minimum free terminals per neuron", default=1
                )

            if min_free_synapses < 0:
                self.console.print(
                    "[red]Minimum free synapses cannot be negative[/red]"
                )
                return
            if min_free_terminals < 0:
                self.console.print(
                    "[red]Minimum free terminals cannot be negative[/red]"
                )
                return

            # auto_connect_neurons returns boolean, but prints its own statistics
            success = self.nn_core.auto_connect_neurons(
                min_free_synapses, min_free_terminals
            )

            if not success:
                self.console.print(f"[red]✗[/red] Auto-connect failed")

        except ValueError as e:
            self.console.print(f"[red]Invalid input: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error during auto-connect: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

    @timed_command
    def cmd_clear_connections(self):
        """Clear all connections"""
        state = self.nn_core.get_network_state()
        if "error" in state:
            self.console.print("[red]No network loaded[/red]")
            return

        num_connections = len(state["network"]["connections"])
        if num_connections == 0:
            self.console.print("[yellow]No connections to clear[/yellow]")
            return

        if Confirm.ask(f"Clear all {num_connections} connections?"):
            if self.nn_core.clear_all_connections():
                self.console.print("[green]All connections cleared[/green]")
            else:
                self.console.print("[red]Failed to clear connections[/red]")
        else:
            self.console.print("[yellow]Operation cancelled[/yellow]")

    @timed_command
    def cmd_list_external(self):
        """List external inputs"""
        state = self.nn_core.get_network_state()
        if "error" in state:
            self.console.print("[red]No network loaded[/red]")
            return

        external_inputs = state["network"].get("external_inputs", {})
        if not external_inputs:
            self.console.print("[yellow]No external inputs found[/yellow]")
            self.console.print(
                "[dim]Tip: Use 'clear_connections' to convert neuron connections to external inputs[/dim]"
            )
            return

        # Create table for external inputs
        ext_table = Table(title="External Input Synapses", show_header=True)
        ext_table.add_column("Neuron ID", style="cyan", no_wrap=True)
        ext_table.add_column("Synapse ID", style="yellow", no_wrap=True)
        ext_table.add_column("Distance", style="green", no_wrap=True)
        ext_table.add_column("Info Signal", style="blue")
        ext_table.add_column("Mod Signals", style="magenta")
        ext_table.add_column("Synapse Potential", style="red")

        # Parse external inputs (format: "neuron_id_synapse_id": {...})
        external_list = []
        for ext_key, ext_data in external_inputs.items():
            # Parse the key format "neuron_id_synapse_id"
            if "_" in ext_key:
                parts = ext_key.split("_")
                if len(parts) >= 2:
                    try:
                        neuron_id = int(parts[0])
                        synapse_id = int(parts[1])
                        external_list.append((neuron_id, synapse_id, ext_data))
                    except ValueError:
                        continue

        # Sort by neuron ID, then synapse ID
        external_list.sort(key=lambda x: (x[0], x[1]))

        for neuron_id, synapse_id, ext_data in external_list:
            # Get detailed synapse information
            synapse_data = self.nn_core.get_synapse(neuron_id, synapse_id)

            distance = "N/A"
            potential = "N/A"
            if synapse_data:
                distance = str(synapse_data.get("distance_to_hillock", "N/A"))
                potential = f"{synapse_data.get('potential', 0):.6f}"

            # Format external input data
            info_signal = f"{ext_data.get('info', 0):.3f}"
            mod_signals = str(ext_data.get("mod", [0.0, 0.0]))

            ext_table.add_row(
                str(neuron_id),
                str(synapse_id),
                distance,
                info_signal,
                mod_signals,
                potential,
            )

        self.console.print(ext_table)

        # Summary information
        self.console.print(f"\n[dim]Total external inputs: {len(external_list)}[/dim]")

        # Group by neuron for summary
        neuron_counts = {}
        for neuron_id, _, _ in external_list:
            neuron_counts[neuron_id] = neuron_counts.get(neuron_id, 0) + 1

        if len(neuron_counts) > 1:
            self.console.print(
                "[dim]Per neuron: "
                + ", ".join(
                    f"N{nid}({count})" for nid, count in sorted(neuron_counts.items())
                )
                + "[/dim]"
            )

        self.console.print(
            "[dim]Use 'signal' command to send signals to these synapses[/dim]"
        )

    @timed_command
    def cmd_send_signal(self, params=None):
        """Send signal to synapse"""
        try:
            # Extract parameters if provided
            neuron_id = None
            synapse_id = None
            strength = None
            repeat_count = None

            if params:
                if len(params) >= 1:
                    try:
                        neuron_id = int(params[0])
                    except ValueError:
                        self.console.print(f"[red]Invalid neuron ID: {params[0]}[/red]")
                        return
                if len(params) >= 2:
                    try:
                        synapse_id = int(params[1])
                    except ValueError:
                        self.console.print(
                            f"[red]Invalid synapse ID: {params[1]}[/red]"
                        )
                        return
                if len(params) >= 3:
                    try:
                        strength = float(params[2])
                    except ValueError:
                        self.console.print(f"[red]Invalid strength: {params[2]}[/red]")
                        return
                if len(params) >= 4:
                    try:
                        repeat_count = int(params[3])
                        if repeat_count <= 0:
                            self.console.print(
                                "[red]Repeat count must be positive[/red]"
                            )
                            return
                    except ValueError:
                        self.console.print(
                            f"[red]Invalid repeat count: {params[3]}[/red]"
                        )
                        return

            # Use context or prompt for missing parameters
            if neuron_id is None:
                if self.context_neuron_id is not None:
                    neuron_id = self.context_neuron_id
                    self.console.print(f"[dim]Using context neuron: {neuron_id}[/dim]")
                else:
                    neuron_id = IntPrompt.ask("Neuron ID")

            if synapse_id is None:
                if self.context_synapse_id is not None:
                    synapse_id = self.context_synapse_id
                    self.console.print(
                        f"[dim]Using context synapse: {synapse_id}[/dim]"
                    )
                else:
                    synapse_id = IntPrompt.ask("Synapse ID")

            if strength is None:
                strength = FloatPrompt.ask("Signal strength", default=1.0)

            if repeat_count is None:
                repeat_count = IntPrompt.ask("Repeat count", default=1)

            if repeat_count < 1:
                self.console.print("[red]Repeat count must be at least 1[/red]")
                return

            # Send signal(s)
            success_count = 0
            for i in range(repeat_count):
                success = self.nn_core.send_signal(neuron_id, synapse_id, strength)
                if success:
                    success_count += 1

                # Show progress for multiple signals
                if repeat_count > 1 and (i + 1) % max(1, repeat_count // 10) == 0:
                    progress = (i + 1) / repeat_count * 100
                    self.console.print(
                        f"[dim]Progress: {progress:.0f}% ({i + 1}/{repeat_count})[/dim]"
                    )

            # Report results
            if repeat_count == 1:
                if success_count > 0:
                    self.console.print(
                        f"[green]✓[/green] Signal sent to neuron {neuron_id}, synapse {synapse_id}"
                    )
                else:
                    self.console.print(f"[red]✗[/red] Failed to send signal")
            else:
                self.console.print(
                    f"[green]✓[/green] Sent {success_count}/{repeat_count} signals successfully"
                )
                if success_count < repeat_count:
                    failed_count = repeat_count - success_count
                    self.console.print(
                        f"[yellow]⚠[/yellow] {failed_count} signals failed"
                    )

        except ValueError as e:
            self.console.print(f"[red]Invalid input: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error sending signal: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

    @timed_command
    def cmd_batch_signals(self):
        """Send batch signals"""
        try:
            self.console.print("[cyan]Batch Signal Configuration[/cyan]")

            # Get number of signals
            num_signals = IntPrompt.ask("Number of signals to send", default=3)

            signals = []
            for i in range(num_signals):
                self.console.print(f"\n[yellow]Signal {i+1}:[/yellow]")
                neuron_id = IntPrompt.ask("  Target neuron ID")
                synapse_id = IntPrompt.ask("  Target synapse ID")
                strength = FloatPrompt.ask("  Signal strength", default=1.5)

                signals.append((neuron_id, synapse_id, strength))

            # Confirm before sending
            table = Table(title="Batch Signals to Send", show_header=True)
            table.add_column("Signal", style="cyan")
            table.add_column("Neuron ID", style="yellow")
            table.add_column("Synapse ID", style="yellow")
            table.add_column("Strength", style="green")

            for i, (nid, sid, strength) in enumerate(signals):
                table.add_row(str(i + 1), str(nid), str(sid), f"{strength:.2f}")

            self.console.print(table)

            if Confirm.ask("Send these signals?"):
                success_count = self.nn_core.send_batch_signals(signals)
                self.console.print(
                    f"[green]Successfully sent {success_count}/{len(signals)} signals[/green]"
                )
                if success_count < len(signals):
                    failed_count = len(signals) - success_count
                    self.console.print(f"[red]{failed_count} signals failed[/red]")
            else:
                self.console.print("[yellow]Batch signals cancelled[/yellow]")

        except ValueError:
            self.console.print("[red]Invalid input[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error creating plot: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

    @timed_command
    def cmd_plot_network(self):
        """Plot network with traveling signals and external synapses"""
        state = self.nn_core.get_network_state()
        if "error" in state:
            self.console.print("[red]No network to plot[/red]")
            return

        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            import matplotlib.colors as mcolors
            import numpy as np

            network_state = state["network"]
            neurons = network_state["neurons"]
            connections = network_state["connections"]
            external_inputs = network_state["external_inputs"]
            traveling_events = network_state["traveling_signals"]
            current_tick = state["core_state"]["current_tick"]

            if not neurons:
                self.console.print("[red]No neurons in network to plot[/red]")
                return

            self.console.print("[cyan]Generating enhanced network plot...[/cyan]")

            # Create directed graph
            G = nx.DiGraph()

            # Add neurons as nodes
            neuron_ids = list(neurons.keys())
            for neuron_id in neuron_ids:
                neuron_data = neurons[neuron_id]
                # Node color based on activity
                output = neuron_data.get("output", 0)
                G.add_node(
                    neuron_id,
                    output=output,
                    potential=neuron_data.get("membrane_potential", 0),
                    firing_rate=neuron_data.get("firing_rate", 0),
                    node_type="neuron",
                )

            # Add external input synapses as small nodes
            external_nodes = []
            for ext_key, ext_data in external_inputs.items():
                if "_" in ext_key:
                    parts = ext_key.split("_")
                    if len(parts) >= 2:
                        try:
                            neuron_id = int(parts[0])
                            synapse_id = int(parts[1])
                            ext_node_id = f"ext_{neuron_id}_{synapse_id}"
                            G.add_node(
                                ext_node_id,
                                node_type="external",
                                info_signal=ext_data.get("info", 0),
                                target_neuron=neuron_id,
                                target_synapse=synapse_id,
                            )
                            external_nodes.append(ext_node_id)

                            # Add edge from external input to target neuron
                            G.add_edge(ext_node_id, neuron_id, edge_type="external")
                        except ValueError:
                            continue

            # Add connections as edges
            for connection in connections:
                source_id, _, target_id, _ = connection
                G.add_edge(source_id, target_id, edge_type="neuron")

            # Create plot
            plt.figure(figsize=(14, 10))
            plt.title(
                "Neural Network with Traveling Events", fontsize=16, fontweight="bold"
            )

            # Layout - position external nodes near their target neurons
            if len(neuron_ids) <= 20:
                pos = dict(nx.spring_layout(G, k=3, iterations=100))
            else:
                pos = dict(nx.spring_layout(G, k=2, iterations=50))

            # Adjust external node positions to be near their target neurons
            for ext_node in external_nodes:
                if ext_node in G.nodes():
                    target_neuron = G.nodes[ext_node]["target_neuron"]
                    if target_neuron in pos:
                        # Position external node near target neuron with small offset
                        offset_angle = np.random.uniform(0, 2 * np.pi)
                        offset_distance = 0.3
                        pos[ext_node] = (
                            pos[target_neuron][0]
                            + offset_distance * np.cos(offset_angle),
                            pos[target_neuron][1]
                            + offset_distance * np.sin(offset_angle),
                        )

            # Separate node types for different styling
            neuron_nodes = [
                n for n in G.nodes() if G.nodes[n].get("node_type") == "neuron"
            ]
            external_input_nodes = [
                n for n in G.nodes() if G.nodes[n].get("node_type") == "external"
            ]

            # Node colors based on activity for neurons
            neuron_colors = []
            for node_id in neuron_nodes:
                output = G.nodes[node_id]["output"]
                if output > 0:
                    neuron_colors.append("red")  # Active/firing
                else:
                    potential = G.nodes[node_id]["potential"]
                    if potential > 0.5:
                        neuron_colors.append("orange")  # High potential
                    elif potential > 0:
                        neuron_colors.append("yellow")  # Some potential
                    else:
                        neuron_colors.append("lightblue")  # Inactive

            # Draw neurons
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=neuron_nodes,
                node_color=neuron_colors,  # type: ignore
                node_size=600,
                alpha=0.8,
            )

            # Draw external input nodes (smaller, green)
            if external_input_nodes:
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=external_input_nodes,
                    node_color="lightgreen",
                    node_size=200,
                    alpha=0.7,
                    node_shape="s",
                )

            # Draw neuron labels
            neuron_labels = {n: str(n) for n in neuron_nodes}
            nx.draw_networkx_labels(
                G, pos, labels=neuron_labels, font_size=8, font_weight="bold"
            )

            # Draw external input labels
            if external_input_nodes:
                ext_labels = {n: "EXT" for n in external_input_nodes}
                nx.draw_networkx_labels(
                    G, pos, labels=ext_labels, font_size=6, font_color="darkgreen"
                )

            # Separate edges by type
            neuron_edges = [
                (u, v)
                for u, v, d in G.edges(data=True)
                if d.get("edge_type") == "neuron"
            ]
            external_edges = [
                (u, v)
                for u, v, d in G.edges(data=True)
                if d.get("edge_type") == "external"
            ]

            # Draw edges
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=neuron_edges,
                edge_color="gray",
                arrows=True,
                arrowsize=20,
                alpha=0.6,
                arrowstyle="->",
            )

            if external_edges:
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=external_edges,
                    edge_color="lightgreen",
                    arrows=True,
                    arrowsize=15,
                    alpha=0.5,
                    arrowstyle="->",
                )

            # Draw traveling events as dots on edges
            if traveling_events:
                event_positions_x = []
                event_positions_y = []
                event_colors = []
                event_sizes = []

                for event in traveling_events:
                    source_id = event.get("source_neuron")
                    target_id = event.get("target_neuron")

                    if not source_id or not target_id:
                        continue

                    if source_id not in pos or target_id not in pos:
                        continue

                    progress = max(
                        0,
                        min(
                            1,
                            (current_tick - event.get("arrival_tick", current_tick) + 5)
                            / 5,
                        ),
                    )

                    source_pos = pos[source_id]
                    target_pos = pos[target_id]

                    event_x = source_pos[0] + progress * (target_pos[0] - source_pos[0])
                    event_y = source_pos[1] + progress * (target_pos[1] - source_pos[1])

                    event_positions_x.append(event_x)
                    event_positions_y.append(event_y)

                    if event.get("event_type") == "PresynapticReleaseEvent":
                        event_colors.append("orange")
                        event_sizes.append(100)
                    elif event.get("event_type") == "RetrogradeSignalEvent":
                        event_colors.append("purple")
                        event_sizes.append(50)
                    else:
                        event_colors.append("blue")  # Default for unknown event types
                        event_sizes.append(75)

                if event_positions_x:
                    plt.scatter(
                        event_positions_x,
                        event_positions_y,
                        c=event_colors,
                        s=event_sizes,
                        alpha=0.8,
                        marker="o",
                        edgecolors="black",
                        linewidths=1,
                        zorder=5,
                    )

            # Add enhanced legend
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="red",
                    markersize=12,
                    label="Firing Neuron (Output > 0)",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="orange",
                    markersize=12,
                    label="High Potential (> 0.5)",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="yellow",
                    markersize=12,
                    label="Some Potential (> 0)",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="lightblue",
                    markersize=12,
                    label="Inactive Neuron",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor="lightgreen",
                    markersize=8,
                    label="External Input",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="orange",
                    markersize=8,
                    label="Presynaptic Event",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="purple",
                    markersize=8,
                    label="Retrograde Event",
                ),
            ]
            plt.legend(handles=legend_elements, loc="upper right", fontsize=10)

            # Network statistics
            num_neurons = len(neurons)
            num_connections = len(connections)
            num_external = len(external_inputs)
            num_events = len(traveling_events)
            synaptic_density = network_state.get("synaptic_density", 0.0)
            graph_density = network_state.get("graph_density", 0.0)
            active_neurons = sum(1 for n in neurons.values() if n.get("output", 0) > 0)

            stats_text = (
                f"Tick: {current_tick} | Neurons: {num_neurons} | Connections: {num_connections} | "
                f"External: {num_external} | Events: {num_events} | "
                f"Synaptic Density: {synaptic_density:.3f} | Graph Density: {graph_density:.3f} | Active: {active_neurons}"
            )
            plt.figtext(0.5, 0.02, stats_text, ha="center", fontsize=10)

            plt.axis("off")
            plt.tight_layout()

            # Save and show
            os.makedirs("plots", exist_ok=True)
            filename = f"plots/network_plot_tick_{current_tick}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            self.console.print(
                f"[green]Enhanced network plot saved as: {filename}[/green]"
            )
            self.console.print(
                f"[cyan]Showing {num_events} traveling events and {num_external} external inputs[/cyan]"
            )

            # Display plot
            plt.show()

        except ImportError as e:
            self.console.print(f"[red]Import error: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error creating plot: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

    @timed_command
    def cmd_detailed_state(self):
        """Show detailed state"""
        state = self.nn_core.get_network_state()
        if "error" in state:
            self.console.print(f"[red]Error: {state['error']}[/red]")
        else:
            core = state["core_state"]
            network = state["network"]

            # Core state table
            core_table = Table(title="Core State", show_header=True)
            core_table.add_column("Property", style="cyan")
            core_table.add_column("Value", style="white")

            core_table.add_row("Current Tick", str(core["current_tick"]))
            core_table.add_row("Is Running", str(core["is_running"]))
            core_table.add_row("Tick Rate", f"{core['tick_rate']:.2f} tps")

            self.console.print(core_table)

            # Network state table
            network_table = Table(title="Network State", show_header=True)
            network_table.add_column("Property", style="cyan")
            network_table.add_column("Value", style="white")

            network_table.add_row("Neurons", str(len(network["neurons"])))
            network_table.add_row("Connections", str(len(network["connections"])))

            self.console.print(network_table)

    @timed_command
    def cmd_status(self):
        """Show current status"""
        self.print_status()

    @timed_command
    def cmd_set_log_level(self):
        """Set log level"""
        try:
            from rich.prompt import Prompt

            # Available log levels
            valid_levels = [
                "TRACE",
                "DEBUG",
                "INFO",
                "SUCCESS",
                "WARNING",
                "ERROR",
                "CRITICAL",
            ]

            self.console.print("[cyan]Available log levels:[/cyan]")
            for i, level in enumerate(valid_levels):
                self.console.print(f"  {i+1}. {level}")

            choice = (
                Prompt.ask(
                    f"Select log level (1-{len(valid_levels)}) or enter level name",
                    default="INFO",
                )
                .strip()
                .upper()
            )

            # Handle numeric choice
            if choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(valid_levels):
                    selected_level = valid_levels[choice_num - 1]
                else:
                    self.console.print(
                        f"[red]Invalid choice. Must be 1-{len(valid_levels)}[/red]"
                    )
                    return
            # Handle level name
            elif choice in valid_levels:
                selected_level = choice
            else:
                self.console.print(f"[red]Invalid log level: {choice}[/red]")
                self.console.print(f"Valid levels: {', '.join(valid_levels)}[/red]")
                return

            # Use NNCore method to set log level
            if self.nn_core.set_log_level(selected_level):
                self.console.print(f"[green]Log level set to: {selected_level}[/green]")

                # Show guidance based on level
                if selected_level in ["TRACE", "DEBUG"]:
                    self.console.print(
                        "[dim]Note: Debug/trace messages will now be visible[/dim]"
                    )
                elif selected_level in ["ERROR", "CRITICAL"]:
                    self.console.print(
                        "[yellow]Note: Only error messages will be visible[/yellow]"
                    )
            else:
                self.console.print("[red]Failed to set log level[/red]")

        except Exception as e:
            self.console.print(f"[red]Error setting log level: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

    @timed_command
    def cmd_show_context(self):
        """Show current context"""
        if self.context_neuron_id is None and self.context_synapse_id is None:
            self.console.print("[yellow]No context set[/yellow]")
            return

        context_table = Table(title="Current Context", show_header=True)
        context_table.add_column("Type", style="cyan", no_wrap=True)
        context_table.add_column("ID", style="yellow", no_wrap=True)
        context_table.add_column("Status", style="green")

        if self.context_neuron_id is not None:
            # Check if neuron exists
            neuron_data = self.nn_core.get_neuron(self.context_neuron_id)
            status = "✓ Valid" if neuron_data else "✗ Not found"
            context_table.add_row("Neuron", str(self.context_neuron_id), status)

        if self.context_synapse_id is not None:
            # Check if synapse exists (requires neuron context)
            if self.context_neuron_id is not None:
                synapse_data = self.nn_core.get_synapse(
                    self.context_neuron_id, self.context_synapse_id
                )
                status = "✓ Valid" if synapse_data else "✗ Not found"
            else:
                status = "? Requires neuron context"
            context_table.add_row("Synapse", str(self.context_synapse_id), status)

        self.console.print(context_table)

        # Show usage tip
        if self.context_neuron_id is not None:
            self.console.print(
                "[dim]Commands will use this neuron ID when not specified[/dim]"
            )
        if self.context_synapse_id is not None:
            self.console.print(
                "[dim]Commands will use this synapse ID when not specified[/dim]"
            )

    @timed_command
    def cmd_set_neuron_context(self, params=None):
        """Set neuron context"""
        try:
            # Extract neuron_id from parameters if provided
            neuron_id = None
            if params and len(params) >= 1:
                try:
                    neuron_id = int(params[0])
                except ValueError:
                    self.console.print(f"[red]Invalid neuron ID: {params[0]}[/red]")
                    return

            # Prompt if not provided
            if neuron_id is None:
                neuron_id = IntPrompt.ask("Neuron ID to set as context")

            # Validate neuron exists
            neuron_data = self.nn_core.get_neuron(neuron_id)
            if not neuron_data:
                self.console.print(f"[red]Neuron {neuron_id} not found[/red]")
                return

            # Set context
            self.context_neuron_id = neuron_id
            # Clear synapse context when neuron context changes
            if self.context_synapse_id is not None:
                self.console.print(
                    "[yellow]Cleared synapse context (neuron changed)[/yellow]"
                )
                self.context_synapse_id = None

            self.console.print(f"[green]✓[/green] Set neuron context to {neuron_id}")

        except ValueError as e:
            self.console.print(f"[red]Invalid input: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error setting neuron context: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

    @timed_command
    def cmd_set_synapse_context(self, params=None):
        """Set synapse context"""
        try:
            # Extract synapse_id from parameters if provided
            synapse_id = None
            if params and len(params) >= 1:
                try:
                    synapse_id = int(params[0])
                except ValueError:
                    self.console.print(f"[red]Invalid synapse ID: {params[0]}[/red]")
                    return

            # Prompt if not provided
            if synapse_id is None:
                synapse_id = IntPrompt.ask("Synapse ID to set as context")

            # Validate synapse exists (requires neuron context)
            if self.context_neuron_id is None:
                self.console.print(
                    "[red]Neuron context must be set to validate synapse[/red]"
                )
                return

            synapse_data = self.nn_core.get_synapse(self.context_neuron_id, synapse_id)
            if not synapse_data:
                self.console.print(
                    f"[red]Synapse {synapse_id} not found in neuron {self.context_neuron_id}[/red]"
                )
                return

            # Set context
            self.context_synapse_id = synapse_id
            self.console.print(f"[green]✓[/green] Set synapse context to {synapse_id}")

        except ValueError as e:
            self.console.print(f"[red]Invalid input: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        except Exception as e:
            self.console.print(f"[red]Error setting synapse context: {e}[/red]")
            self.console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

    @timed_command
    def cmd_clear_context(self):
        """Clear all context"""
        self.context_neuron_id = None
        self.context_synapse_id = None
        self.console.print("[green]Context cleared[/green]")

    @timed_command
    def cmd_clear(self):
        """Clear screen"""
        clear()

    def cmd_exit(self):
        """Exit application"""
        self.running = False
        if self.nn_core.state.is_running:
            self.nn_core.stop_time_flow()
        self.console.print("[bold cyan]Goodbye![/bold cyan]")

    @timed_command
    def cmd_toggle_timing(self):
        """Toggle command timing display"""
        self.timing_enabled = not self.timing_enabled
        status = "enabled" if self.timing_enabled else "disabled"
        self.console.print(f"[green]Command timing is now {status}[/green]")

    @timed_command
    def cmd_list_free_outputs(self):
        """List free presynaptic terminals"""
        state = self.nn_core.get_network_state()
        if "error" in state:
            self.console.print("[red]No network loaded[/red]")
            return

        network = state["network"]
        neurons = network.get("neurons", {})
        connections = network.get("connections", [])

        if not neurons:
            self.console.print("[yellow]No neurons in the network[/yellow]")
            return

        # Create a set of used source terminals for quick lookup
        used_terminals = set()
        for conn in connections:
            if len(conn) >= 2:
                source_neuron, source_terminal = conn[0], conn[1]
                used_terminals.add(f"{source_neuron}_{source_terminal}")

        # Create table for free terminals
        free_table = Table(title="Free Presynaptic Terminals", show_header=True)
        free_table.add_column("Neuron ID", style="cyan", no_wrap=True)
        free_table.add_column("Terminal ID", style="yellow", no_wrap=True)

        total_free_terminals = 0
        for neuron_id, neuron_data in neurons.items():
            terminals = neuron_data.get("terminals", [])
            for terminal_id in terminals:
                terminal_key = f"{neuron_id}_{terminal_id}"
                if terminal_key not in used_terminals:
                    free_table.add_row(str(neuron_id), str(terminal_id))
                    total_free_terminals += 1

        if total_free_terminals > 0:
            self.console.print(free_table)
            self.console.print(
                f"\n[dim]Total free terminals: {total_free_terminals}[/dim]"
            )
        else:
            self.console.print("[yellow]No free presynaptic terminals found[/yellow]")


def main():
    """Main entry point"""

    def handle_sigint(sig, frame):
        """Handle Ctrl+C gracefully"""
        console = Console()
        console.print("\n[yellow]Interrupted. Type 'exit' or 'quit' to exit.[/yellow]")

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        cli = NeuralNetworkCLI()
        cli.run()
    except Exception as e:
        console = Console()
        console.print(f"[bold red]A critical error occurred: {e}[/bold red]")
        console.print(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
