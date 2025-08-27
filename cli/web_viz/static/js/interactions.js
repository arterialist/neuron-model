/**
 * User interaction handlers for the network visualization
 */

class InteractionHandler {
    constructor() {
        this.selectedNeuronId = null;
        this.detailsPanelVisible = false;

        this.setupEventHandlers();
    }

    setupEventHandlers() {
        // Control button handlers
        this.setupControlButtons();

        // Input handlers
        this.setupInputHandlers();

        // Network visualization handlers
        this.setupVisualizationHandlers();

        // Details panel handlers
        this.setupDetailsPanelHandlers();

        // Keyboard shortcuts
        this.setupKeyboardShortcuts();
    }

    setupControlButtons() {
        // Single tick button
        const tickBtn = document.getElementById('btn-tick');
        if (tickBtn) {
            tickBtn.addEventListener('click', async () => {
                try {
                    this.setStatusMessage('Executing tick...');
                    await window.dataManager.executeTick();
                    this.setStatusMessage('Tick executed');
                } catch (error) {
                    this.setStatusMessage(`Error: ${error.message}`, 'error');
                }
            });
        }

        // Multiple ticks button
        const multiTickBtn = document.getElementById('btn-multi-tick');
        if (multiTickBtn) {
            multiTickBtn.addEventListener('click', async () => {
                try {
                    const tickCount = parseInt(document.getElementById('tick-count').value) || 10;
                    this.setStatusMessage(`Executing ${tickCount} ticks...`);
                    await window.dataManager.executeMultipleTicks(tickCount);
                    this.setStatusMessage(`${tickCount} ticks executed`);
                } catch (error) {
                    this.setStatusMessage(`Error: ${error.message}`, 'error');
                }
            });
        }

        // Start auto button
        const startBtn = document.getElementById('btn-start');
        if (startBtn) {
            startBtn.addEventListener('click', async () => {
                try {
                    const tickRate = parseFloat(document.getElementById('tick-rate').value) || 1.0;
                    this.setStatusMessage(`Starting auto mode at ${tickRate} TPS...`);
                    await window.dataManager.startTimeFlow(tickRate);
                    this.setStatusMessage(`Auto mode started at ${tickRate} TPS`);

                    // Update button states
                    startBtn.disabled = true;
                    document.getElementById('btn-stop').disabled = false;
                } catch (error) {
                    this.setStatusMessage(`Error: ${error.message}`, 'error');
                }
            });
        }

        // Stop auto button
        const stopBtn = document.getElementById('btn-stop');
        if (stopBtn) {
            stopBtn.addEventListener('click', async () => {
                try {
                    this.setStatusMessage('Stopping auto mode...');
                    await window.dataManager.stopTimeFlow();
                    this.setStatusMessage('Auto mode stopped');

                    // Update button states
                    document.getElementById('btn-start').disabled = false;
                    stopBtn.disabled = true;
                } catch (error) {
                    this.setStatusMessage(`Error: ${error.message}`, 'error');
                }
            });
        }

        // Send signal button
        const signalBtn = document.getElementById('btn-send-signal');
        if (signalBtn) {
            signalBtn.addEventListener('click', async () => {
                try {
                    const neuronId = parseInt(document.getElementById('signal-neuron').value);
                    const synapseId = parseInt(document.getElementById('signal-synapse').value) || 0;
                    const strength = parseFloat(document.getElementById('signal-strength').value) || 1.5;

                    if (isNaN(neuronId)) {
                        this.setStatusMessage('Please enter a valid neuron ID', 'error');
                        return;
                    }

                    this.setStatusMessage(`Sending signal to neuron ${neuronId}...`);

                    // Send via WebSocket if connected, otherwise use REST API
                    if (window.wsClient.isConnected()) {
                        window.wsClient.send({
                            type: 'send_signal',
                            neuron_id: neuronId,
                            synapse_id: synapseId,
                            strength: strength
                        });
                    } else {
                        await window.dataManager.sendSignal(neuronId, synapseId, strength);
                    }

                    this.setStatusMessage(`Signal sent to neuron ${neuronId}`);

                    // Animate the signal if visualization is available
                    if (window.animationSystem && window.networkViz) {
                        window.animationSystem.animateSignalReceived(`neuron_${neuronId}`);
                    }
                } catch (error) {
                    this.setStatusMessage(`Error: ${error.message}`, 'error');
                }
            });
        }

        // Layout buttons
        const applyLayoutBtn = document.getElementById('btn-apply-layout');
        if (applyLayoutBtn) {
            applyLayoutBtn.addEventListener('click', async () => {
                try {
                    const layoutName = document.getElementById('layout-select').value;
                    this.setStatusMessage(`Applying ${layoutName} layout...`);
                    await window.networkViz.applyLayout(layoutName);
                    this.setStatusMessage(`${layoutName} layout applied`);
                } catch (error) {
                    this.setStatusMessage(`Error: ${error.message}`, 'error');
                }
            });
        }

        const fitViewBtn = document.getElementById('btn-fit-view');
        if (fitViewBtn) {
            fitViewBtn.addEventListener('click', () => {
                if (window.networkViz) {
                    window.networkViz.fitToView();
                    this.setStatusMessage('View fitted to network');
                }
            });
        }

        // Toggle layout button
        const toggleLayoutBtn = document.getElementById('btn-toggle-layout');
        if (toggleLayoutBtn) {
            toggleLayoutBtn.addEventListener('click', () => {
                if (window.networkViz) {
                    window.networkViz.toggleLayout();
                    const currentLayout = window.networkViz.layoutName;
                    this.setStatusMessage(`Switched to ${currentLayout} layout`);
                }
            });
        }
    }

    updateLayerInformation(networkState) {
        // Update the layer information panel with current network structure
        if (!networkState || !networkState.elements || !networkState.elements.nodes) {
            return;
        }

        const nodes = networkState.elements.nodes;
        const layers = {};

        // Count neurons by layer
        nodes.forEach(node => {
            if (node.data.type === 'neuron' && node.data.layer !== undefined) {
                const layer = node.data.layer;
                const layerName = node.data.layer_name || 'unknown';
                
                if (!layers[layer]) {
                    layers[layer] = {
                        name: layerName,
                        count: 0
                    };
                }
                layers[layer].count++;
            }
        });

        // Update layer counts
        const inputCount = layers[0] ? layers[0].count : 0;
        const outputCount = layers[Math.max(...Object.keys(layers).map(Number))] ? 
                           layers[Math.max(...Object.keys(layers).map(Number))].count : 0;
        const hiddenCount = Object.keys(layers).length - 2; // Subtract input and output
        const totalLayers = Object.keys(layers).length;

        // Update DOM elements
        const inputElement = document.getElementById('layer-input-count');
        const hiddenElement = document.getElementById('layer-hidden-count');
        const outputElement = document.getElementById('layer-output-count');
        const totalElement = document.getElementById('layer-total-count');

        if (inputElement) inputElement.textContent = inputCount;
        if (hiddenElement) hiddenElement.textContent = hiddenCount;
        if (outputElement) outputElement.textContent = outputCount;
        if (totalElement) totalElement.textContent = totalLayers;
    }

    setupInputHandlers() {
        // Auto-populate neuron ID from selected node
        const neuronInput = document.getElementById('signal-neuron');
        if (neuronInput) {
            neuronInput.addEventListener('focus', () => {
                if (this.selectedNeuronId && !neuronInput.value) {
                    neuronInput.value = this.selectedNeuronId;
                }
            });
        }

        // Enter key handlers for inputs
        const inputs = ['signal-neuron', 'signal-synapse', 'signal-strength', 'tick-count', 'tick-rate'];
        inputs.forEach(inputId => {
            const input = document.getElementById(inputId);
            if (input) {
                input.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        // Trigger appropriate action based on input
                        if (inputId.startsWith('signal-')) {
                            document.getElementById('btn-send-signal').click();
                        } else if (inputId === 'tick-count') {
                            document.getElementById('btn-multi-tick').click();
                        }
                    }
                });
            }
        });
    }

    setupVisualizationHandlers() {
        // Wait for network visualization to be initialized
        const checkVisualization = () => {
            if (window.networkViz) {
                // Node selection handler
                window.networkViz.on('node_selected', (node) => {
                    this.handleNodeSelected(node);
                });

                // Node deselection handler
                window.networkViz.on('node_deselected', (node) => {
                    this.handleNodeDeselected(node);
                });

                // Layout complete handler
                window.networkViz.on('layout_complete', (layout) => {
                    this.setStatusMessage('Layout complete');
                });
            } else {
                setTimeout(checkVisualization, 100);
            }
        };
        checkVisualization();
    }

    setupDetailsPanelHandlers() {
        const closeBtn = document.getElementById('btn-close-details');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                this.hideDetailsPanel();
            });
        }

        // Click outside to close
        document.addEventListener('click', (e) => {
            const detailsPanel = document.getElementById('neuron-details');
            if (this.detailsPanelVisible && detailsPanel &&
                !detailsPanel.contains(e.target) &&
                !e.target.closest('#cy')) {
                this.hideDetailsPanel();
            }
        });
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Don't trigger shortcuts when typing in inputs
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') {
                return;
            }

            switch (e.key) {
                case ' ': // Spacebar - single tick
                    e.preventDefault();
                    document.getElementById('btn-tick').click();
                    break;
                case 'Enter': // Enter - multiple ticks
                    e.preventDefault();
                    document.getElementById('btn-multi-tick').click();
                    break;
                case 's': // S - start auto
                    e.preventDefault();
                    document.getElementById('btn-start').click();
                    break;
                case 'x': // X - stop auto
                    e.preventDefault();
                    document.getElementById('btn-stop').click();
                    break;
                case 'f': // F - fit view
                    e.preventDefault();
                    document.getElementById('btn-fit-view').click();
                    break;
                case 'Escape': // Escape - close details panel
                    if (this.detailsPanelVisible) {
                        this.hideDetailsPanel();
                    }
                    break;
            }
        });
    }

    async handleNodeSelected(node) {
        const nodeData = node.data();

        if (nodeData.type === 'neuron') {
            this.selectedNeuronId = nodeData.neuron_id;

            // Show details panel
            await this.showNeuronDetails(nodeData.neuron_id);

            // Highlight connected nodes and edges
            this.highlightConnections(node);

            this.setStatusMessage(`Selected neuron ${nodeData.neuron_id}`);
        } else if (nodeData.type === 'external') {
            this.setStatusMessage(`Selected external input for neuron ${nodeData.target_neuron}`);
        }
    }

    handleNodeDeselected(node) {
        const nodeData = node.data();

        if (nodeData.type === 'neuron' && this.selectedNeuronId === nodeData.neuron_id) {
            this.selectedNeuronId = null;
        }

        // Clear highlights
        this.clearHighlights();

        this.setStatusMessage('Node deselected');
    }

    async showNeuronDetails(neuronId) {
        try {
            // Get detailed neuron information
            const neuronData = await window.dataManager.getNeuronDetails(neuronId);

            // Update details panel
            document.getElementById('detail-id').textContent = neuronId;
            document.getElementById('detail-potential').textContent =
                window.dataManager.formatNumber(neuronData.membrane_potential);
            document.getElementById('detail-firing-rate').textContent =
                window.dataManager.formatNumber(neuronData.firing_rate);
            document.getElementById('detail-output').textContent =
                window.dataManager.formatNumber(neuronData.output);
            document.getElementById('detail-synapses').textContent =
                neuronData.synapses ? neuronData.synapses.length : 0;
            document.getElementById('detail-terminals').textContent =
                neuronData.terminals ? neuronData.terminals.length : 0;

            // Show the panel
            const detailsPanel = document.getElementById('neuron-details');
            if (detailsPanel) {
                detailsPanel.classList.remove('hidden');
                this.detailsPanelVisible = true;
            }
        } catch (error) {
            console.error('Failed to load neuron details:', error);
            this.setStatusMessage(`Error loading neuron details: ${error.message}`, 'error');
        }
    }

    hideDetailsPanel() {
        const detailsPanel = document.getElementById('neuron-details');
        if (detailsPanel) {
            detailsPanel.classList.add('hidden');
            this.detailsPanelVisible = false;
        }
    }

    highlightConnections(node) {
        if (!window.networkViz) return;

        // Clear previous highlights
        this.clearHighlights();

        // Highlight the selected node
        node.addClass('highlighted');

        // Highlight connected edges and nodes
        const connectedEdges = node.connectedEdges();
        const connectedNodes = node.neighborhood();

        connectedEdges.addClass('highlighted');
        connectedNodes.addClass('connected');
    }

    clearHighlights() {
        if (window.networkViz && window.networkViz.cy) {
            window.networkViz.cy.elements().removeClass('highlighted connected');
        }
    }

    updateStatistics(stats) {
        // Update statistics display
        const statElements = {
            'stat-tick': stats.current_tick || 0,
            'stat-neurons': stats.num_neurons || 0,
            'stat-active': stats.active_neurons || 0,
            'stat-connections': stats.num_connections || 0,
            'stat-external': stats.num_external_inputs || 0,
            'stat-signals': stats.num_traveling_signals || 0,
            'stat-avg-potential': window.dataManager.formatNumber(stats.avg_potential),
            'stat-max-potential': window.dataManager.formatNumber(stats.max_potential)
        };

        Object.entries(statElements).forEach(([elementId, value]) => {
            const element = document.getElementById(elementId);
            if (element) {
                element.textContent = value;
            }
        });
    }

    setStatusMessage(message, type = 'info') {
        const statusElement = document.getElementById('status-message');
        if (statusElement) {
            statusElement.textContent = message;
            statusElement.className = `status-${type}`;

            // Clear status after 5 seconds for non-error messages
            if (type !== 'error') {
                setTimeout(() => {
                    if (statusElement.textContent === message) {
                        statusElement.textContent = 'Ready';
                        statusElement.className = '';
                    }
                }, 5000);
            }
        }
    }

    // Public methods for external control
    selectNeuron(neuronId) {
        if (window.networkViz && window.networkViz.cy) {
            const node = window.networkViz.cy.getElementById(`neuron_${neuronId}`);
            if (node.length > 0) {
                window.networkViz.cy.elements().unselect();
                node.select();
                window.networkViz.centerOnNode(`neuron_${neuronId}`);
            }
        }
    }

    flashNeuron(neuronId, duration = 500) {
        if (window.animationSystem) {
            window.animationSystem.createNodeFlash(`neuron_${neuronId}`, '#ffff00', duration);
        }
    }

    showSignalAnimation(sourceId, targetId, eventType) {
        if (window.animationSystem) {
            window.animationSystem.animateSignal(`neuron_${sourceId}`, `neuron_${targetId}`, eventType);
        }
    }
}

// Global interaction handler instance
window.interactionHandler = new InteractionHandler();