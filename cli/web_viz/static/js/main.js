/**
 * Main application initialization and coordination
 */

class NeuralNetworkVisualizationApp {
    constructor() {
        this.initialized = false;
        this.loadingOverlay = null;
    }

    async initialize() {
        try {
            this.showLoading('Initializing application...');

            // Initialize data manager
            this.showLoading('Loading network data...');
            await window.dataManager.initialize();

            // Initialize network visualization
            this.showLoading('Setting up visualization...');
            window.networkViz = new NetworkVisualization('cy');
            await window.networkViz.initialize();

            // Setup event handlers first
            this.setupEventHandlers();

            // Initialize WebSocket connection after event handlers are set up
            this.showLoading('Connecting to server...');
            window.wsClient.connect();

            // Start animation system
            window.animationSystem.start();

            this.hideLoading();
            this.initialized = true;

            console.log('Neural Network Visualization App initialized successfully');
            if (window.interactionHandler) {
                window.interactionHandler.setStatusMessage('Application ready');
            } else {
                // Wait for interaction handler to be ready
                this.waitForInteractionHandler();
            }

        } catch (error) {
            console.error('Failed to initialize application:', error);
            this.showError(`Initialization failed: ${error.message}`);
        }
    }

    waitForInteractionHandler() {
        // Wait for interaction handler to be available and then set status message
        const checkHandler = () => {
            if (window.interactionHandler) {
                window.interactionHandler.setStatusMessage('Application ready');
            } else {
                setTimeout(checkHandler, 100);
            }
        };
        checkHandler();
    }

    setupEventHandlers() {
        // Data manager events
        window.dataManager.on('state_updated', (state) => {
            this.handleNetworkStateUpdate(state, 'internal');
            // Update layer information
            if (window.interactionHandler) {
                window.interactionHandler.updateLayerInformation(state);
            }
        });

        // Network visualization events
        if (window.networkViz) {
            window.networkViz.on('update_complete', () => {
                // Handle update completion
                console.log('Network update completed');
            });
        }

        window.dataManager.on('error', (error) => {
            console.error('Data manager error:', error);
            if (window.interactionHandler) {
                window.interactionHandler.setStatusMessage(`Data error: ${error.message}`, 'error');
            }
        });

        // WebSocket events
        window.wsClient.on('connect', () => {
            console.log('WebSocket connected');
            if (window.interactionHandler) {
                window.interactionHandler.setStatusMessage('Connected to server');
            }
        });

        window.wsClient.on('disconnect', (reason) => {
            console.log('WebSocket disconnected:', reason);
            if (window.interactionHandler) {
                window.interactionHandler.setStatusMessage('Disconnected from server', 'error');
            }
        });

        window.wsClient.on('network_state', (state) => {
            this.handleNetworkStateUpdate(state, 'external');
        });

        window.wsClient.on('network_update', (state) => {
            this.handleNetworkStateUpdate(state, 'external');
        });

        window.wsClient.on('error', (data) => {
            console.error('WebSocket error:', data);
            if (window.interactionHandler) {
                window.interactionHandler.setStatusMessage(`Server error: ${data.message}`, 'error');
            }
        });

        window.wsClient.on('signal_result', (data) => {
            if (data.success) {
                if (window.interactionHandler) {
                    window.interactionHandler.setStatusMessage(
                        `Signal sent to neuron ${data.neuron_id}`
                    );
                }

                // Animate the signal
                if (window.animationSystem) {
                    window.animationSystem.animateSignalReceived(`neuron_${data.neuron_id}`);
                }
            } else {
                if (window.interactionHandler) {
                    window.interactionHandler.setStatusMessage('Failed to send signal', 'error');
                }
            }
        });

        window.wsClient.on('tick_result', (result) => {
            if (result.error) {
                if (window.interactionHandler) {
                    window.interactionHandler.setStatusMessage(`Tick error: ${result.error}`, 'error');
                }
            } else {
                const tick = result.tick || 'N/A';
                const activity = result.total_activity || 0;
                if (window.interactionHandler) {
                    window.interactionHandler.setStatusMessage(
                        `Tick ${tick} completed${activity > 0 ? ` (activity: ${activity.toFixed(3)})` : ''}`
                    );
                }
            }
        });

        // Window events
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });

        // Handle visibility change to pause/resume when tab is not visible
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                // Tab is hidden, potentially pause updates
                if (window.animationSystem) {
                    window.animationSystem.stop();
                }
            } else {
                // Tab is visible, resume updates
                if (window.animationSystem) {
                    window.animationSystem.start();
                }
            }
        });
    }

    handleNetworkStateUpdate(state, source = 'external') {
        try {
            if (!state || state.error) {
                console.warn('Received invalid network state:', state);
                return;
            }

            // Only update data manager if this is an external update (WebSocket)
            // to prevent circular dependency with the data manager's state_updated event
            if (source === 'external') {
                window.dataManager.updateNetworkState(state);
            }

            // Update visualization
            if (window.networkViz) {
                window.networkViz.updateNetworkData(state);
            }

            // Update statistics display
            if (window.interactionHandler && state.statistics) {
                window.interactionHandler.updateStatistics(state.statistics);
            }

            // Process traveling signals for animation
            if (window.animationSystem && state.traveling_signals) {
                this.processTravelingSignals(state.traveling_signals);
            }

            // Update neuron details panel if open
            if (window.interactionHandler && 
                window.interactionHandler.detailsPanelVisible &&
                window.interactionHandler.selectedNeuronId) {
                this.updateNeuronDetailsIfOpen(state);
            }



        } catch (error) {
            console.error('Error handling network state update:', error);
        }
    }

    processTravelingSignals(signals) {
        // Process new traveling signals for animation
        signals.forEach(signal => {
            if (signal.progress < 1) { // Only animate signals that haven't reached their target
                // Check if we should create a new animation for this signal
                const shouldAnimate = this.shouldAnimateSignal(signal);
                if (shouldAnimate) {
                    window.animationSystem.animateSignal(
                        signal.source.replace('neuron_', ''),
                        signal.target.replace('neuron_', ''),
                        signal.event_type
                    );
                }
            }
        });
    }

    shouldAnimateSignal(signal) {
        // Simple logic to avoid duplicate animations
        // In a more sophisticated implementation, you might track signal IDs
        return Math.random() < 0.3; // Randomly animate some signals to avoid overwhelming the UI
    }

    updateNeuronDetailsIfOpen(state) {
        // Update the details panel with current neuron data if it's open
        if (!window.interactionHandler) {
            return;
        }
        
        const neuronId = window.interactionHandler.selectedNeuronId;
        if (!neuronId || !state.elements || !state.elements.nodes) {
            return;
        }

        const neuronNode = state.elements.nodes.find(node =>
            node.data.neuron_id === neuronId
        );

        if (neuronNode) {
            const data = neuronNode.data;
            document.getElementById('detail-potential').textContent =
                window.dataManager.formatNumber(data.membrane_potential);
            document.getElementById('detail-firing-rate').textContent =
                window.dataManager.formatNumber(data.firing_rate);
            document.getElementById('detail-output').textContent =
                window.dataManager.formatNumber(data.output);
        }
    }

    showLoading(message = 'Loading...') {
        this.loadingOverlay = document.getElementById('loading-overlay');
        const loadingText = document.querySelector('.loading-text');

        if (this.loadingOverlay) {
            this.loadingOverlay.classList.remove('hidden');
        }

        if (loadingText) {
            loadingText.textContent = message;
        }
    }

    hideLoading() {
        if (this.loadingOverlay) {
            this.loadingOverlay.classList.add('hidden');
        }
    }

    showError(message) {
        this.hideLoading();

        // Create error overlay
        const errorOverlay = document.createElement('div');
        errorOverlay.className = 'error-overlay';
        errorOverlay.innerHTML = `
            <div class="error-content">
                <h2>Initialization Error</h2>
                <p>${message}</p>
                <button onclick="location.reload()" class="btn btn-primary">Reload Page</button>
            </div>
        `;

        // Add error styles
        errorOverlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 10000;
            color: white;
            text-align: center;
        `;

        const errorContent = errorOverlay.querySelector('.error-content');
        errorContent.style.cssText = `
            background-color: #2c3e50;
            padding: 2rem;
            border-radius: 8px;
            max-width: 500px;
            width: 90%;
        `;

        document.body.appendChild(errorOverlay);
    }

    cleanup() {
        try {
            // Stop animation system
            if (window.animationSystem) {
                window.animationSystem.stop();
            }

            // Disconnect WebSocket
            if (window.wsClient) {
                window.wsClient.disconnect();
            }

            // Destroy visualization
            if (window.networkViz) {
                window.networkViz.destroy();
            }

            console.log('Application cleanup completed');
        } catch (error) {
            console.error('Error during cleanup:', error);
        }
    }

    // Public methods for external control
    refresh() {
        if (window.dataManager) {
            window.dataManager.loadNetworkState();
        }
    }
    


    exportVisualization() {
        if (window.networkViz) {
            const blob = window.networkViz.exportImage('png', 2);
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `neural_network_${Date.now()}.png`;
            a.click();
            URL.revokeObjectURL(url);
        }
    }

    getAppInfo() {
        return {
            initialized: this.initialized,
            connected: window.wsClient ? window.wsClient.isConnected() : false,
            networkState: window.dataManager ? window.dataManager.getCurrentState() : null,
            selectedNeuron: window.interactionHandler ? window.interactionHandler.selectedNeuronId : null
        };
    }
}

// Global app instance
window.neuralNetApp = new NeuralNetworkVisualizationApp();

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing Neural Network Visualization App...');
    window.neuralNetApp.initialize();
});

// Add global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    if (window.interactionHandler && event.error && event.error.message) {
        window.interactionHandler.setStatusMessage(`Error: ${event.error.message}`, 'error');
    }
    
    // Special handling for CSS color errors
    if (event.message && event.message.includes('background-color') && event.message.includes('NaN')) {
        console.error('CSS color error detected:', event.message);
    }
});

// Add unhandled promise rejection handler
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    if (window.interactionHandler && event.reason) {
        const message = typeof event.reason === 'string' ? event.reason : 'Promise rejection';
        window.interactionHandler.setStatusMessage(`Promise error: ${message}`, 'error');
    }
});

// Expose useful functions to global scope for debugging
window.debugViz = {
    refresh: () => window.neuralNetApp.refresh(),
    export: () => window.neuralNetApp.exportVisualization(),
    info: () => window.neuralNetApp.getAppInfo(),
    selectNeuron: (id) => window.interactionHandler ? window.interactionHandler.selectNeuron(id) : null,
    flashNeuron: (id) => window.interactionHandler ? window.interactionHandler.flashNeuron(id) : null,
    fitView: () => window.networkViz ? window.networkViz.fitToView() : null
};