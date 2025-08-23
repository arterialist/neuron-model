/**
 * Data manager for handling network state and API communication
 */

class DataManager {
    constructor() {
        this.currentNetworkState = null;
        this.networkStyle = null;
        this.layoutConfigs = {};
        this.apiBase = '/api';
        
        // Performance optimization: throttle rapid updates
        this.updateThrottle = null;
        this.lastUpdateTime = 0;
        this.minUpdateInterval = 100; // Minimum 100ms between updates for stability
        
        // Prevent recursive updates
        this.isUpdating = false;

        this.eventHandlers = {
            'state_updated': [],
            'error': []
        };
    }

    // Event system
    on(event, handler) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].push(handler);
        }
    }

    emit(event, data) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Error in data manager event handler for ${event}:`, error);
                }
            });
        }
    }

    async initialize() {
        try {
            // Load initial network style
            await this.loadNetworkStyle();

            // Load initial network state
            await this.loadNetworkState();

            console.log('Data manager initialized');
        } catch (error) {
            console.error('Failed to initialize data manager:', error);
            this.emit('error', error);
        }
    }

    async loadNetworkState() {
        try {
            const response = await fetch(`${this.apiBase}/network/state`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            this.currentNetworkState = data;
            
            // Only emit if not currently updating to prevent circular dependency
            if (!this.isUpdating) {
                this.emit('state_updated', data);
            }

            return data;
        } catch (error) {
            console.error('Failed to load network state:', error);
            this.emit('error', error);
            throw error;
        }
    }

    async loadNetworkStyle() {
        try {
            const response = await fetch(`${this.apiBase}/network/style`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const style = await response.json();
            this.networkStyle = style;

            return style;
        } catch (error) {
            console.error('Failed to load network style:', error);
            this.emit('error', error);
            throw error;
        }
    }

    async loadLayoutConfig(layoutName) {
        if (this.layoutConfigs[layoutName]) {
            return this.layoutConfigs[layoutName];
        }

        try {
            const response = await fetch(`${this.apiBase}/network/layout/${layoutName}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const config = await response.json();
            this.layoutConfigs[layoutName] = config;

            return config;
        } catch (error) {
            console.error(`Failed to load layout config for ${layoutName}:`, error);
            this.emit('error', error);
            throw error;
        }
    }

    async sendSignal(neuronId, synapseId, strength) {
        try {
            const response = await fetch(`${this.apiBase}/network/signal`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    neuron_id: neuronId,
                    synapse_id: synapseId,
                    strength: strength
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            if (result.error) {
                throw new Error(result.error);
            }

            return result;
        } catch (error) {
            console.error('Failed to send signal:', error);
            this.emit('error', error);
            throw error;
        }
    }

    async executeTick() {
        try {
            const response = await fetch(`${this.apiBase}/network/tick`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            if (result.error) {
                throw new Error(result.error);
            }

            return result;
        } catch (error) {
            console.error('Failed to execute tick:', error);
            this.emit('error', error);
            throw error;
        }
    }

    async executeMultipleTicks(nTicks) {
        try {
            const response = await fetch(`${this.apiBase}/network/ticks`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    n_ticks: nTicks
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            if (result.error) {
                throw new Error(result.error);
            }

            return result;
        } catch (error) {
            console.error('Failed to execute multiple ticks:', error);
            this.emit('error', error);
            throw error;
        }
    }

    async startTimeFlow(tickRate) {
        try {
            const response = await fetch(`${this.apiBase}/network/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    tick_rate: tickRate
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            if (result.error) {
                throw new Error(result.error);
            }

            return result;
        } catch (error) {
            console.error('Failed to start time flow:', error);
            this.emit('error', error);
            throw error;
        }
    }

    async stopTimeFlow() {
        try {
            const response = await fetch(`${this.apiBase}/network/stop`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            if (result.error) {
                throw new Error(result.error);
            }

            return result;
        } catch (error) {
            console.error('Failed to stop time flow:', error);
            this.emit('error', error);
            throw error;
        }
    }

    async getNeuronDetails(neuronId) {
        try {
            const response = await fetch(`${this.apiBase}/neuron/${neuronId}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            return data;
        } catch (error) {
            console.error(`Failed to get neuron details for ${neuronId}:`, error);
            this.emit('error', error);
            throw error;
        }
    }

    updateNetworkState(newState) {
        // Prevent recursive updates
        if (this.isUpdating) {
            console.warn('Recursive update detected, skipping to prevent infinite loop');
            return;
        }
        
        // Prevent updating with the same state to avoid unnecessary events
        // Check if the new state is actually different from the current one
        if (this.currentNetworkState && 
            this.currentNetworkState.current_tick === newState.current_tick &&
            this.currentNetworkState.is_running === newState.is_running) {
            // Only update if there are actual changes in the network state
            return;
        }
        
        this.isUpdating = true;
        try {
            this.currentNetworkState = newState;
            
            // Simple throttling to prevent overwhelming the UI
            const now = Date.now();
            if (now - this.lastUpdateTime >= this.minUpdateInterval) {
                this.emit('state_updated', newState);
                this.lastUpdateTime = now;
            }
            // If throttled, just skip the update rather than scheduling a delayed one
        } finally {
            this.isUpdating = false;
        }
    }

    getCurrentState() {
        return this.currentNetworkState;
    }

    getNetworkStyle() {
        return this.networkStyle;
    }

    // Utility methods for extracting data
    getNetworkElements() {
        if (!this.currentNetworkState || !this.currentNetworkState.elements) {
            return { nodes: [], edges: [] };
        }
        return this.currentNetworkState.elements;
    }

    getTravelingSignals() {
        if (!this.currentNetworkState) {
            return [];
        }
        return this.currentNetworkState.traveling_signals || [];
    }

    getStatistics() {
        if (!this.currentNetworkState) {
            return {};
        }
        return this.currentNetworkState.statistics || {};
    }

    // Helper method to format numbers for display
    formatNumber(value, decimals = 3) {
        if (typeof value !== 'number') {
            return '-';
        }
        return value.toFixed(decimals);
    }

    // Helper method to extract neuron ID from Cytoscape node ID
    extractNeuronId(nodeId) {
        if (typeof nodeId === 'string' && nodeId.startsWith('neuron_')) {
            return parseInt(nodeId.substring(7));
        }
        return null;
    }
}

// Global data manager instance
window.dataManager = new DataManager();