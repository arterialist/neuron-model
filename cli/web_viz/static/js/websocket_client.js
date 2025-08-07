/**
 * WebSocket client for real-time neural network updates
 */

class WebSocketClient {
    constructor() {
        this.socket = null;
        this.connected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // Start with 1 second
        this.maxReconnectDelay = 30000; // Max 30 seconds

        this.eventHandlers = {
            'connect': [],
            'disconnect': [],
            'network_state': [],
            'network_update': [],
            'error': [],
            'signal_result': [],
            'tick_result': []
        };
    }

    connect() {
        try {
            // Initialize Socket.IO connection
            this.socket = io({
                transports: ['websocket', 'polling'],
                upgrade: true,
                rememberUpgrade: true
            });

            this.setupEventHandlers();

        } catch (error) {
            console.error('Failed to establish WebSocket connection:', error);
            this.updateConnectionStatus(false);
        }
    }

    setupEventHandlers() {
        // Connection events
        this.socket.on('connect', () => {
            console.log('WebSocket connected');
            this.connected = true;
            this.reconnectAttempts = 0;
            this.reconnectDelay = 1000;
            this.updateConnectionStatus(true);
            this.emit('connect');
        });

        this.socket.on('disconnect', (reason) => {
            console.log('WebSocket disconnected:', reason);
            this.connected = false;
            this.updateConnectionStatus(false);
            this.emit('disconnect', reason);

            // Attempt to reconnect if not manually disconnected
            if (reason !== 'io client disconnect') {
                this.scheduleReconnect();
            }
        });

        this.socket.on('connect_error', (error) => {
            console.error('WebSocket connection error:', error);
            this.connected = false;
            this.updateConnectionStatus(false);
            this.scheduleReconnect();
        });

        // Data events
        this.socket.on('network_state', (data) => {
            this.emit('network_state', data);
        });

        this.socket.on('network_update', (data) => {
            this.emit('network_update', data);
        });

        this.socket.on('error', (data) => {
            console.error('Server error:', data.message);
            this.emit('error', data);
        });

        this.socket.on('signal_result', (data) => {
            this.emit('signal_result', data);
        });

        this.socket.on('tick_result', (data) => {
            this.emit('tick_result', data);
        });
    }

    scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            return;
        }

        this.reconnectAttempts++;
        const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), this.maxReconnectDelay);

        console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

        setTimeout(() => {
            if (!this.connected) {
                this.connect();
            }
        }, delay);
    }

    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            statusElement.textContent = connected ? 'Connected' : 'Disconnected';
            statusElement.className = connected ? 'status-connected' : 'status-disconnected';
        }
    }

    // Event system
    on(event, handler) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].push(handler);
        }
    }

    off(event, handler) {
        if (this.eventHandlers[event]) {
            const index = this.eventHandlers[event].indexOf(handler);
            if (index > -1) {
                this.eventHandlers[event].splice(index, 1);
            }
        }
    }

    emit(event, data) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Error in event handler for ${event}:`, error);
                }
            });
        }
    }

    // Send methods
    sendSignal(neuronId, synapseId, strength) {
        if (this.connected && this.socket) {
            this.socket.emit('send_signal', {
                neuron_id: neuronId,
                synapse_id: synapseId,
                strength: strength
            });
        } else {
            console.warn('Cannot send signal: WebSocket not connected');
        }
    }

    executeTick() {
        if (this.connected && this.socket) {
            this.socket.emit('execute_tick');
        } else {
            console.warn('Cannot execute tick: WebSocket not connected');
        }
    }

    requestNetworkState() {
        if (this.connected && this.socket) {
            this.socket.emit('get_network_state');
        } else {
            console.warn('Cannot request network state: WebSocket not connected');
        }
    }

    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
        }
        this.connected = false;
        this.updateConnectionStatus(false);
    }

    isConnected() {
        return this.connected;
    }
}

// Global WebSocket client instance
window.wsClient = new WebSocketClient();