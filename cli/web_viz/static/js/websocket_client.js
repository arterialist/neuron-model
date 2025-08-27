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
            // Connect to native WebSocket server
            this.socket = new WebSocket('ws://127.0.0.1:5556');
            this.setupEventHandlers();

        } catch (error) {
            console.error('Failed to establish WebSocket connection:', error);
            this.updateConnectionStatus(false);
        }
    }

    setupEventHandlers() {
        // Connection events
        this.socket.onopen = () => {
            console.log('WebSocket connected');
            this.connected = true;
            this.reconnectAttempts = 0;
            this.reconnectDelay = 1000;
            this.updateConnectionStatus(true);
            this.emit('connect');
        };

        this.socket.onclose = (event) => {
            console.log('WebSocket disconnected:', event.code, event.reason);
            this.connected = false;
            this.updateConnectionStatus(false);
            this.emit('disconnect', event.reason);

            // Attempt to reconnect if not manually disconnected
            if (event.code !== 1000) { // 1000 = normal closure
                this.scheduleReconnect();
            }
        };

        this.socket.onerror = (error) => {
            console.error('WebSocket connection error:', error);
            this.connected = false;
            this.updateConnectionStatus(false);
            this.scheduleReconnect();
        };

        // Message handling
        this.socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };
    }

    handleMessage(data) {
        const messageType = data.type;
        
        switch (messageType) {
            case 'connected':
                this.emit('connected', data);
                break;
            case 'network_update':
                this.emit('network_update', data.data);
                break;
            case 'network_state':
                this.emit('network_state', data.data);
                break;
            case 'signal_result':
                this.emit('signal_result', data);
                break;
            case 'tick_result':
                this.emit('tick_result', data);
                break;
            case 'error':
                console.error('Server error:', data.message);
                this.emit('error', data);
                break;
            default:
                console.log('Unknown message type:', messageType, data);
        }
    }

    scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            return;
        }

        const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts), this.maxReconnectDelay);
        console.log(`Scheduling reconnection attempt ${this.reconnectAttempts + 1} in ${delay}ms`);

        setTimeout(() => {
            this.reconnectAttempts++;
            this.connect();
        }, delay);
    }

    disconnect() {
        if (this.socket) {
            this.socket.close(1000, 'Client disconnect'); // Normal closure
        }
    }

    send(data) {
        if (this.socket && this.connected) {
            try {
                this.socket.send(JSON.stringify(data));
            } catch (error) {
                console.error('Error sending message:', error);
            }
        } else {
            console.warn('Cannot send message: WebSocket not connected');
        }
    }

    // Event handling methods
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

    updateConnectionStatus(connected) {
        this.connected = connected;
        // Update UI if status element exists
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            statusElement.textContent = connected ? 'Connected' : 'Disconnected';
            statusElement.className = connected ? 'status-connected' : 'status-disconnected';
        }
    }

    isConnected() {
        return this.connected;
    }
}
window.wsClient = new WebSocketClient();