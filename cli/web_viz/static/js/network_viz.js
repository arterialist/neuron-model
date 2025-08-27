/**
 * Main network visualization using Cytoscape.js
 */

class NetworkVisualization {
    constructor(containerId) {
        this.containerId = containerId;
        this.cy = null;
        this.currentLayout = null;
        this.layoutName = 'layers'; // Default to layer-based layout

        // Animation state
        this.animationFrame = null;
        this.lastFrameTime = 0;
        this.fps = 0;
        this.frameCount = 0;
        this.fpsUpdateTime = 0;

        this.eventHandlers = {
            'node_selected': [],
            'node_deselected': [],
            'layout_complete': []
        };
        
        // Performance optimization properties
        this.updateAnimationFrame = null;
        this.updateQueue = [];
        this.isUpdating = false;
    }

    async initialize() {
        try {
            // Wait for data manager to be ready
            if (!window.dataManager.getNetworkStyle()) {
                await window.dataManager.initialize();
            }

            // Initialize Cytoscape
            this.initializeCytoscape();

            // Setup event handlers
            this.setupEventHandlers();

            // Load initial data
            await this.loadNetworkData();

            // Apply layer-based layout by default
            this.applyLayerLayout();

            // Start animation loop
            this.startAnimationLoop();

            console.log('Network visualization initialized');
        } catch (error) {
            console.error('Failed to initialize network visualization:', error);
            throw error;
        }
    }

    initializeCytoscape() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            throw new Error(`Container element with ID '${this.containerId}' not found`);
        }

        // Get style from data manager
        const style = window.dataManager.getNetworkStyle();

        this.cy = cytoscape({
            container: container,
            style: style,
            layout: { name: 'preset' }, // We'll apply layout manually

            // Interaction options
            userZoomingEnabled: true,
            userPanningEnabled: true,
            boxSelectionEnabled: true,
            selectionType: 'single',

            // Performance options - optimized for smooth updates
            textureOnViewport: true,
            motionBlur: false, // Disable for better performance
            wheelSensitivity: 0.5,
            autoungrabify: false, // Better for performance
            autolock: false, // Better for performance

            // Styling options
            hideEdgesOnViewport: true, // Hide edges when zoomed out for performance
            hideLabelsOnViewport: true, // Hide labels when zoomed out for performance
            pixelRatio: 'auto',
            
            // Additional performance options
            autoungrabify: false,
            autolock: false,
            autounselectify: false
        });

        console.log('Cytoscape instance created');
    }

    setupEventHandlers() {
        // Node selection events
        this.cy.on('select', 'node', (evt) => {
            const node = evt.target;
            this.emit('node_selected', node);
        });

        this.cy.on('unselect', 'node', (evt) => {
            const node = evt.target;
            this.emit('node_deselected', node);
        });

        // Layout events
        this.cy.on('layoutstop', (evt) => {
            this.emit('layout_complete', evt.layout);
        });

        // Mouse events for hover effects
        this.cy.on('mouseover', 'node', (evt) => {
            const node = evt.target;
            node.addClass('hovered');
        });

        this.cy.on('mouseout', 'node', (evt) => {
            const node = evt.target;
            node.removeClass('hovered');
        });

        // Double-click to fit view
        this.cy.on('dblclick', (evt) => {
            if (evt.target === this.cy) {
                this.fitToView();
            }
        });
    }

        async loadNetworkData() {
        try {
            const state = window.dataManager.getCurrentState();
            if (!state || !state.elements) {
                console.warn('No network data available');
                return;
            }

            // Clear existing elements
            this.cy.elements().remove();
            
            // Add new elements safely
            if (state.elements.nodes && state.elements.nodes.length > 0) {
                this.cy.add(state.elements.nodes);
            }
            if (state.elements.edges && state.elements.edges.length > 0) {
                this.cy.add(state.elements.edges);
            }
            
            // Apply layer-based layout by default
            this.applyLayerLayout();
            
            console.log(`Loaded ${state.elements.nodes?.length || 0} nodes and ${state.elements.edges?.length || 0} edges`);
        } catch (error) {
            console.error('Failed to load network data:', error);
            throw error;
        }
    }

    async updateNetworkData(state) {
        try {
            if (!state || !state.elements) {
                return;
            }

            // Use requestAnimationFrame for smooth updates
            if (this.updateAnimationFrame) {
                cancelAnimationFrame(this.updateAnimationFrame);
            }

            this.updateAnimationFrame = requestAnimationFrame(() => {
                this._performBatchedUpdate(state);
            });

        } catch (error) {
            console.error('Failed to update network data:', error);
        }
    }

    _performBatchedUpdate(state) {
        try {
            // Batch size for performance
            const BATCH_SIZE = 50;
            const UPDATE_DELAY = 16; // ~60 FPS

            // Prepare update batches
            const nodeUpdates = this._prepareNodeUpdates(state.elements.nodes);
            const edgeUpdates = this._prepareEdgeUpdates(state.elements.edges);
            const removals = this._prepareRemovals(state);

            // Process updates in batches to prevent blocking
            this._processBatchedUpdates(nodeUpdates, edgeUpdates, removals, BATCH_SIZE, UPDATE_DELAY);

        } catch (error) {
            console.error('Failed to perform batched update:', error);
        }
    }

    _prepareNodeUpdates(nodes) {
        const updates = [];
        const additions = [];

        nodes.forEach(nodeData => {
            const node = this.cy.getElementById(nodeData.data.id);
            if (node.length > 0) {
                updates.push({ node, data: nodeData });
            } else {
                additions.push(nodeData);
            }
        });

        return { updates, additions };
    }

    _prepareEdgeUpdates(edges) {
        const updates = [];
        const additions = [];

        edges.forEach(edgeData => {
            const edge = this.cy.getElementById(edgeData.data.id);
            if (edge.length > 0) {
                updates.push({ edge, data: edgeData });
            } else {
                additions.push(edgeData);
            }
        });

        return { updates, additions };
    }

    _prepareRemovals(state) {
        const currentNodeIds = new Set(state.elements.nodes.map(n => n.data.id));
        const currentEdgeIds = new Set(state.elements.edges.map(e => e.data.id));

        const nodesToRemove = this.cy.nodes().filter(node => !currentNodeIds.has(node.id()));
        const edgesToRemove = this.cy.edges().filter(edge => !currentEdgeIds.has(edge.id()));

        return { nodes: nodesToRemove, edges: edgesToRemove };
    }

    _processBatchedUpdates(nodeUpdates, edgeUpdates, removals, batchSize, delay) {
        let currentIndex = 0;
        const allUpdates = [
            ...nodeUpdates.updates,
            ...edgeUpdates.updates,
            ...nodeUpdates.additions,
            ...edgeUpdates.additions,
            ...removals.nodes,
            ...removals.edges
        ];

        const processBatch = () => {
            const batch = allUpdates.slice(currentIndex, currentIndex + batchSize);
            
            batch.forEach(item => {
                try {
                    if (item.node) {
                        // Update existing node
                        item.node.data(item.data.data);
                        if (item.data.style) {
                            item.node.style(item.data.style);
                        }
                        // Preserve position if it exists
                        if (item.data.data.position) {
                            item.node.position(item.data.data.position);
                        }
                    } else if (item.edge) {
                        // Update existing edge
                        item.edge.data(item.data.data);
                        if (item.data.style) {
                            item.edge.style(item.data.style);
                        }
                    } else if (item.data && item.data.data) {
                        // Add new element
                        this.cy.add(item.data);
                        // Apply position immediately if it exists and preserve it
                        if (item.data.data.position) {
                            const addedNode = this.cy.getElementById(item.data.data.id);
                            if (addedNode.length > 0) {
                                addedNode.position(item.data.data.position);
                                
                                // Preserve original position for layer layout
                                if (item.data.data.layer !== undefined) {
                                    addedNode.data('original_position', item.data.data.position);
                                }
                            }
                        }
                    } else if (item.remove) {
                        // Remove element
                        item.remove();
                    }
                } catch (error) {
                    console.warn('Failed to update element:', error);
                }
            });

            currentIndex += batchSize;

            if (currentIndex < allUpdates.length) {
                // Schedule next batch
                setTimeout(processBatch, delay);
            } else {
                // All updates complete
                this._onUpdateComplete();
            }
        };

        // Start processing
        processBatch();
    }

    _onUpdateComplete() {
        // Trigger any post-update actions
        this.emit('update_complete');
        
        // Force a redraw if needed
        if (this.cy) {
            this.cy.style().update();
        }
    }






    async applyLayout(layoutName) {
        try {
            // Get layout configuration
            const layoutConfig = window.dataManager.getLayoutConfig(layoutName);

            // Stop current layout if running
            if (this.currentLayout) {
                this.currentLayout.stop();
            }

            // Apply new layout
            this.currentLayout = this.cy.layout(layoutConfig);
            this.layoutName = layoutName;

            // Run the layout
            this.currentLayout.run();

            console.log(`Applied ${layoutName} layout`);
        } catch (error) {
            console.error(`Failed to apply ${layoutName} layout:`, error);

            // Fallback to default layout
            this.currentLayout = this.cy.layout({ name: 'cose' });
            this.currentLayout.run();
        }
    }

    fitToView() {
        this.cy.fit(null, 50); // 50px padding
    }

    centerOnNode(nodeId) {
        const node = this.cy.getElementById(nodeId);
        if (node.length > 0) {
            this.cy.center(node);
            this.cy.zoom({
                level: 1.5,
                renderedPosition: node.renderedPosition()
            });
        }
    }

    highlightNode(nodeId) {
        // Remove previous highlights
        this.cy.elements().removeClass('highlighted');

        const node = this.cy.getElementById(nodeId);
        if (node.length > 0) {
            node.addClass('highlighted');

            // Highlight connected edges
            node.connectedEdges().addClass('highlighted');
        }
    }

    clearHighlights() {
        this.cy.elements().removeClass('highlighted');
    }

    // Animation methods
    startAnimationLoop() {
        const animate = (timestamp) => {
            // Calculate FPS
            this.frameCount++;
            if (timestamp - this.fpsUpdateTime >= 1000) {
                this.fps = Math.round((this.frameCount * 1000) / (timestamp - this.fpsUpdateTime));
                this.frameCount = 0;
                this.fpsUpdateTime = timestamp;

                // Update FPS display
                const fpsElement = document.getElementById('fps-counter');
                if (fpsElement) {
                    fpsElement.textContent = `FPS: ${this.fps}`;
                }
            }

            // Update traveling signals
            this.updateTravelingSignals();

            this.lastFrameTime = timestamp;
            this.animationFrame = requestAnimationFrame(animate);
        };

        this.animationFrame = requestAnimationFrame(animate);
    }

    stopAnimationLoop() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
    }

    updateTravelingSignals() {
        const signals = window.dataManager.getTravelingSignals();

        // Remove existing signal overlays
        this.removeSignalOverlays();

        // Add new signal overlays
        signals.forEach(signal => {
            this.addSignalOverlay(signal);
        });
    }

    addSignalOverlay(signal) {
        try {
            const sourceNode = this.cy.getElementById(signal.source);
            const targetNode = this.cy.getElementById(signal.target);

            if (sourceNode.length === 0 || targetNode.length === 0) {
                return;
            }

            const sourcePos = sourceNode.renderedPosition();
            const targetPos = targetNode.renderedPosition();

            // Calculate signal position based on progress
            const progress = signal.progress || 0;
            const signalPos = {
                x: sourcePos.x + (targetPos.x - sourcePos.x) * progress,
                y: sourcePos.y + (targetPos.y - sourcePos.y) * progress
            };

            // Simple color validation - just check for NaN
            let backgroundColor = signal.color || '#ff8800';
            if (backgroundColor.includes('NaN')) {
                backgroundColor = '#ff8800';
            }

            // Simple size validation
            let signalSize = 10;
            if (signal.size && !isNaN(signal.size) && signal.size > 0) {
                signalSize = Math.sqrt(signal.size);
            }

            // Create signal element overlay
            const container = document.getElementById(this.containerId);
            const signalElement = document.createElement('div');
            signalElement.className = 'traveling-signal';
            signalElement.style.cssText = `
                position: absolute;
                width: ${signalSize}px;
                height: ${signalSize}px;
                background-color: ${backgroundColor};
                border-radius: 50%;
                border: 1px solid #000;
                pointer-events: none;
                z-index: 1000;
                transform: translate(-50%, -50%);
                left: ${signalPos.x}px;
                top: ${signalPos.y}px;
                opacity: 0.8;
            `;
            signalElement.setAttribute('data-signal-id', signal.id);

            container.appendChild(signalElement);

        } catch (error) {
            console.error('Error adding signal overlay:', error);
        }
    }

    removeSignalOverlays() {
        const container = document.getElementById(this.containerId);
        const signals = container.querySelectorAll('.traveling-signal');
        signals.forEach(signal => signal.remove());
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
                    console.error(`Error in visualization event handler for ${event}:`, error);
                }
            });
        }
    }

    // Apply layer-based layout
    applyLayerLayout() {
        if (!this.cy) return;
        
        // Stop any running layout
        if (this.currentLayout) {
            this.currentLayout.stop();
        }
        
        // Get nodes with position data from the server
        const nodes = this.cy.nodes();
        let hasPositions = false;
        
        nodes.forEach(node => {
            const data = node.data();
            if (data.layer !== undefined && data.layer >= 0) {
                // Check if the node has position data in the data section
                let position = data.position;
                
                // If no position in data, try to restore from preserved node data
                if (!position) {
                    const originalPos = node.data('original_position');
                    if (originalPos) {
                        position = originalPos;
                    }
                }
                
                // If still no position, try to restore from stored original positions
                if (!position && this.originalPositions && this.originalPositions[data.id]) {
                    position = this.originalPositions[data.id];
                }
                
                if (position && position.x !== undefined && position.y !== undefined) {
                    // Apply the position
                    node.position({
                        x: position.x,
                        y: position.y
                    });
                    hasPositions = true;
                }
                
                // Set z-index for layering (external inputs get higher z-index to appear above neurons)
                if (data.type === 'external') {
                    node.style('z-index', 2000 + data.layer);  // External inputs above neurons
                } else {
                    node.style('z-index', 1000 - data.layer);  // Neurons with layer-based z-index
                }
            }
        });
        
        if (hasPositions) {
            // Fit the view to show all nodes
            this.cy.fit();
            this.layoutName = 'layers';
            console.log('Applied layer-based layout with positions');
        } else {
            console.log('No position data found, falling back to traditional layout');
            this.applyTraditionalLayout('cose');
        }
    }
    
    // Store original positions before applying traditional layout
    storeOriginalPositions() {
        if (!this.cy) return;
        
        this.originalPositions = {};
        const nodes = this.cy.nodes();
        
        nodes.forEach(node => {
            const data = node.data();
            if (data.layer !== undefined && data.layer >= 0) {
                // Get current position (either from data or from node's current position)
                let currentPosition = data.position;
                if (!currentPosition) {
                    const nodePos = node.position();
                    if (nodePos && nodePos.x !== undefined && nodePos.y !== undefined) {
                        currentPosition = { x: nodePos.x, y: nodePos.y };
                    }
                }
                
                if (currentPosition) {
                    // Store in memory for quick access
                    this.originalPositions[data.id] = {
                        x: currentPosition.x,
                        y: currentPosition.y,
                        layer: data.layer
                    };
                    
                    // Also ensure it's preserved in the node's data (this should never be overwritten)
                    node.data('original_position', {
                        x: currentPosition.x,
                        y: currentPosition.y
                    });
                }
            }
        });
        
        console.log(`Stored ${Object.keys(this.originalPositions).length} original positions`);
    }
    
    // Apply traditional layout
    applyTraditionalLayout(layoutName = 'cose') {
        if (!this.cy) return;
        
        // Store original positions before applying traditional layout
        this.storeOriginalPositions();
        
        // Stop any running layout
        if (this.currentLayout) {
            this.currentLayout.stop();
        }
        
        // Get layout configuration from data manager
        const layoutConfig = window.dataManager.getLayoutConfig(layoutName);
        
        // Apply the layout
        this.currentLayout = this.cy.layout(layoutConfig);
        this.currentLayout.run();
        
        this.layoutName = layoutName;
        console.log(`Applied ${layoutName} layout`);
    }
    
    // Toggle between layer and traditional layout
    toggleLayout() {
        if (this.layoutName === 'layers') {
            this.applyTraditionalLayout('cose');
        } else {
            this.applyLayerLayout();
        }
    }

    // Check if current layout is layer-based
    isLayerLayout() {
        return this.layoutName === 'layers';
    }

    // Utility methods
    getSelectedNodes() {
        return this.cy.nodes(':selected');
    }



    getSelectedEdges() {
        return this.cy.edges(':selected');
    }

    exportImage(format = 'png', scale = 2) {
        return this.cy.png({
            output: 'blob',
            scale: scale,
            full: true,
            bg: '#ffffff'
        });
    }

    destroy() {
        this.stopAnimationLoop();
        this.removeSignalOverlays();
        if (this.cy) {
            this.cy.destroy();
        }
    }
}

// Global network visualization instance
window.networkViz = null;