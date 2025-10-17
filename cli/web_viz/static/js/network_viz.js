/**
 * Main network visualization using Sigma.js
 */

class NetworkVisualization {
    constructor(containerId) {
        this.containerId = containerId;
        this.sigma = null;
        this.graph = null;
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

        // Store original positions for layer layout
        this.originalPositions = {};

        // Selected node state
        this.selectedNode = null;
    }

    async initialize() {
        try {
            // Wait for data manager to be ready
            if (!window.dataManager.getNetworkStyle()) {
                await window.dataManager.initialize();
            }

            // Initialize Graph and Sigma
            this.initializeSigma();

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

    initializeSigma() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            throw new Error(`Container element with ID '${this.containerId}' not found`);
        }

        // Create a new Graphology graph
        this.graph = new graphology.Graph();

        // Initialize Sigma.js renderer
        this.sigma = new Sigma(this.graph, container, {
            // Rendering settings
            renderLabels: true,
            renderEdgeLabels: false,
            enableEdgeEvents: true,

            // Default styles
            defaultNodeColor: '#87ceeb',
            defaultEdgeColor: '#999',

            // Interaction settings
            allowInvalidContainer: false,

            // Performance settings
            hideEdgesOnMove: false,  // Keep edges visible for better UX
            hideLabelsOnMove: false,

            // Label settings
            labelSize: 12,
            labelWeight: 'normal',
            labelColor: { color: '#000' }
        });

        console.log('Sigma instance created');
    }

    setupEventHandlers() {
        if (!this.sigma) return;

        // Node click events
        this.sigma.on('clickNode', (event) => {
            const nodeId = event.node;
            this.handleNodeClick(nodeId);
        });

        // Click on stage (deselect)
        this.sigma.on('clickStage', (event) => {
            this.handleStageClick();
        });

        // Node hover events
        this.sigma.on('enterNode', (event) => {
            const nodeId = event.node;
            this.handleNodeEnter(nodeId);
        });

        this.sigma.on('leaveNode', (event) => {
            const nodeId = event.node;
            this.handleNodeLeave(nodeId);
        });

        // Double-click to fit view
        this.sigma.on('doubleClickStage', (event) => {
            this.fitToView();
        });

        // Mouse wheel for zoom (get container reference)
        const container = document.getElementById(this.containerId);
        if (container) {
            container.addEventListener('wheel', (e) => {
                e.preventDefault();
            }, { passive: false });
        }

        // Enable node dragging
        this.setupDragging();
    }

    setupDragging() {
        let isDragging = false;
        let draggedNode = null;
        let startPos = null;
        let startNodePos = null;
        let lastRefresh = 0;

        // Mouse down on node - start dragging
        this.sigma.on('downNode', (e) => {
            isDragging = true;
            draggedNode = e.node;
            startPos = { x: e.event.x, y: e.event.y };

            const nodeAttrs = this.graph.getNodeAttributes(e.node);
            startNodePos = { x: nodeAttrs.x, y: nodeAttrs.y };

            // Prevent default camera behavior
            e.preventSigmaDefault();
            if (e.original) {
                e.original.preventDefault();
                e.original.stopPropagation();
            }
        });

        // Mouse move - update node position
        this.sigma.on('mousemove', (e) => {
            if (isDragging && draggedNode && startPos && startNodePos) {
                // Calculate the movement delta
                const deltaX = e.event.x - startPos.x;
                const deltaY = e.event.y - startPos.y;

                // Convert screen delta to graph coordinates using camera ratio
                const camera = this.sigma.getCamera();
                const ratio = camera.ratio || 1;

                // Update node position
                const newX = startNodePos.x + (deltaX * ratio);
                const newY = startNodePos.y + (deltaY * ratio);

                this.graph.setNodeAttribute(draggedNode, 'x', newX);
                this.graph.setNodeAttribute(draggedNode, 'y', newY);

                // Throttled refresh for smooth dragging
                const now = Date.now();
                if (now - lastRefresh > 16) { // ~60 FPS
                    this.sigma.refresh();
                    lastRefresh = now;
                }

                e.preventSigmaDefault();
                if (e.original) {
                    e.original.preventDefault();
                    e.original.stopPropagation();
                }
            }
        });

        // Mouse up - stop dragging
        this.sigma.on('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                draggedNode = null;
                startPos = null;
                startNodePos = null;

                // Final refresh after dragging is done
                this.sigma.refresh();
            }
        });

        // Mouse leave - stop dragging if mouse leaves the canvas
        this.sigma.on('mouseleave', () => {
            if (isDragging) {
                isDragging = false;
                draggedNode = null;
                startPos = null;
                startNodePos = null;
                // Dragging stopped - no need to re-enable camera
            }
        });
    }

    handleNodeClick(nodeId) {
        // Deselect previous node
        if (this.selectedNode && this.selectedNode !== nodeId) {
            this.graph.setNodeAttribute(this.selectedNode, 'highlighted', false);
            const oldNodeData = { data: () => ({ type: this.graph.getNodeAttribute(this.selectedNode, 'type') }) };
            this.emit('node_deselected', oldNodeData);
        }

        // Select new node
        this.selectedNode = nodeId;
        this.graph.setNodeAttribute(nodeId, 'highlighted', true);

        // Update node appearance
        this.updateNodeHighlight(nodeId, true);

        // Emit selection event
        const nodeData = this.createNodeDataObject(nodeId);
        this.emit('node_selected', nodeData);

        this.sigma.refresh();
    }

    handleStageClick() {
        if (this.selectedNode) {
            this.graph.setNodeAttribute(this.selectedNode, 'highlighted', false);
            this.updateNodeHighlight(this.selectedNode, false);

            const nodeData = this.createNodeDataObject(this.selectedNode);
            this.emit('node_deselected', nodeData);

            this.selectedNode = null;
            this.sigma.refresh();
        }
    }

    handleNodeEnter(nodeId) {
        // Add hover styling
        this.graph.setNodeAttribute(nodeId, 'hovered', true);
        const currentSize = this.graph.getNodeAttribute(nodeId, 'size');
        this.graph.setNodeAttribute(nodeId, 'size', currentSize * 1.2);
        this.sigma.refresh();
    }

    handleNodeLeave(nodeId) {
        // Remove hover styling
        this.graph.setNodeAttribute(nodeId, 'hovered', false);
        const originalSize = this.graph.getNodeAttribute(nodeId, 'originalSize') || 10;
        this.graph.setNodeAttribute(nodeId, 'size', originalSize);
        this.sigma.refresh();
    }

    updateNodeHighlight(nodeId, highlighted) {
        if (highlighted) {
            // Highlight the node
            const currentColor = this.graph.getNodeAttribute(nodeId, 'color');
            this.graph.setNodeAttribute(nodeId, 'originalColor', currentColor);
            this.graph.setNodeAttribute(nodeId, 'borderColor', '#FFD700');
            this.graph.setNodeAttribute(nodeId, 'borderSize', 3);

            // Highlight connected edges
            this.graph.forEachEdge(nodeId, (edge, attributes, source, target) => {
                this.graph.setEdgeAttribute(edge, 'highlighted', true);
                this.graph.setEdgeAttribute(edge, 'size', 3);
            });
        } else {
            // Remove highlight
            this.graph.removeNodeAttribute(nodeId, 'borderColor');
            this.graph.removeNodeAttribute(nodeId, 'borderSize');

            // Remove edge highlights
            this.graph.forEachEdge(nodeId, (edge) => {
                this.graph.setEdgeAttribute(edge, 'highlighted', false);
                this.graph.setEdgeAttribute(edge, 'size', 1);
            });
        }
    }

    createNodeDataObject(nodeId) {
        // Create a node data object compatible with the existing event handlers
        const attributes = this.graph.getNodeAttributes(nodeId);
        return {
            data: () => ({
                ...attributes,
                type: attributes.nodeType  // Map nodeType back to type for compatibility
            }),
            id: () => nodeId
        };
    }

    async loadNetworkData() {
        try {
            const state = window.dataManager.getCurrentState();
            if (!state || !state.elements) {
                console.warn('No network data available');
                return;
            }

            // Clear existing elements
            this.graph.clear();

            // Add nodes
            if (state.elements.nodes && state.elements.nodes.length > 0) {
                state.elements.nodes.forEach(nodeData => {
                    this.addNode(nodeData);
                });
            }

            // Add edges
            if (state.elements.edges && state.elements.edges.length > 0) {
                state.elements.edges.forEach(edgeData => {
                    this.addEdge(edgeData);
                });
            }

            // Apply layer-based layout by default
            this.applyLayerLayout();

            console.log(`Loaded ${this.graph.order} nodes and ${this.graph.size} edges`);
        } catch (error) {
            console.error('Failed to load network data:', error);
            throw error;
        }
    }

    addNode(nodeData) {
        const id = nodeData.data.id;
        const data = nodeData.data;

        // Determine node color based on state
        let color = this.getNodeColor(data);
        let size = this.getNodeSize(data);

        // Debug initial node creation
        console.log(`Creating node ${id}: type=${data.type}, output=${data.output}, is_firing=${data.is_firing}, color=${color}`);

        // Use 'circle' for all nodes (Sigma.js v2 only supports circle by default)
        const sigmaNodeType = 'circle';

        // Add visual distinction for external nodes
        const nodeAttributes = {
            x: data.position?.x || Math.random() * 100,
            y: data.position?.y || Math.random() * 100,
            size: size,
            originalSize: size,
            color: color,
            label: data.label || id,
            type: sigmaNodeType,  // Use Sigma.js built-in types
            nodeType: data.type,  // Store original type for logic
            layer: data.layer,
            layer_name: data.layer_name,
            neuron_id: data.neuron_id,
            membrane_potential: data.membrane_potential,
            firing_rate: data.firing_rate,
            output: data.output,
            is_firing: data.is_firing,
            highlighted: false,
            hovered: false,
            // Store original position for layer layout
            original_position: data.position
        };

        // Add border for external nodes to make them visually distinct
        if (data.type === 'external') {
            nodeAttributes.borderSize = 2;
            nodeAttributes.borderColor = '#006400';
        }

        // Add the node to the graph
        this.graph.addNode(id, nodeAttributes);
    }

    addEdge(edgeData) {
        const id = edgeData.data.id;
        const source = edgeData.data.source;
        const target = edgeData.data.target;
        const data = edgeData.data;

        // Ensure both nodes exist
        if (!this.graph.hasNode(source) || !this.graph.hasNode(target)) {
            console.warn(`Cannot add edge ${id}: source or target node missing`);
            return;
        }

        // Determine edge color and style
        let color = data.color || '#999';
        let size = data.weight || 1;

        // Add the edge to the graph
        this.graph.addEdge(source, target, {
            id: id,
            size: size,
            color: color,
            type: 'arrow',  // Sigma.js built-in edge type
            edgeType: data.type,  // Store original type for logic
            label: data.label || '',
            weight: data.weight || 1,
            highlighted: false
        });
    }

    getNodeColor(data) {
        // Color based on neuron state
        if (data.type === 'external') {
            return '#90ee90'; // Light green for external inputs
        }

        // Check for firing state (output > 0 or is_firing flag)
        const output = data.output || 0;
        const isFiring = data.is_firing || output > 0;

        if (isFiring) {
            console.log(`Firing neuron detected: ${data.neuron_id || data.id}, output: ${output}, is_firing: ${data.is_firing}`);
            return '#ff4444'; // Red for firing
        }

        const potential = data.membrane_potential || 0;
        const threshold = 0.9; // Approximate firing threshold

        if (potential > threshold * 0.8) {
            return '#ff8800'; // Orange for high potential
        } else if (potential > threshold * 0.5) {
            return '#ffdd00'; // Yellow for moderate potential
        } else {
            return '#87ceeb'; // Light blue for low/inactive
        }
    }

    getNodeSize(data) {
        if (data.type === 'external') {
            return 6;  // Smaller size for external inputs
        }

        // Size based on activity for neurons
        const baseSize = 12;
        const potential = data.membrane_potential || 0;
        const activity = data.firing_rate || 0;

        return baseSize + activity * 5 + potential * 2;
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
            // Update nodes
            if (state.elements.nodes) {
                state.elements.nodes.forEach(nodeData => {
                    const id = nodeData.data.id;
                    if (this.graph.hasNode(id)) {
                        this.updateNode(id, nodeData.data);
                    } else {
                        this.addNode(nodeData);
                    }
                });
            }

            // Update edges
            if (state.elements.edges) {
                state.elements.edges.forEach(edgeData => {
                    const id = edgeData.data.id;
                    const source = edgeData.data.source;
                    const target = edgeData.data.target;

                    if (this.graph.hasEdge(source, target)) {
                        this.updateEdge(source, target, edgeData.data);
                    } else {
                        this.addEdge(edgeData);
                    }
                });
            }

            // Force refresh after all updates
            if (this.sigma) {
                this.sigma.refresh();
            }

            this._onUpdateComplete();

        } catch (error) {
            console.error('Failed to perform batched update:', error);
        }
    }

    updateNode(nodeId, data) {
        if (!this.graph.hasNode(nodeId)) return;

        // Update node attributes
        const color = this.getNodeColor(data);
        const size = this.getNodeSize(data);

        // Debug firing neurons
        if (data.is_firing || data.output > 0) {
            console.log(`Updating firing node ${nodeId}: output=${data.output}, is_firing=${data.is_firing}, color=${color}`);
        }

        this.graph.setNodeAttribute(nodeId, 'color', color);
        this.graph.setNodeAttribute(nodeId, 'size', size);
        this.graph.setNodeAttribute(nodeId, 'originalSize', size);
        this.graph.setNodeAttribute(nodeId, 'membrane_potential', data.membrane_potential);
        this.graph.setNodeAttribute(nodeId, 'firing_rate', data.firing_rate);
        this.graph.setNodeAttribute(nodeId, 'output', data.output);
        this.graph.setNodeAttribute(nodeId, 'is_firing', data.is_firing);

        // Add border for external nodes
        if (data.type === 'external') {
            this.graph.setNodeAttribute(nodeId, 'borderSize', 2);
            this.graph.setNodeAttribute(nodeId, 'borderColor', '#006400');
        }

        // Preserve position if in layer layout
        if (data.position && this.layoutName === 'layers') {
            this.graph.setNodeAttribute(nodeId, 'x', data.position.x);
            this.graph.setNodeAttribute(nodeId, 'y', data.position.y);
        }
    }

    updateEdge(source, target, data) {
        const color = data.color || '#999';
        const size = data.weight || 1;

        this.graph.setEdgeAttribute(source, target, 'color', color);
        this.graph.setEdgeAttribute(source, target, 'size', size);
    }

    _onUpdateComplete() {
        // Trigger any post-update actions
        this.emit('update_complete');
    }

    async applyLayout(layoutName) {
        try {
            console.log(`Applying ${layoutName} layout...`);

            switch (layoutName) {
                case 'circular':
                    this.applyCircularLayout();
                    break;
                case 'random':
                    this.applyRandomLayout();
                    break;
                case 'grid':
                    this.applyGridLayout();
                    break;
                case 'forceAtlas2':
                    this.applyForceAtlas2Layout();
                    break;
                case 'noverlap':
                    this.applyNoverlapLayout();
                    break;
                case 'cose':
                case 'force':
                    this.applyForceAtlas2Layout();
                    break;
                default:
                    console.warn(`Unknown layout: ${layoutName}, using circular`);
                    this.applyCircularLayout();
            }

            this.layoutName = layoutName;
            this.fitToView();
            this.sigma.refresh();

            console.log(`Applied ${layoutName} layout`);
            this.emit('layout_complete', { name: layoutName });
        } catch (error) {
            console.error(`Failed to apply ${layoutName} layout:`, error);
        }
    }

    applyCircularLayout() {
        window.SigmaLayouts.circular(this.graph, {
            radius: 200
        });
    }

    applyRandomLayout() {
        window.SigmaLayouts.random(this.graph, {
            scale: 300
        });
    }

    applyGridLayout() {
        window.SigmaLayouts.grid(this.graph, {
            spacing: 80
        });
    }

    applyForceAtlas2Layout() {
        window.SigmaLayouts.forceAtlas2(this.graph, {
            iterations: 100,
            gravity: 1,
            scalingRatio: 10
        });
    }

    applyNoverlapLayout() {
        // First apply a force layout, then noverlap
        this.applyForceAtlas2Layout();
        window.SigmaLayouts.noverlap(this.graph, {
            iterations: 50,
            margin: 10
        });
    }

    fitToView() {
        if (this.sigma && this.graph.order > 0) {
            // Calculate bounds
            let minX = Infinity, maxX = -Infinity;
            let minY = Infinity, maxY = -Infinity;
            let hasNodes = false;

            this.graph.forEachNode((node, attributes) => {
                if (attributes.x !== undefined && attributes.y !== undefined) {
                    minX = Math.min(minX, attributes.x);
                    maxX = Math.max(maxX, attributes.x);
                    minY = Math.min(minY, attributes.y);
                    maxY = Math.max(maxY, attributes.y);
                    hasNodes = true;
                }
            });

            if (!hasNodes) {
                // No positioned nodes, reset to default view
                const camera = this.sigma.getCamera();
                camera.animate({ x: 0, y: 0, ratio: 1 }, { duration: 300 });
                return;
            }

            // Add padding proportional to the graph size
            const graphWidth = maxX - minX;
            const graphHeight = maxY - minY;
            const padding = Math.max(50, Math.min(graphWidth, graphHeight) * 0.1);

            const centerX = (minX + maxX) / 2;
            const centerY = (minY + maxY) / 2;

            // Get container dimensions
            const container = this.sigma.getContainer();
            const containerWidth = container.offsetWidth;
            const containerHeight = container.offsetHeight;

            // Calculate zoom to fit content with padding
            const contentWidth = graphWidth + 2 * padding;
            const contentHeight = graphHeight + 2 * padding;

            const zoomX = containerWidth / contentWidth;
            const zoomY = containerHeight / contentHeight;
            const zoom = Math.min(zoomX, zoomY, 2); // Cap at 2x zoom

            // Apply camera position (ratio is 1/zoom in Sigma.js)
            const camera = this.sigma.getCamera();
            camera.animate({
                x: centerX,
                y: centerY,
                ratio: 1 / Math.max(zoom, 0.1) // Ensure minimum zoom
            }, { duration: 500 });
        }
    }

    centerOnNode(nodeId) {
        if (this.graph.hasNode(nodeId)) {
            const { x, y } = this.graph.getNodeAttributes(nodeId);
            const camera = this.sigma.getCamera();
            camera.animate({ x, y, ratio: 0.5 }, { duration: 300 });
        }
    }

    highlightNode(nodeId) {
        // Remove previous highlights
        this.clearHighlights();

        if (this.graph.hasNode(nodeId)) {
            this.updateNodeHighlight(nodeId, true);
            this.sigma.refresh();
        }
    }

    clearHighlights() {
        this.graph.forEachNode((node) => {
            this.graph.removeNodeAttribute(node, 'borderColor');
            this.graph.removeNodeAttribute(node, 'borderSize');
        });

        this.graph.forEachEdge((edge) => {
            this.graph.setEdgeAttribute(edge, 'highlighted', false);
            this.graph.setEdgeAttribute(edge, 'size', 1);
        });
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
        // Signal rendering is handled by animations.js
    }

    // Apply layer-based layout
    applyLayerLayout() {
        if (!this.graph) return;

        // Store original positions before applying layer layout
        this.storeOriginalPositions();

        // Use built-in layer layout
        window.SigmaLayouts.layerBased(this.graph, {
            spacing: 100,
            layerSpacing: 150
        });

        this.layoutName = 'layers';
        this.fitToView();
        this.sigma.refresh();
        console.log('Applied layer-based layout');
    }

    storeOriginalPositions() {
        if (!this.graph) return;

        this.originalPositions = {};
        this.graph.forEachNode((node, attributes) => {
            if (attributes.original_position) {
                this.originalPositions[node] = {
                    x: attributes.original_position.x,
                    y: attributes.original_position.y,
                    layer: attributes.layer
                };
            } else if (attributes.x !== undefined && attributes.y !== undefined) {
                this.originalPositions[node] = {
                    x: attributes.x,
                    y: attributes.y,
                    layer: attributes.layer
                };
            }
        });

        console.log(`Stored ${Object.keys(this.originalPositions).length} original positions`);
    }

    toggleLayout() {
        if (this.layoutName === 'layers') {
            this.applyLayout('forceAtlas2');
        } else {
            this.applyLayerLayout();
        }
    }

    isLayerLayout() {
        return this.layoutName === 'layers';
    }

    // Utility methods
    getSelectedNodes() {
        return this.selectedNode ? [this.selectedNode] : [];
    }

    getSelectedEdges() {
        // Not directly supported in Sigma.js
        return [];
    }

    exportImage(format = 'png', scale = 2) {
        // Sigma.js doesn't have built-in export, would need to use html2canvas or similar
        console.warn('Image export not implemented for Sigma.js yet');
        return null;
    }

    destroy() {
        this.stopAnimationLoop();
        if (this.sigma) {
            this.sigma.kill();
        }
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

    // Getter for Cytoscape compatibility
    get cy() {
        // Return a compatibility object
        return {
            nodes: () => ({
                filter: (selector) => [],
                forEach: (callback) => {
                    if (selector === ':selected') {
                        if (this.selectedNode) {
                            callback(this.createNodeDataObject(this.selectedNode));
                        }
                    } else {
                        this.graph.forEachNode((node, attributes) => {
                            callback(this.createNodeDataObject(node));
                        });
                    }
                }
            }),
            edges: () => ({
                filter: (selector) => [],
                forEach: (callback) => {
                    this.graph.forEachEdge((edge, attributes) => {
                        callback({ id: () => edge });
                    });
                }
            }),
            elements: () => ({
                removeClass: (className) => {
                    // Clear highlights
                    this.clearHighlights();
                }
            }),
            getElementById: (id) => {
                if (this.graph.hasNode(id)) {
                    return this.createNodeDataObject(id);
                }
                return { length: 0 };
            }
        };
    }
}

// Global network visualization instance
window.networkViz = null;
