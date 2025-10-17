/**
 * Layout algorithms for Sigma.js
 * Includes: ForceAtlas2, Circular, Random, Grid, NoOverlap
 */

const SigmaLayouts = {
    /**
     * Apply circular layout to a graph
     */
    circular(graph, options = {}) {
        const nodes = graph.nodes();
        const center = { x: options.center?.x || 0, y: options.center?.y || 0 };
        const radius = options.radius || 100;
        const startAngle = options.startAngle || 0;

        nodes.forEach((node, index) => {
            const angle = startAngle + (2 * Math.PI * index) / nodes.length;
            graph.setNodeAttribute(node, 'x', center.x + radius * Math.cos(angle));
            graph.setNodeAttribute(node, 'y', center.y + radius * Math.sin(angle));
        });

        return graph;
    },

    /**
     * Apply random layout to a graph
     */
    random(graph, options = {}) {
        const nodes = graph.nodes();
        const scale = options.scale || 100;

        nodes.forEach(node => {
            graph.setNodeAttribute(node, 'x', (Math.random() - 0.5) * scale);
            graph.setNodeAttribute(node, 'y', (Math.random() - 0.5) * scale);
        });

        return graph;
    },

    /**
     * Apply grid layout to a graph
     */
    grid(graph, options = {}) {
        const nodes = graph.nodes();
        const cols = options.cols || Math.ceil(Math.sqrt(nodes.length));
        const spacing = options.spacing || 50;
        const center = { x: options.center?.x || 0, y: options.center?.y || 0 };

        nodes.forEach((node, index) => {
            const col = index % cols;
            const row = Math.floor(index / cols);
            graph.setNodeAttribute(node, 'x', center.x + (col - cols / 2) * spacing);
            graph.setNodeAttribute(node, 'y', center.y + (row - Math.floor(nodes.length / cols) / 2) * spacing);
        });

        return graph;
    },

    /**
     * Apply simplified ForceAtlas2 layout (optimized for performance)
     */
    forceAtlas2(graph, options = {}) {
        const iterations = Math.min(options.iterations || 50, 50); // Cap iterations
        const nodes = graph.nodes();

        // Skip if too many nodes to prevent freezing
        if (nodes.length > 200) {
            console.warn('Too many nodes for ForceAtlas2, using random layout');
            return this.random(graph, options);
        }

        // Initialize random positions if not set
        nodes.forEach(node => {
            if (!graph.hasNodeAttribute(node, 'x')) {
                graph.setNodeAttribute(node, 'x', (Math.random() - 0.5) * 200);
            }
            if (!graph.hasNodeAttribute(node, 'y')) {
                graph.setNodeAttribute(node, 'y', (Math.random() - 0.5) * 200);
            }
        });

        // Simple force simulation
        for (let iter = 0; iter < iterations; iter++) {
            const forces = new Map();

            // Initialize forces
            nodes.forEach(node => {
                forces.set(node, { x: 0, y: 0 });
            });

            // Repulsion (simplified O(n²) but limited)
            for (let i = 0; i < Math.min(nodes.length, 100); i++) {
                for (let j = i + 1; j < Math.min(nodes.length, 100); j++) {
                    const node1 = nodes[i];
                    const node2 = nodes[j];

                    const x1 = graph.getNodeAttribute(node1, 'x');
                    const y1 = graph.getNodeAttribute(node1, 'y');
                    const x2 = graph.getNodeAttribute(node2, 'x');
                    const y2 = graph.getNodeAttribute(node2, 'y');

                    const dx = x2 - x1;
                    const dy = y2 - y1;
                    const distance = Math.max(Math.sqrt(dx * dx + dy * dy), 0.1);

                    const repulsion = 1000 / (distance * distance);
                    const fx = (dx / distance) * repulsion;
                    const fy = (dy / distance) * repulsion;

                    const force1 = forces.get(node1);
                    const force2 = forces.get(node2);
                    force1.x -= fx;
                    force1.y -= fy;
                    force2.x += fx;
                    force2.y += fy;
                }
            }

            // Attraction along edges
            graph.forEachEdge((edge, attributes, source, target) => {
                const x1 = graph.getNodeAttribute(source, 'x');
                const y1 = graph.getNodeAttribute(source, 'y');
                const x2 = graph.getNodeAttribute(target, 'x');
                const y2 = graph.getNodeAttribute(target, 'y');

                const dx = x2 - x1;
                const dy = y2 - y1;
                const distance = Math.max(Math.sqrt(dx * dx + dy * dy), 0.1);

                const attraction = distance * 0.01;
                const fx = (dx / distance) * attraction;
                const fy = (dy / distance) * attraction;

                const forceSource = forces.get(source);
                const forceTarget = forces.get(target);
                forceSource.x += fx;
                forceSource.y += fy;
                forceTarget.x -= fx;
                forceTarget.y -= fy;
            });

            // Apply forces
            nodes.forEach(node => {
                const force = forces.get(node);
                const x = graph.getNodeAttribute(node, 'x');
                const y = graph.getNodeAttribute(node, 'y');

                // Damping and movement
                const damping = 0.1;
                graph.setNodeAttribute(node, 'x', x + force.x * damping);
                graph.setNodeAttribute(node, 'y', y + force.y * damping);
            });
        }

        return graph;
    },

    /**
     * Apply NoOverlap layout - prevent node overlaps (optimized)
     */
    noverlap(graph, options = {}) {
        const iterations = Math.min(options.iterations || 20, 20);
        const margin = options.margin || 10;
        const nodes = graph.nodes();

        // Skip if too many nodes
        if (nodes.length > 200) {
            console.warn('Too many nodes for NoOverlap');
            return graph;
        }

        for (let iter = 0; iter < iterations; iter++) {
            let moved = false;

            for (let i = 0; i < Math.min(nodes.length, 100); i++) {
                for (let j = i + 1; j < Math.min(nodes.length, 100); j++) {
                    const node1 = nodes[i];
                    const node2 = nodes[j];

                    const x1 = graph.getNodeAttribute(node1, 'x');
                    const y1 = graph.getNodeAttribute(node1, 'y');
                    const size1 = graph.getNodeAttribute(node1, 'size') || 10;

                    const x2 = graph.getNodeAttribute(node2, 'x');
                    const y2 = graph.getNodeAttribute(node2, 'y');
                    const size2 = graph.getNodeAttribute(node2, 'size') || 10;

                    const dx = x2 - x1;
                    const dy = y2 - y1;
                    const distance = Math.max(Math.sqrt(dx * dx + dy * dy), 0.1);
                    const minDistance = (size1 + size2) / 2 + margin;

                    if (distance < minDistance) {
                        const pushDistance = (minDistance - distance) / 2;
                        const pushX = (dx / distance) * pushDistance;
                        const pushY = (dy / distance) * pushDistance;

                        graph.setNodeAttribute(node1, 'x', x1 - pushX);
                        graph.setNodeAttribute(node1, 'y', y1 - pushY);
                        graph.setNodeAttribute(node2, 'x', x2 + pushX);
                        graph.setNodeAttribute(node2, 'y', y2 + pushY);
                        moved = true;
                    }
                }
            }

            // Early termination if no nodes moved
            if (!moved) break;
        }

        return graph;
    },

    /**
     * Layer-based layout for hierarchical networks (improved)
     */
    layerBased(graph, options = {}) {
        const nodes = graph.nodes();
        const nodeSpacing = options.spacing || 80;
        const layerSpacing = options.layerSpacing || 200;

        // Group nodes by layer and type
        const layers = {};
        const externalInputs = [];

        nodes.forEach(node => {
            const nodeType = graph.getNodeAttribute(node, 'nodeType');
            const layer = graph.getNodeAttribute(node, 'layer') || 0;

            if (nodeType === 'external') {
                externalInputs.push(node);
            } else {
                if (!layers[layer]) {
                    layers[layer] = [];
                }
                layers[layer].push(node);
            }
        });

        // Position neuron layers
        const layerKeys = Object.keys(layers).sort((a, b) => parseInt(a) - parseInt(b));

        layerKeys.forEach((layerKey, layerIndex) => {
            const layerNodes = layers[layerKey];
            const x = layerIndex * layerSpacing;

            // Arrange nodes in layer vertically with proper spacing
            const totalHeight = (layerNodes.length - 1) * nodeSpacing;
            const startY = -totalHeight / 2;

            layerNodes.forEach((node, index) => {
                const y = startY + index * nodeSpacing;
                graph.setNodeAttribute(node, 'x', x);
                graph.setNodeAttribute(node, 'y', y);
            });
        });

        // Position external inputs above their target layers
        externalInputs.forEach(extNode => {
            const targetNeuron = graph.getNodeAttribute(extNode, 'neuron_id');
            const targetLayer = graph.getNodeAttribute(extNode, 'layer') || 0;

            // Find the layer position
            const layerIndex = layerKeys.indexOf(targetLayer.toString());
            const layerX = layerIndex >= 0 ? layerIndex * layerSpacing : 0;

            // Position above the layer
            const offsetY = -100; // Position above neurons
            const offsetX = (Math.random() - 0.5) * 40; // Small random offset

            graph.setNodeAttribute(extNode, 'x', layerX + offsetX);
            graph.setNodeAttribute(extNode, 'y', offsetY);
        });

        return graph;
    }
};

// Make available globally
window.SigmaLayouts = SigmaLayouts;

