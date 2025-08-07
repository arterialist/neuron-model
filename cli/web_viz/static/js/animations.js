/**
 * Animation system for traveling signals and visual effects
 */

class AnimationSystem {
    constructor() {
        this.animations = new Map();
        this.animationId = 0;
        this.isRunning = false;
        this.lastFrameTime = 0;

        // Animation settings
        this.settings = {
            signalSpeed: 0.02, // Progress per frame (0-1)
            signalFadeTime: 500, // ms
            nodeFlashDuration: 200, // ms
            edgeHighlightDuration: 300, // ms
            smoothingFactor: 0.1 // For smooth transitions
        };
    }

    start() {
        if (!this.isRunning) {
            this.isRunning = true;
            this.animate();
        }
    }

    stop() {
        this.isRunning = false;
        this.clearAllAnimations();
    }

    animate(timestamp = 0) {
        if (!this.isRunning) return;

        const deltaTime = timestamp - this.lastFrameTime;
        this.lastFrameTime = timestamp;

        // Update all active animations
        this.updateAnimations(deltaTime);

        // Continue animation loop
        requestAnimationFrame((ts) => this.animate(ts));
    }

    updateAnimations(deltaTime) {
        const completedAnimations = [];

        this.animations.forEach((animation, id) => {
            try {
                const isComplete = this.updateAnimation(animation, deltaTime);
                if (isComplete) {
                    completedAnimations.push(id);
                }
            } catch (error) {
                console.error(`Error updating animation ${id}:`, error);
                completedAnimations.push(id);
            }
        });

        // Remove completed animations
        completedAnimations.forEach(id => {
            this.removeAnimation(id);
        });
    }

    updateAnimation(animation, deltaTime) {
        switch (animation.type) {
            case 'traveling_signal':
                return this.updateTravelingSignal(animation, deltaTime);
            case 'node_flash':
                return this.updateNodeFlash(animation, deltaTime);
            case 'edge_highlight':
                return this.updateEdgeHighlight(animation, deltaTime);
            case 'fade_out':
                return this.updateFadeOut(animation, deltaTime);
            default:
                console.warn(`Unknown animation type: ${animation.type}`);
                return true; // Mark as complete
        }
    }

    updateTravelingSignal(animation, deltaTime) {
        const { element, startTime, duration, startPos, endPos, color, size } = animation;

        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Calculate current position
        const currentPos = {
            x: startPos.x + (endPos.x - startPos.x) * progress,
            y: startPos.y + (endPos.y - startPos.y) * progress
        };

        // Update element position
        if (element && element.parentNode) {
            element.style.left = `${currentPos.x}px`;
            element.style.top = `${currentPos.y}px`;

            // Add pulsing effect
            const pulse = 1 + 0.2 * Math.sin(elapsed * 0.01);
            element.style.transform = `translate(-50%, -50%) scale(${pulse})`;
        }

        return progress >= 1;
    }

    updateNodeFlash(animation, deltaTime) {
        const { nodeId, startTime, duration, originalColor, flashColor } = animation;

        if (!window.networkViz || !window.networkViz.cy) {
            return true;
        }

        const node = window.networkViz.cy.getElementById(nodeId);
        if (node.length === 0) {
            return true;
        }

        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Interpolate between flash color and original color
        const intensity = Math.sin(progress * Math.PI); // Sine wave for smooth flash
        const currentColor = this.interpolateColor(originalColor, flashColor, intensity);

        node.style('background-color', currentColor);

        if (progress >= 1) {
            // Restore original color
            node.style('background-color', originalColor);
            return true;
        }

        return false;
    }

    updateEdgeHighlight(animation, deltaTime) {
        const { edgeId, startTime, duration, originalColor, highlightColor, originalWidth } = animation;

        if (!window.networkViz || !window.networkViz.cy) {
            return true;
        }

        const edge = window.networkViz.cy.getElementById(edgeId);
        if (edge.length === 0) {
            return true;
        }

        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Animate color and width
        const intensity = 1 - progress; // Fade out highlight
        const currentColor = this.interpolateColor(originalColor, highlightColor, intensity);
        const currentWidth = originalWidth + (4 - originalWidth) * intensity;

        edge.style({
            'line-color': currentColor,
            'target-arrow-color': currentColor,
            'width': currentWidth
        });

        if (progress >= 1) {
            // Restore original style
            edge.style({
                'line-color': originalColor,
                'target-arrow-color': originalColor,
                'width': originalWidth
            });
            return true;
        }

        return false;
    }

    updateFadeOut(animation, deltaTime) {
        const { element, startTime, duration, startOpacity } = animation;

        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);

        const currentOpacity = startOpacity * (1 - progress);

        if (element && element.parentNode) {
            element.style.opacity = currentOpacity;
        }

        if (progress >= 1) {
            // Remove element
            if (element && element.parentNode) {
                element.parentNode.removeChild(element);
            }
            return true;
        }

        return false;
    }

    // Animation creation methods
    createTravelingSignal(sourcePos, targetPos, color = '#ff8800', size = 10, duration = 2000) {
        const container = document.getElementById('cy');
        if (!container) {
            console.error('Cytoscape container not found');
            return null;
        }

        // Create signal element
        const element = document.createElement('div');
        element.className = 'traveling-signal';
        element.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            background-color: ${color};
            border-radius: 50%;
            border: 1px solid #000;
            pointer-events: none;
            z-index: 1000;
            transform: translate(-50%, -50%);
            left: ${sourcePos.x}px;
            top: ${sourcePos.y}px;
            opacity: 0.8;
            box-shadow: 0 0 5px rgba(0,0,0,0.3);
        `;

        container.appendChild(element);

        // Create animation
        const animationId = this.animationId++;
        const animation = {
            id: animationId,
            type: 'traveling_signal',
            element: element,
            startTime: Date.now(),
            duration: duration,
            startPos: sourcePos,
            endPos: targetPos,
            color: color,
            size: size
        };

        this.animations.set(animationId, animation);

        // Schedule fade out after reaching target
        setTimeout(() => {
            this.createFadeOut(element, this.settings.signalFadeTime);
        }, duration);

        return animationId;
    }

    createNodeFlash(nodeId, flashColor = '#ffff00', duration = null) {
        if (!window.networkViz || !window.networkViz.cy) {
            return null;
        }

        const node = window.networkViz.cy.getElementById(nodeId);
        if (node.length === 0) {
            return null;
        }

        const originalColor = node.style('background-color');
        duration = duration || this.settings.nodeFlashDuration;

        const animationId = this.animationId++;
        const animation = {
            id: animationId,
            type: 'node_flash',
            nodeId: nodeId,
            startTime: Date.now(),
            duration: duration,
            originalColor: originalColor,
            flashColor: flashColor
        };

        this.animations.set(animationId, animation);
        return animationId;
    }

    createEdgeHighlight(edgeId, highlightColor = '#ffff00', duration = null) {
        if (!window.networkViz || !window.networkViz.cy) {
            return null;
        }

        const edge = window.networkViz.cy.getElementById(edgeId);
        if (edge.length === 0) {
            return null;
        }

        const originalColor = edge.style('line-color');
        const originalWidth = parseFloat(edge.style('width'));
        duration = duration || this.settings.edgeHighlightDuration;

        const animationId = this.animationId++;
        const animation = {
            id: animationId,
            type: 'edge_highlight',
            edgeId: edgeId,
            startTime: Date.now(),
            duration: duration,
            originalColor: originalColor,
            highlightColor: highlightColor,
            originalWidth: originalWidth
        };

        this.animations.set(animationId, animation);
        return animationId;
    }

    createFadeOut(element, duration = 500) {
        if (!element || !element.parentNode) {
            return null;
        }

        const startOpacity = parseFloat(element.style.opacity) || 1;

        const animationId = this.animationId++;
        const animation = {
            id: animationId,
            type: 'fade_out',
            element: element,
            startTime: Date.now(),
            duration: duration,
            startOpacity: startOpacity
        };

        this.animations.set(animationId, animation);
        return animationId;
    }

    // Utility methods
    interpolateColor(color1, color2, factor) {
        // Simple color interpolation (assumes hex colors)
        if (color1.charAt(0) === '#') {
            color1 = color1.substring(1);
        }
        if (color2.charAt(0) === '#') {
            color2 = color2.substring(1);
        }

        const c1 = {
            r: parseInt(color1.substring(0, 2), 16),
            g: parseInt(color1.substring(2, 4), 16),
            b: parseInt(color1.substring(4, 6), 16)
        };

        const c2 = {
            r: parseInt(color2.substring(0, 2), 16),
            g: parseInt(color2.substring(2, 4), 16),
            b: parseInt(color2.substring(4, 6), 16)
        };

        const r = Math.round(c1.r + (c2.r - c1.r) * factor);
        const g = Math.round(c1.g + (c2.g - c1.g) * factor);
        const b = Math.round(c1.b + (c2.b - c1.b) * factor);

        return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
    }

    removeAnimation(id) {
        this.animations.delete(id);
    }

    clearAllAnimations() {
        this.animations.clear();

        // Remove all traveling signal elements
        const container = document.getElementById('cy');
        if (container) {
            const signals = container.querySelectorAll('.traveling-signal');
            signals.forEach(signal => signal.remove());
        }
    }

    // Public methods for triggering animations
    animateSignal(sourceNodeId, targetNodeId, eventType = 'PresynapticReleaseEvent') {
        if (!window.networkViz || !window.networkViz.cy) {
            return;
        }

        const sourceNode = window.networkViz.cy.getElementById(sourceNodeId);
        const targetNode = window.networkViz.cy.getElementById(targetNodeId);

        if (sourceNode.length === 0 || targetNode.length === 0) {
            return;
        }

        const sourcePos = sourceNode.renderedPosition();
        const targetPos = targetNode.renderedPosition();

        // Choose color and size based on event type
        let color = '#0066cc';
        let size = 8;
        let duration = 2000;

        switch (eventType) {
            case 'PresynapticReleaseEvent':
                color = '#ff8800';
                size = 10;
                break;
            case 'RetrogradeSignalEvent':
                color = '#8a2be2';
                size = 6;
                duration = 1500;
                break;
        }

        // Create traveling signal
        this.createTravelingSignal(sourcePos, targetPos, color, size, duration);

        // Flash source node
        this.createNodeFlash(sourceNodeId);

        // Highlight connecting edge
        const edge = sourceNode.edgesWith(targetNode);
        if (edge.length > 0) {
            this.createEdgeHighlight(edge.id());
        }
    }

    animateNeuronFiring(neuronId) {
        // Flash the neuron with firing color
        this.createNodeFlash(neuronId, '#ff4444', 300);

        // Highlight all outgoing edges
        if (window.networkViz && window.networkViz.cy) {
            const node = window.networkViz.cy.getElementById(neuronId);
            if (node.length > 0) {
                node.outgoers('edge').forEach(edge => {
                    this.createEdgeHighlight(edge.id(), '#ff4444', 500);
                });
            }
        }
    }

    animateSignalReceived(neuronId) {
        // Flash the neuron with receiving color
        this.createNodeFlash(neuronId, '#00ff00', 200);
    }
}

// Global animation system instance
window.animationSystem = new AnimationSystem();