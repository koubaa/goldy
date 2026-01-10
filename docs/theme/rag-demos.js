// RAG Interactive Demo Loader
// Loads WebGPU-based examples into canvas elements

class RagDemo {
    constructor(canvasId, wasmModule, demoClass) {
        this.canvasId = canvasId;
        this.wasmModule = wasmModule;
        this.demoClass = demoClass;
        this.demo = null;
        this.animationId = null;
        this.isRunning = false;
    }

    async init() {
        const canvas = document.getElementById(this.canvasId);
        if (!canvas) {
            console.error(`Canvas '${this.canvasId}' not found`);
            return false;
        }

        // Check WebGPU support
        if (!navigator.gpu) {
            this.showFallback(canvas, "WebGPU not supported in this browser. Try Chrome 113+ or Edge 113+.");
            return false;
        }

        try {
            // Import the WASM module
            // Get base URL from the page, handling both root and subpath hosting
            const pathParts = window.location.pathname.split('/').filter(p => p);
            const htmlFile = pathParts[pathParts.length - 1]?.endsWith('.html');
            const depth = htmlFile ? pathParts.length - 1 : pathParts.length;
            const toRoot = depth > 0 ? '../'.repeat(depth) : './';
            const wasmUrl = new URL(toRoot + 'wasm/rag_web.js', window.location.href).href;
            const wasm = await import(wasmUrl);
            await wasm.default();
            
            // Create the demo using factory function
            const factoryName = 'create_' + this.demoClass.replace(/([A-Z])/g, '_$1').toLowerCase().slice(1);
            this.demo = await wasm[factoryName](this.canvasId);
            return true;
        } catch (e) {
            console.error("Failed to initialize demo:", e);
            this.showFallback(canvas, `Failed to load: ${e.message}`);
            return false;
        }
    }

    showFallback(canvas, message) {
        const container = canvas.parentElement;
        const fallback = document.createElement('div');
        fallback.className = 'rag-demo-fallback';
        fallback.innerHTML = `
            <div class="fallback-message">
                <p>⚠️ ${message}</p>
                <p>Run locally: <code>cargo run --example plasma --release</code></p>
            </div>
        `;
        container.replaceChild(fallback, canvas);
    }

    start() {
        if (!this.demo || this.isRunning) return;
        this.isRunning = true;
        this.animate();
    }

    stop() {
        this.isRunning = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }

    animate() {
        if (!this.isRunning) return;
        
        try {
            this.demo.render();
        } catch (e) {
            console.error("Render error:", e);
            this.stop();
            return;
        }
        
        this.animationId = requestAnimationFrame(() => this.animate());
    }
}

// Auto-initialize demos when the page loads
document.addEventListener('DOMContentLoaded', async () => {
    // Find all demo containers
    const demoContainers = document.querySelectorAll('.rag-demo');
    
    for (const container of demoContainers) {
        const canvasId = container.dataset.canvas;
        const demoType = container.dataset.demo;
        
        if (!canvasId || !demoType) continue;
        
        const demo = new RagDemo(canvasId, './wasm/rag_web.js', demoType);
        const success = await demo.init();
        
        if (success) {
            const canvas = document.getElementById(canvasId);
            canvas.tabIndex = 0; // Make canvas focusable
            
            // Add play/pause button
            const controls = document.createElement('div');
            controls.className = 'rag-demo-controls';
            controls.innerHTML = `
                <button class="play-btn" title="Play/Pause">▶️</button>
            `;
            container.appendChild(controls);
            
            const playBtn = controls.querySelector('.play-btn');
            playBtn.addEventListener('click', () => {
                if (demo.isRunning) {
                    demo.stop();
                    playBtn.textContent = '▶️';
                } else {
                    demo.start();
                    playBtn.textContent = '⏸️';
                }
            });
            
            // Keyboard controls for Mandelbrot
            if (demoType === 'MandelbrotDemo') {
                canvas.addEventListener('keydown', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    switch(e.key) {
                        case 'ArrowUp': demo.demo.pan(0, -0.1); break;
                        case 'ArrowDown': demo.demo.pan(0, 0.1); break;
                        case 'ArrowLeft': demo.demo.pan(-0.1, 0); break;
                        case 'ArrowRight': demo.demo.pan(0.1, 0); break;
                        case '+': case '=': demo.demo.zoom_in(); break;
                        case '-': demo.demo.zoom_out(); break;
                        case 'r': case 'R': demo.demo.reset(); break;
                    }
                    return false;
                });
                const hint = document.createElement('div');
                hint.className = 'rag-demo-hint';
                hint.textContent = 'Click canvas, then: Arrows=pan, +/-=zoom, R=reset';
                container.appendChild(hint);
            }
            
            // Keyboard controls for Digital Clock (timer)
            if (demoType === 'DigitalClockDemo') {
                canvas.addEventListener('keydown', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    switch(e.key) {
                        case ' ': demo.demo.toggle_pause(); break;
                        case 'c': case 'C': demo.demo.change_color(); break;
                    }
                    return false;
                });
                canvas.addEventListener('click', () => {
                    demo.demo.change_color();
                });
                const hint = document.createElement('div');
                hint.className = 'rag-demo-hint';
                hint.textContent = 'Click canvas, then: Space=pause/resume, C or Click=change color';
                container.appendChild(hint);
            }
            
            // Keyboard controls for Particles (rain/snow toggle)
            if (demoType === 'ParticlesDemo') {
                canvas.addEventListener('keydown', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    if (e.key === ' ') {
                        demo.demo.toggle_mode();
                    }
                    return false;
                });
                const hint = document.createElement('div');
                hint.className = 'rag-demo-hint';
                hint.textContent = 'Click canvas, then: Space=toggle rain/snow';
                container.appendChild(hint);
            }
            
            // Auto-start
            demo.start();
            playBtn.textContent = '⏸️';
        }
    }
});
