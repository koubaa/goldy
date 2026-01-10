// RAG Interactive Demo Loader
// Loads WebGPU-based examples into canvas elements
//
// SLANG-ONLY ARCHITECTURE:
// - All shaders are written in Slang (single source of truth in rag::shaders)
// - Shader sources are exported from rag-web via wasm-bindgen
// - slang-wasm compiles Slang to WGSL at runtime in the browser
// - No embedded shaders or fallbacks - everything flows from Rust

// Map demo types to shader getter function names in the WASM module
const DEMO_SHADER_GETTERS = {
    'TriangleDemo': 'get_triangle_shader',
    'PlasmaDemo': 'get_plasma_shader',
    'DigitalClockDemo': 'get_digital_clock_shader',
    'MandelbrotDemo': 'get_mandelbrot_shader',
    'GradientDemo': 'get_gradient_shader',
    'TunnelDemo': 'get_tunnel_shader',
    'StarfieldDemo': 'get_starfield_shader',
    'ParticlesDemo': 'get_particles_shader',
    'SpinningCubeDemo': 'get_spinning_cube_shader'
};

// ============================================================================
// Slang Compiler (using slang-wasm)
// ============================================================================
let slangModule = null;
let slangGlobalSession = null;
let slangInitialized = false;
let slangInitPromise = null;

async function initSlangCompiler() {
    if (slangInitialized) return true;
    if (slangInitPromise) return slangInitPromise;
    
    slangInitPromise = (async () => {
        try {
            console.time('⏱️ TOTAL slang init');
            
            // Get path to slang-wasm
            const pathParts = window.location.pathname.split('/').filter(p => p);
            const htmlFile = pathParts[pathParts.length - 1]?.endsWith('.html');
            const depth = htmlFile ? pathParts.length - 1 : pathParts.length;
            const toRoot = depth > 0 ? '../'.repeat(depth) : './';
            
            const slangUrl = new URL(toRoot + 'wasm/slang-wasm.js', window.location.href).href;
            console.log("Loading slang-wasm from:", slangUrl);
            
            // Load slang-wasm as ES module (it uses import.meta)
            console.time('⏱️ 1. Import slang-wasm.js (ES module)');
            const slangWasm = await import(slangUrl);
            console.timeEnd('⏱️ 1. Import slang-wasm.js (ES module)');
            
            // slang-wasm exports a default function that returns a promise
            console.time('⏱️ 2. Initialize Slang WASM module');
            slangModule = await slangWasm.default();
            console.timeEnd('⏱️ 2. Initialize Slang WASM module');
            
            console.time('⏱️ 3. createGlobalSession');
            slangGlobalSession = slangModule.createGlobalSession();
            console.timeEnd('⏱️ 3. createGlobalSession');
            
            if (!slangGlobalSession) {
                throw new Error("Failed to create Slang global session");
            }
            
            slangInitialized = true;
            console.timeEnd('⏱️ TOTAL slang init');
            console.log("Slang compiler initialized successfully");
            return true;
        } catch (e) {
            console.error("Failed to initialize Slang compiler:", e);
            slangInitPromise = null;
            throw e; // Re-throw instead of returning false - we want hard errors
        }
    })();
    
    return slangInitPromise;
}

function compileSlangToWgsl(slangSource, vertexEntry = 'vs_main', fragmentEntry = 'fs_main') {
    if (!slangModule || !slangGlobalSession) {
        throw new Error("Slang compiler not initialized");
    }
    
    const timerId = `⏱️ compileSlangToWgsl(${vertexEntry}/${fragmentEntry})`;
    console.time(timerId);
    
    // Find WGSL target
    const targets = slangModule.getCompileTargets();
    let wgslTarget = null;
    for (let i = 0; i < targets.length; i++) {
        if (targets[i].name === 'WGSL') {
            wgslTarget = targets[i].value;
            break;
        }
    }
    
    if (wgslTarget === null) {
        throw new Error("Slang module doesn't support WGSL target");
    }
    
    // Create session for WGSL output
    const session = slangGlobalSession.createSession(wgslTarget);
    if (!session) {
        throw new Error("Failed to create Slang session");
    }
    
    try {
        // Load module from source
        const module = session.loadModuleFromSource(slangSource, "shader", "/shader.slang");
        if (!module) {
            const error = slangModule.getLastError();
            throw new Error("Failed to load Slang module: " + (error?.message || "unknown error"));
        }
        
        // Find entry points
        const vertexEntryPoint = module.findAndCheckEntryPoint(vertexEntry, 1); // STAGE_VERTEX = 1
        const fragmentEntryPoint = module.findAndCheckEntryPoint(fragmentEntry, 5); // STAGE_FRAGMENT = 5
        
        if (!vertexEntryPoint) {
            throw new Error("Failed to find vertex entry point: " + vertexEntry);
        }
        if (!fragmentEntryPoint) {
            throw new Error("Failed to find fragment entry point: " + fragmentEntry);
        }
        
        // Link and get code
        const composite = session.createCompositeComponentType([module, vertexEntryPoint, fragmentEntryPoint]);
        const linkedProgram = composite.link();
        
        const vertexWgsl = linkedProgram.getEntryPointCode(0, 0);
        const fragmentWgsl = linkedProgram.getEntryPointCode(1, 0);
        
        // Combine into single WGSL module
        const combinedWgsl = vertexWgsl + '\n' + fragmentWgsl;
        
        console.timeEnd(timerId);
        
        return { 
            vertex: vertexWgsl, 
            fragment: fragmentWgsl,
            combined: combinedWgsl
        };
    } finally {
        session.delete();
    }
}

// Cache for compiled shaders
const compiledShaderCache = {};

// Get Slang source from WASM module and compile to WGSL
async function getCompiledShader(demoType, wasmModule) {
    const getterName = DEMO_SHADER_GETTERS[demoType];
    if (!getterName) {
        throw new Error(`No shader getter mapping for demo type: ${demoType}`);
    }
    
    // Check cache
    if (compiledShaderCache[demoType]) {
        return compiledShaderCache[demoType];
    }
    
    // Get Slang source from WASM module
    const getterFn = wasmModule[getterName];
    if (!getterFn) {
        throw new Error(`Shader getter function '${getterName}' not found in WASM module`);
    }
    
    const slangSource = getterFn();
    if (!slangSource) {
        throw new Error(`No Slang shader returned for: ${demoType}`);
    }
    
    console.log(`Got Slang shader for ${demoType} from WASM (${slangSource.length} chars)`);
    
    try {
        const compiled = compileSlangToWgsl(slangSource);
        compiledShaderCache[demoType] = compiled;
        console.log(`Compiled shader for '${demoType}' from Slang to WGSL`);
        return compiled;
    } catch (e) {
        console.error(`Failed to compile shader for '${demoType}':`, e);
        throw e;
    }
}

// ============================================================================
// Demo Class
// ============================================================================
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
            const demoTimer = `⏱️ TOTAL demo init: ${this.demoClass}`;
            console.time(demoTimer);
            
            // Import the WASM module
            console.time(`⏱️ ${this.demoClass}: 1. import rag_web.js`);
            const pathParts = window.location.pathname.split('/').filter(p => p);
            const htmlFile = pathParts[pathParts.length - 1]?.endsWith('.html');
            const depth = htmlFile ? pathParts.length - 1 : pathParts.length;
            const toRoot = depth > 0 ? '../'.repeat(depth) : './';
            const wasmUrl = new URL(toRoot + 'wasm/rag_web.js', window.location.href).href;
            const wasm = await import(wasmUrl);
            console.timeEnd(`⏱️ ${this.demoClass}: 1. import rag_web.js`);
            
            console.time(`⏱️ ${this.demoClass}: 2. wasm.default() (WASM instantiation)`);
            await wasm.default();
            console.timeEnd(`⏱️ ${this.demoClass}: 2. wasm.default() (WASM instantiation)`);
            
            // Get Slang source from WASM and compile to WGSL - REQUIRED
            console.time(`⏱️ ${this.demoClass}: 3. getCompiledShader (Slang→WGSL)`);
            const wgslShader = await getCompiledShader(this.demoClass, wasm);
            console.timeEnd(`⏱️ ${this.demoClass}: 3. getCompiledShader (Slang→WGSL)`);
            
            // Create the demo using factory function with compiled WGSL
            const factoryName = 'create_' + this.demoClass.replace(/([A-Z])/g, '_$1').toLowerCase().slice(1);
            
            console.time(`⏱️ ${this.demoClass}: 4. create demo (WebGPU setup)`);
            this.demo = await wasm[factoryName](this.canvasId, wgslShader.combined);
            console.log(`Created ${this.demoClass} with Slang-compiled shader`);
            console.timeEnd(`⏱️ ${this.demoClass}: 4. create demo (WebGPU setup)`);
            
            console.timeEnd(demoTimer);
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

// ============================================================================
// Page Initialization
// ============================================================================
document.addEventListener('DOMContentLoaded', async () => {
    console.time('⏱️ PAGE TOTAL (DOMContentLoaded to all demos ready)');
    
    // Initialize Slang compiler first - REQUIRED, no fallback
    console.log("Initializing Slang compiler...");
    try {
        await initSlangCompiler();
        console.log("Slang compiler ready - demos will use Slang-compiled shaders");
    } catch (e) {
        console.error("FATAL: Slang compiler failed to initialize:", e);
        // Show error on all demo containers
        document.querySelectorAll('.rag-demo').forEach(container => {
            container.innerHTML = `<div class="rag-demo-fallback"><p>⚠️ Slang compiler failed: ${e.message}</p></div>`;
        });
        return;
    }
    
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
    
    console.timeEnd('⏱️ PAGE TOTAL (DOMContentLoaded to all demos ready)');
});

// Export for potential external use
window.RAG = {
    DEMO_SHADER_GETTERS,
    initSlangCompiler,
    compileSlangToWgsl,
    getCompiledShader,
    RagDemo
};
