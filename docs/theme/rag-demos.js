// RAG Interactive Demo Loader
// Loads WebGPU-based examples into canvas elements
// Uses slang-wasm to compile Slang shaders to WGSL at runtime

// ============================================================================
// Slang Shader Sources (embedded from rag/shaders/*.slang)
// ============================================================================
const SLANG_SHADERS = {
    vertex_color_2d: `
struct VertexInput {
    float2 position : POSITION;
    float4 color : COLOR;
};

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return input.color;
}
`,

    triangle: `
struct VertexOutput {
    float4 position : SV_Position;
    float3 color : COLOR;
};

static const float2 positions[3] = {
    float2(0.0, 0.5),
    float2(-0.5, -0.5),
    float2(0.5, -0.5)
};

static const float3 colors[3] = {
    float3(1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, 1.0)
};

[shader("vertex")]
VertexOutput vs_main(uint idx : SV_VertexID) {
    VertexOutput output;
    output.position = float4(positions[idx], 0.0, 1.0);
    output.color = colors[idx];
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return float4(input.color, 1.0);
}
`,

    plasma: `
struct VertexInput {
    float2 position : POSITION;
    float2 uv : TEXCOORD0;
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

// Uniform buffer for time - Slang generates @group(0) @binding(0)
[[vk::binding(0, 0)]]
cbuffer TimeBuffer {
    float time;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    float2 uv = input.uv * 4.0;
    float t = time;
    
    float v = sin(uv.x + t);
    v += sin(uv.y + t);
    v += sin(uv.x + uv.y + t);
    
    float cx = uv.x + 0.5 * sin(t / 3.0);
    float cy = uv.y + 0.5 * cos(t / 2.0);
    v += sin(sqrt(cx * cx + cy * cy + 1.0) + t);
    
    v = v / 2.0;
    
    float r = sin(v * 3.14159);
    float g = sin(v * 3.14159 + 2.094);
    float b = sin(v * 3.14159 + 4.188);
    
    return float4(r * 0.5 + 0.5, g * 0.5 + 0.5, b * 0.5 + 0.5, 1.0);
}
`,

    digital_clock: `
struct VertexInput {
    float2 position : POSITION;
    float4 color : COLOR;
};

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return input.color;
}
`,

    mandelbrot: `
struct VertexInput {
    float2 position : POSITION;
    float2 uv : TEXCOORD0;
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

// Uniform buffer for mandelbrot parameters
[[vk::binding(0, 0)]]
cbuffer Uniforms {
    float2 center;
    float zoom;
    float max_iter;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.uv = input.uv;
    return output;
}

float3 hsv_to_rgb(float h, float s, float v) {
    float c = v * s;
    float x = c * (1.0 - abs(fmod(h * 6.0, 2.0) - 1.0));
    float m = v - c;
    
    float3 rgb;
    int hi = int(h * 6.0) % 6;
    if (hi == 0) rgb = float3(c, x, 0.0);
    else if (hi == 1) rgb = float3(x, c, 0.0);
    else if (hi == 2) rgb = float3(0.0, c, x);
    else if (hi == 3) rgb = float3(0.0, x, c);
    else if (hi == 4) rgb = float3(x, 0.0, c);
    else rgb = float3(c, 0.0, x);
    
    return rgb + float3(m, m, m);
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    float aspect = 16.0 / 9.0;
    float2 c = center + (input.uv - 0.5) * float2(aspect, 1.0) * 3.0 / zoom;
    
    float2 z = float2(0.0, 0.0);
    float iter = 0.0;
    
    for (float i = 0.0; i < max_iter; i += 1.0) {
        if (dot(z, z) > 4.0) break;
        z = float2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        iter = i;
    }
    
    if (iter >= max_iter - 1.0) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    
    float smooth_iter = iter - log2(log2(dot(z, z))) + 4.0;
    float hue = smooth_iter / 50.0;
    float3 color = hsv_to_rgb(fmod(hue, 1.0), 0.8, 1.0);
    
    return float4(color, 1.0);
}
`,

    gradient: `
struct VertexInput {
    float2 position : POSITION;
    float2 uv : TEXCOORD0;
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[[vk::binding(0, 0)]]
cbuffer TimeBuffer {
    float time;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    float2 uv = input.uv;
    float t = time * 0.5;
    
    float angle1 = t;
    float angle2 = t * 0.7 + 1.0;
    float angle3 = t * 0.3 + 2.0;
    
    float d1 = dot(uv - 0.5, float2(cos(angle1), sin(angle1)));
    float d2 = dot(uv - 0.5, float2(cos(angle2), sin(angle2)));
    float d3 = dot(uv - 0.5, float2(cos(angle3), sin(angle3)));
    
    float3 c1 = float3(0.2, 0.4, 0.8) * (d1 + 0.5);
    float3 c2 = float3(0.8, 0.2, 0.5) * (d2 + 0.5);
    float3 c3 = float3(0.3, 0.8, 0.4) * (d3 + 0.5);
    
    float3 color = c1 + c2 + c3;
    color = color / (color + 1.0);
    
    return float4(color, 1.0);
}
`,

    tunnel: `
struct VertexInput {
    float2 position : POSITION;
    float2 uv : TEXCOORD0;
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[[vk::binding(0, 0)]]
cbuffer TimeBuffer {
    float time;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    float2 uv = (input.uv - 0.5) * 2.0;
    float t = time;
    
    float dist = length(uv);
    float angle = atan2(uv.y, uv.x);
    
    float tunnel_depth = 1.0 / (dist + 0.1);
    float tunnel_angle = angle / 3.14159 + t * 0.2;
    
    float tx = tunnel_angle * 4.0;
    float ty = tunnel_depth - t * 2.0;
    
    float checker = floor(tx) + floor(ty);
    bool is_white = fmod(checker, 2.0) == 0.0;
    
    float depth_color = 1.0 - dist * 0.5;
    float3 color;
    
    if (is_white) {
        color = float3(0.8, 0.2, 0.4) * depth_color;
    } else {
        color = float3(0.2, 0.4, 0.8) * depth_color;
    }
    
    color += float3(0.3, 0.5, 1.0) * (1.0 - dist) * (1.0 - dist);
    color *= 1.0 - dist * 0.3;
    
    return float4(color, 1.0);
}
`,

    starfield: `
struct VertexInput {
    float2 position : POSITION;
    float4 color : COLOR;
};

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return input.color;
}
`,

    particles: `
struct VertexInput {
    float2 position : POSITION;
    float4 color : COLOR;
};

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return input.color;
}
`,

    spinning_cube: `
struct VertexInput {
    float2 position : POSITION;
    float4 color : COLOR;
};

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return input.color;
}
`
};

// Map demo types to shader sources
const DEMO_SHADER_MAP = {
    'TriangleDemo': 'triangle',
    'PlasmaDemo': 'plasma',
    'DigitalClockDemo': 'digital_clock',
    'MandelbrotDemo': 'mandelbrot',
    'GradientDemo': 'gradient',
    'TunnelDemo': 'tunnel',
    'StarfieldDemo': 'starfield',
    'ParticlesDemo': 'particles',
    'SpinningCubeDemo': 'spinning_cube'
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

async function getCompiledShader(demoType) {
    const shaderName = DEMO_SHADER_MAP[demoType];
    if (!shaderName) {
        console.warn(`No shader mapping for demo type: ${demoType}`);
        return null;
    }
    
    // Check cache
    if (compiledShaderCache[shaderName]) {
        return compiledShaderCache[shaderName];
    }
    
    // Get Slang source
    const slangSource = SLANG_SHADERS[shaderName];
    if (!slangSource) {
        console.warn(`No Slang shader found for: ${shaderName}`);
        return null;
    }
    
    try {
        const compiled = compileSlangToWgsl(slangSource);
        compiledShaderCache[shaderName] = compiled;
        console.log(`Compiled shader '${shaderName}' from Slang to WGSL`);
        return compiled;
    } catch (e) {
        console.error(`Failed to compile shader '${shaderName}':`, e);
        return null;
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
            
            // Compile shader from Slang - REQUIRED
            console.time(`⏱️ ${this.demoClass}: 3. getCompiledShader (Slang→WGSL)`);
            const wgslShader = await getCompiledShader(this.demoClass);
            console.timeEnd(`⏱️ ${this.demoClass}: 3. getCompiledShader (Slang→WGSL)`);
            
            if (!wgslShader) {
                throw new Error(`No Slang shader found for ${this.demoClass}`);
            }
            
            // Create the demo using factory function with compiled WGSL
            const factoryName = 'create_' + this.demoClass.replace(/([A-Z])/g, '_$1').toLowerCase().slice(1);
            
            console.time(`⏱️ ${this.demoClass}: 4. create demo (WebGPU setup)`);
            this.demo = await wasm[factoryName](this.canvasId, wgslShader.vertex, wgslShader.fragment);
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
    SLANG_SHADERS,
    DEMO_SHADER_MAP,
    initSlangCompiler,
    compileSlangToWgsl,
    getCompiledShader,
    RagDemo
};
