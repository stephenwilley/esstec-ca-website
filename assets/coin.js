/**
 * coin.js — WebGPU rotating coin
 *
 * Replaces the static avatar image with a 3D spinning coin whose faces
 * show the avatar texture. Falls back silently to the static image when
 * the browser does not support WebGPU.
 */

// ────────────────────────────────────────────────────────────────
//  Configuration
// ────────────────────────────────────────────────────────────────

const COIN_SEGMENTS  = 64;           // smoothness of the circle
const COIN_THICKNESS = 0.25;         // depth, relative to radius 1
const ROTATION_SPEED = 1.2;          // radians per second
const CANVAS_CSS_PX  = 128;          // matches the container div
const PROJ_MATRIX    = mat4Ortho(0.96, -1.5, 1.5);  // orthographic, never changes

// ────────────────────────────────────────────────────────────────
//  WGSL shader source
// ────────────────────────────────────────────────────────────────

const SHADER_CODE = /* wgsl */ `

// Two matrices: the full model-view-projection and just the model
// (rotation only) so we can transform normals for lighting.
struct Uniforms {
  mvp   : mat4x4f,
  model : mat4x4f,
};

@group(0) @binding(0) var<uniform> uniforms      : Uniforms;
@group(0) @binding(1) var          avatarSampler  : sampler;
@group(0) @binding(2) var          avatarTexture  : texture_2d<f32>;

// ── Vertex stage ──────────────────────────────────────────────

struct VertexIn {
  @location(0) position : vec3f,
  @location(1) normal   : vec3f,
  @location(2) uv       : vec2f,
  @location(3) isFace   : f32,     // 1 = coin face, 0 = rim edge
};

struct VertexOut {
  @builtin(position) clipPos : vec4f,
  @location(0) worldNormal   : vec3f,
  @location(1) uv            : vec2f,
  @location(2) isFace        : f32,
};

@vertex fn vertexMain(v : VertexIn) -> VertexOut {
  var out : VertexOut;
  out.clipPos     = uniforms.mvp * vec4f(v.position, 1.0);
  out.worldNormal = (uniforms.model * vec4f(v.normal, 0.0)).xyz;
  out.uv          = v.uv;
  out.isFace      = v.isFace;
  return out;
}

// ── Fragment stage ────────────────────────────────────────────

@fragment fn fragmentMain(f : VertexOut) -> @location(0) vec4f {
  let n        = normalize(f.worldNormal);
  let lightDir = normalize(vec3f(0.3, 0.5, 1.0));
  let diffuse  = max(dot(n, lightDir), 0.0);
  let ambient  = 0.35;
  let lighting = ambient + diffuse * 0.65;

  // Scale UV outward from centre so the image fills only the inner ~80% of
  // the disc radius; the outer ring clamps to the texture edge (transparent)
  // and composites to white, creating a border.
  let borderScale = 1.2;
  let scaledUv = vec2f(0.5) + (f.uv - vec2f(0.5)) * borderScale;

  // textureSample requires uniform control flow, so sample unconditionally.
  let texel = textureSample(avatarTexture, avatarSampler, scaledUv);

  if (f.isFace > 0.5) {
    // ── Coin face: composite texture over white, then apply lighting ──
    let composited = texel.rgb * texel.a + vec3f(1.0) * (1.0 - texel.a);
    return vec4f(composited * lighting, 1.0);

  } else {
    // ── Coin rim: solid white ──
    return vec4f(vec3f(lighting), 1.0);
  }
}
`;

// ────────────────────────────────────────────────────────────────
//  Entry point
// ────────────────────────────────────────────────────────────────

initCoin();

function initCoin() {
  if (!navigator.gpu) return;

  const container = document.getElementById("avatar-container");
  const avatarImg = container?.querySelector("img");
  if (!avatarImg) return;

  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let stop = () => {};

  function updateMotion() {
    stop();
    stop = motion.matches ? () => {} : startCoin(container, avatarImg);
  }

  motion.addEventListener("change", updateMotion);
  updateMotion();
}

function startCoin(container, avatarImg) {
  let stopped = false;
  let device;
  let context;
  let stopAnimation = () => {};

  function stop() {
    if (stopped) return;
    stopped = true;
    stopAnimation();
    container.replaceChildren(avatarImg);
    container.classList.remove("is-animated");
    context?.unconfigure();
    device?.destroy();
  }

  function fail(err) {
    if (stopped) return;
    console.warn("WebGPU coin failed, using static image fallback:", err);
    stop();
  }

  async function initialize() {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter || stopped) return;

    device = await adapter.requestDevice();
    if (stopped) {
      device.destroy();
      return;
    }

    device.lost.then(fail);
    device.addEventListener("uncapturederror", event => fail(event.error));

    await avatarImg.decode();
    if (stopped) return;
    const avatarTexture               = await loadAvatarTexture(device, avatarImg);
    if (stopped) return;
    const setup = setupCanvas(device);
    context = setup.context;
    const { canvas, format } = setup;
    const { vertexBuffer, count }     = buildCoinGeometry(device);
    const pipeline                    = await createPipeline(device, format);
    if (stopped) return;
    const { uniformBuffer, bindGroup }= createBindings(device, pipeline, avatarTexture);
    const depthTexture                = createDepthTexture(device, canvas);

    stopAnimation = runAnimationLoop(
      device, context, pipeline, bindGroup,
      vertexBuffer, count, uniformBuffer, depthTexture,
      () => {
        if (stopped) return;
        canvas.setAttribute("role", "img");
        canvas.setAttribute("aria-label", avatarImg.alt);
        container.replaceChildren(canvas);
        container.classList.add("is-animated");
      },
      fail,
    );
  }

  initialize().catch(fail);
  return stop;
}

// ────────────────────────────────────────────────────────────────
//  Canvas & WebGPU context
// ────────────────────────────────────────────────────────────────

function setupCanvas(device) {
  const dpr       = window.devicePixelRatio || 1;
  const pixelSize = Math.round(CANVAS_CSS_PX * dpr);

  const canvas    = document.createElement("canvas");
  canvas.width    = pixelSize;
  canvas.height   = pixelSize;
  canvas.style.width  = CANVAS_CSS_PX + "px";
  canvas.style.height = CANVAS_CSS_PX + "px";

  // Glow that follows the coin silhouette as it spins.
  canvas.style.filter =
    "drop-shadow(0 0 14px rgba(255,255,255,0.65)) " +
    "drop-shadow(0 0 5px  rgba(255,255,255,0.30))";

  const format  = navigator.gpu.getPreferredCanvasFormat();
  const context = canvas.getContext("webgpu");
  if (!context) throw new Error("WebGPU canvas context is unavailable");
  context.configure({ device, format, alphaMode: "premultiplied" });

  return { canvas, context, format };
}

// ────────────────────────────────────────────────────────────────
//  Avatar texture
// ────────────────────────────────────────────────────────────────

async function loadAvatarTexture(device, imgElement) {
  const bitmap  = await createImageBitmap(imgElement);
  const texture = device.createTexture({
    size:   [bitmap.width, bitmap.height],
    format: "rgba8unorm",
    usage:  GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST        |
            GPUTextureUsage.RENDER_ATTACHMENT,
  });
  try {
    device.queue.copyExternalImageToTexture(
      { source: bitmap },
      { texture },
      [bitmap.width, bitmap.height],
    );
  } finally {
    bitmap.close();
  }
  return texture;
}

// ────────────────────────────────────────────────────────────────
//  Coin geometry
// ────────────────────────────────────────────────────────────────
//
//  The coin is a thin cylinder of radius 1 centred at the origin.
//  Each vertex carries 9 floats:
//
//    position (3) · normal (3) · uv (2) · isFace (1)
//
//  isFace = 1  →  the vertex belongs to a face  (textured)
//  isFace = 0  →  the vertex belongs to the rim  (white)

function buildCoinGeometry(device) {
  const data    = [];
  const half    = COIN_THICKNESS / 2;
  const step    = (Math.PI * 2) / COIN_SEGMENTS;

  // Helper: push one vertex (9 floats) into the array.
  function vert(px, py, pz, nx, ny, nz, u, v, face) {
    data.push(px, py, pz, nx, ny, nz, u, v, face);
  }

  for (let i = 0; i < COIN_SEGMENTS; i++) {
    const a0 = i * step;
    const a1 = (i + 1) * step;
    const c0 = Math.cos(a0), s0 = Math.sin(a0);
    const c1 = Math.cos(a1), s1 = Math.sin(a1);

    // ── Front face (normal +Z, triangle-fan wedge) ──
    vert(  0,  0,  half,  0, 0, 1,  0.5,            0.5,            1);
    vert( c0, s0,  half,  0, 0, 1,  0.5 + c0 * 0.5, 0.5 - s0 * 0.5, 1);
    vert( c1, s1,  half,  0, 0, 1,  0.5 + c1 * 0.5, 0.5 - s1 * 0.5, 1);

    // ── Back face (normal −Z, winding reversed so it faces −Z) ──
    // UVs are mirrored in U so the image reads correctly from behind.
    vert(  0,  0, -half,  0, 0, -1,  0.5,            0.5,            1);
    vert( c1, s1, -half,  0, 0, -1,  0.5 - c1 * 0.5, 0.5 - s1 * 0.5, 1);
    vert( c0, s0, -half,  0, 0, -1,  0.5 - c0 * 0.5, 0.5 - s0 * 0.5, 1);

    // ── Rim (two triangles per quad, normals point radially out) ──
    vert( c0, s0,  half,  c0, s0, 0,  0, 0, 0);
    vert( c0, s0, -half,  c0, s0, 0,  0, 0, 0);
    vert( c1, s1,  half,  c1, s1, 0,  0, 0, 0);

    vert( c1, s1,  half,  c1, s1, 0,  0, 0, 0);
    vert( c0, s0, -half,  c0, s0, 0,  0, 0, 0);
    vert( c1, s1, -half,  c1, s1, 0,  0, 0, 0);
  }

  const floats = new Float32Array(data);
  const vertexBuffer = device.createBuffer({
    size:  floats.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, floats);

  // 4 triangles per segment × 3 verts = 12 verts (front 3 + back 3 + rim 6).
  const count = COIN_SEGMENTS * 12;
  return { vertexBuffer, count };
}

// ────────────────────────────────────────────────────────────────
//  Render pipeline
// ────────────────────────────────────────────────────────────────

function createPipeline(device, format) {
  const module = device.createShaderModule({ code: SHADER_CODE });

  return device.createRenderPipelineAsync({
    layout: "auto",
    vertex: {
      module,
      entryPoint: "vertexMain",
      buffers: [{
        arrayStride: 9 * 4,   // 9 floats × 4 bytes
        attributes: [
          { shaderLocation: 0, offset:  0, format: "float32x3" },  // position
          { shaderLocation: 1, offset: 12, format: "float32x3" },  // normal
          { shaderLocation: 2, offset: 24, format: "float32x2" },  // uv
          { shaderLocation: 3, offset: 32, format: "float32"   },  // isFace
        ],
      }],
    },
    fragment: {
      module,
      entryPoint: "fragmentMain",
      targets: [{ format }],
    },
    primitive: {
      topology: "triangle-list",
      cullMode: "back",        // back-face culling keeps only outward faces
    },
    depthStencil: {
      format:            "depth24plus",
      depthWriteEnabled: true,
      depthCompare:      "less",
    },
  });
}

// ────────────────────────────────────────────────────────────────
//  Bind group (uniforms + texture)
// ────────────────────────────────────────────────────────────────

function createBindings(device, pipeline, avatarTexture) {
  // 2 × mat4x4f = 128 bytes
  const uniformBuffer = device.createBuffer({
    size:  128,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: sampler },
      { binding: 2, resource: avatarTexture.createView() },
    ],
  });

  return { uniformBuffer, bindGroup };
}

// ────────────────────────────────────────────────────────────────
//  Depth buffer
// ────────────────────────────────────────────────────────────────

function createDepthTexture(device, canvas) {
  return device.createTexture({
    size:   [canvas.width, canvas.height],
    format: "depth24plus",
    usage:  GPUTextureUsage.RENDER_ATTACHMENT,
  });
}

// ────────────────────────────────────────────────────────────────
//  Uniform updates (called every frame)
// ────────────────────────────────────────────────────────────────

function writeUniforms(device, uniformBuffer, elapsedSec) {
  const model = mat4RotateY(elapsedSec * ROTATION_SPEED);
  const mvp   = mat4Multiply(PROJ_MATRIX, model);

  device.queue.writeBuffer(uniformBuffer,  0, mvp);
  device.queue.writeBuffer(uniformBuffer, 64, model);
}

// ────────────────────────────────────────────────────────────────
//  Animation loop
// ────────────────────────────────────────────────────────────────

function runAnimationLoop(
  device, context, pipeline, bindGroup,
  vertexBuffer, vertexCount, uniformBuffer, depthTexture,
  onReady, onError,
) {
  const t0 = performance.now() / 1000;
  let stopped = false;
  let frameId;

  function render() {
    const elapsed = performance.now() / 1000 - t0;
    writeUniforms(device, uniformBuffer, elapsed);

    const colorView = context.getCurrentTexture().createView();
    const depthView = depthTexture.createView();

    const encoder = device.createCommandEncoder();
    const pass    = encoder.beginRenderPass({
      colorAttachments: [{
        view:       colorView,
        clearValue: { r: 0, g: 0, b: 0, a: 0 },   // transparent
        loadOp:     "clear",
        storeOp:    "store",
      }],
      depthStencilAttachment: {
        view:            depthView,
        depthClearValue: 1.0,
        depthLoadOp:     "clear",
        depthStoreOp:    "store",
      },
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.draw(vertexCount);
    pass.end();

    device.queue.submit([encoder.finish()]);
  }

  function frame() {
    if (stopped) return;
    try {
      render();
      frameId = requestAnimationFrame(frame);
    } catch (err) {
      onError(err);
    }
  }

  async function firstFrame() {
    // Validation errors are asynchronous: check the first draw before
    // replacing the fallback, then monitor uncaptured errors while spinning.
    device.pushErrorScope("validation");
    try {
      render();
    } finally {
      const error = await device.popErrorScope();
      if (error) throw error;
    }
    await device.queue.onSubmittedWorkDone();
    if (stopped) return;
    onReady();
    frameId = requestAnimationFrame(frame);
  }

  firstFrame().catch(onError);
  return () => {
    stopped = true;
    cancelAnimationFrame(frameId);
  };
}

// ────────────────────────────────────────────────────────────────
//  Matrix helpers  (column-major Float32Arrays for WebGPU)
// ────────────────────────────────────────────────────────────────

function mat4RotateY(angle) {
  const c = Math.cos(angle), s = Math.sin(angle);
  // Column-major layout:
  //   col0        col1    col2        col3
  return new Float32Array([
     c,  0, -s,  0,
     0,  1,  0,  0,
     s,  0,  c,  0,
     0,  0,  0,  1,
  ]);
}

function mat4Ortho(scale, zNear, zFar) {
  // XY scaled uniformly; Z mapped from [zNear, zFar] → [0, 1].
  // Camera looks along −Z, so larger world-Z = closer = smaller clip-Z.
  const zRange = zFar - zNear;
  return new Float32Array([
    scale,  0,      0,            0,
    0,      scale,  0,            0,
    0,      0,     -1 / zRange,   0,
    0,      0,      zFar / zRange, 1,
  ]);
}

function mat4Multiply(a, b) {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      out[col * 4 + row] =
        a[row]         * b[col * 4 + 0] +
        a[1 * 4 + row] * b[col * 4 + 1] +
        a[2 * 4 + row] * b[col * 4 + 2] +
        a[3 * 4 + row] * b[col * 4 + 3];
    }
  }
  return out;
}
