const assert = require("node:assert/strict");
const fs = require("node:fs");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(`${__dirname}/../assets/coin.js`, "utf8");
const flush = () => new Promise(resolve => setImmediate(resolve));

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

function harness(options = {}) {
  const classes = new Set();
  const img = { alt: "esstec avatar", decode: async () => {} };
  const container = {
    children: [img],
    querySelector: () => img,
    replaceChildren(child) { this.children = [child]; },
    classList: { add: value => classes.add(value), remove: value => classes.delete(value) },
  };
  const motion = {
    matches: options.reduce ?? false,
    addEventListener(type, listener) { this.listener = listener; },
    change(value) { this.matches = value; this.listener(); },
  };
  const frames = new Map();
  const devices = [];
  const work = options.work ?? Promise.resolve();
  let frameId = 0;
  let adapterRequests = 0;
  const context = {
    configure() {}, unconfigure() {},
    getCurrentTexture() {
      if (options.renderError) throw new Error("render failed");
      return { createView() { return {}; } };
    },
  };
  const canvas = { style: {}, setAttribute() {}, getContext: () => options.noContext ? null : context };
  const adapter = {
    async requestDevice() {
      const lost = deferred();
      const device = {
        lost: lost.promise, lose: lost.resolve,
        destroy() { this.destroyed = true; lost.resolve({ reason: "destroyed" }); },
        addEventListener(type, listener) { this.errorListener = listener; },
        createTexture: () => ({ createView: () => ({}) }),
        createBuffer: () => ({}), createSampler: () => ({}),
        createBindGroup: () => ({}), createShaderModule: () => ({}),
        async createRenderPipelineAsync() {
          if (options.pipelineError) throw new Error("pipeline failed");
          return { getBindGroupLayout: () => ({}) };
        },
        pushErrorScope() {},
        popErrorScope: async () => options.validationError ? new Error("invalid draw") : null,
        queue: {
          copyExternalImageToTexture() {}, writeBuffer() {}, submit() {},
          onSubmittedWorkDone: () => work,
        },
        createCommandEncoder: () => ({
          beginRenderPass: () => ({
            setPipeline() {}, setBindGroup() {}, setVertexBuffer() {}, draw() {}, end() {},
          }),
          finish: () => ({}),
        }),
      };
      devices.push(device);
      return device;
    },
  };
  const gpu = {
    async requestAdapter() {
      adapterRequests++;
      if (options.adapterGate) await options.adapterGate;
      return options.noAdapter ? null : adapter;
    },
    getPreferredCanvasFormat: () => "bgra8unorm",
  };
  vm.runInNewContext(source, {
    navigator: { gpu: options.noGPU ? undefined : gpu },
    window: { devicePixelRatio: 1, matchMedia: () => motion },
    document: { getElementById: () => container, createElement: () => canvas },
    createImageBitmap: async () => ({ width: 96, height: 96, close() {} }),
    GPUTextureUsage: { TEXTURE_BINDING: 1, COPY_DST: 2, RENDER_ATTACHMENT: 4 },
    GPUBufferUsage: { VERTEX: 1, COPY_DST: 2, UNIFORM: 4 },
    performance: { now: () => 0 }, console: { warn() {} },
    requestAnimationFrame(fn) { frames.set(++frameId, fn); return frameId; },
    cancelAnimationFrame(id) { frames.delete(id); },
  });
  return {
    img, canvas, container, motion, frames, devices, classes,
    get adapterRequests() { return adapterRequests; },
    frame() {
      const [id, callback] = frames.entries().next().value;
      frames.delete(id);
      callback();
    },
  };
}

for (const option of ["noGPU", "noAdapter", "noContext", "pipelineError", "validationError", "renderError"]) {
  test(`${option} preserves the static avatar`, async () => {
    const h = harness({ [option]: true });
    await flush();
    assert.equal(h.container.children[0], h.img);
    assert.equal(h.frames.size, 0);
    assert.equal(h.classes.size, 0);
  });
}

test("waits for GPU completion before swapping the avatar", async () => {
  const work = deferred();
  const h = harness({ work: work.promise });
  await flush();
  assert.equal(h.container.children[0], h.img);
  work.resolve();
  await flush();
  assert.equal(h.container.children[0], h.canvas);
  assert.equal(h.frames.size, 1);
});

test("responds to reduced-motion changes and releases the old device", async () => {
  const h = harness({ reduce: true });
  await flush();
  assert.equal(h.adapterRequests, 0);
  h.motion.change(false);
  await flush();
  assert.equal(h.container.children[0], h.canvas);
  h.motion.change(true);
  assert.equal(h.container.children[0], h.img);
  assert.equal(h.frames.size, 0);
  assert.equal(h.devices[0].destroyed, true);
  h.motion.change(false);
  await flush();
  assert.equal(h.container.children[0], h.canvas);
});

test("a canceled initialization cannot replace a newer animation", async () => {
  const gate = deferred();
  const h = harness({ adapterGate: gate.promise });
  h.motion.change(true);
  h.motion.change(false);
  gate.resolve();
  await flush();
  assert.equal(h.devices.length, 1);
  assert.equal(h.container.children[0], h.canvas);
  assert.equal(h.frames.size, 1);
});

for (const failure of ["lost", "uncaptured", "frame"]) {
  test(`${failure} restores the avatar and stops animation`, async () => {
    const options = {};
    const h = harness(options);
    await flush();
    assert.equal(h.container.children[0], h.canvas);
    if (failure === "lost") h.devices[0].lose({ reason: "unknown" });
    if (failure === "uncaptured") h.devices[0].errorListener({ error: new Error("GPU error") });
    if (failure === "frame") {
      options.renderError = true;
      h.frame();
    }
    await flush();
    assert.equal(h.container.children[0], h.img);
    assert.equal(h.frames.size, 0);
    assert.equal(h.classes.size, 0);
    assert.equal(h.devices[0].destroyed, true);
  });
}
