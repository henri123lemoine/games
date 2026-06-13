// WebGPU evaluator for AZWEBGO1 exports: the 9-plane residual policy-value
// net as two compute pipelines (a zero-padded conv specialized to a
// size×size board, and a linear kernel). Go's policy head is a linear to
// size²+1 logits (placements then pass) — no channel-major rearrange, unlike
// chess. Validated against goinfer's reference fp32 forward by the
// /azero-test.html page — keep them agreeing to ~1e-3.

const CONV_WGSL = `
struct Params { c_in: u32, c_out: u32, k: u32, relu: u32, residual: u32, batch: u32, size: u32, area: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> res: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<uniform> P: Params;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let co = wg.x; let b = wg.y;
  if (b >= P.batch) { return; }
  let s = i32(P.size);
  let half = i32(P.k) / 2;
  for (var sq = li.x; sq < P.area; sq += 64u) {
    let yy = i32(sq) / s; let xx = i32(sq) % s;
    var acc = bias[co];
    for (var ci = 0u; ci < P.c_in; ci++) {
      let xbase = (b * P.c_in + ci) * P.area;
      let wbase = (co * P.c_in + ci) * P.k * P.k;
      for (var dy = -half; dy <= half; dy++) {
        let sy = yy + dy;
        if (sy < 0 || sy >= s) { continue; }
        for (var dx = -half; dx <= half; dx++) {
          let sx = xx + dx;
          if (sx < 0 || sx >= s) { continue; }
          let wi = wbase + u32(dy + half) * P.k + u32(dx + half);
          acc += w[wi] * x[xbase + u32(sy * s + sx)];
        }
      }
    }
    let oi = (b * P.c_out + co) * P.area + sq;
    if (P.residual == 1u) { acc += res[oi]; }
    if (P.relu == 1u) { acc = max(acc, 0.0); }
    y[oi] = acc;
  }
}`;

const LINEAR_WGSL = `
struct Params { n_in: u32, n_out: u32, act: u32, batch: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> P: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gi: vec3<u32>) {
  let idx = gi.x;
  if (idx >= P.n_out * P.batch) { return; }
  let b = idx / P.n_out; let o = idx % P.n_out;
  var acc = bias[o];
  let xb = b * P.n_in; let wb = o * P.n_in;
  for (var i = 0u; i < P.n_in; i++) { acc += w[wb + i] * x[xb + i]; }
  if (P.act == 1u) { acc = max(acc, 0.0); }
  if (P.act == 2u) { acc = tanh(clamp(acc, -15.0, 15.0)); }
  y[(b * P.n_out) + o] = acc;
}`;

export const PLANES = 9;
/** Largest batch the pre-allocated buffers accept per `forward` call. */
export const MAX_BATCH = 32;
/** Policy width for a board: one logit per point plus the pass. */
export function policyLen(size: number): number {
  return size * size + 1;
}

interface Conv {
  w: Float32Array<ArrayBuffer>;
  b: Float32Array<ArrayBuffer>;
  ci: number;
  co: number;
  k: number;
}

interface Linear {
  w: Float32Array<ArrayBuffer>;
  b: Float32Array<ArrayBuffer>;
  ni: number;
  no: number;
}

export interface GoModel {
  blocks: number;
  C: number;
  size: number;
  stem: Conv;
  tower: [Conv, Conv][];
  p1: Conv;
  pf: Linear;
  v1: Conv;
  vf1: Linear;
  vf2: Linear;
}

/** Parses an AZWEBGO1 export (see azgo's export and goinfer's reference). */
export function parseModel(buf: ArrayBuffer): GoModel {
  const dv = new DataView(buf);
  const magic = new TextDecoder().decode(buf.slice(0, 8));
  if (magic !== 'AZWEBGO1') throw new Error('bad magic: ' + magic);
  const blocks = dv.getUint32(8, true);
  const C = dv.getUint32(12, true);
  const size = dv.getUint32(16, true);
  const area = size * size;
  const policy = area + 1;
  let pos = 20;
  const floats = (n: number): Float32Array<ArrayBuffer> => {
    const v = new Float32Array(buf, pos, n);
    pos += n * 4;
    return v;
  };
  const conv = (ci: number, co: number, k: number): Conv => ({
    w: floats(co * ci * k * k),
    b: floats(co),
    ci,
    co,
    k,
  });
  const stem = conv(PLANES, C, 3);
  const tower: [Conv, Conv][] = [];
  for (let i = 0; i < blocks; i++) tower.push([conv(C, C, 3), conv(C, C, 3)]);
  const p1 = conv(C, 2, 1);
  const pf: Linear = { w: floats(policy * 2 * area), b: floats(policy), ni: 2 * area, no: policy };
  const v1 = conv(C, 2, 1);
  const vf1: Linear = { w: floats(128 * 2 * area), b: floats(128), ni: 2 * area, no: 128 };
  const vf2: Linear = { w: floats(128), b: floats(1), ni: 128, no: 1 };
  if (pos !== buf.byteLength) throw new Error('trailing bytes: ' + (buf.byteLength - pos));
  return { blocks, C, size, stem, tower, p1, pf, v1, vf1, vf2 };
}

type ConvLayer = {
  kind: 'conv';
  u: GPUBuffer;
  bg: GPUBindGroup;
  cv: Conv;
  relu: number;
  residual: boolean;
};
type LinLayer = { kind: 'lin'; u: GPUBuffer; bg: GPUBindGroup; l: Linear; act: number };
type Layer = ConvLayer | LinLayer;

export class GoGpu {
  model!: GoModel;
  private dev!: GPUDevice;
  private convPipe!: GPUComputePipeline;
  private linPipe!: GPUComputePipeline;
  private layers: Layer[] = [];
  private inBuf!: GPUBuffer;
  private polBuf!: GPUBuffer;
  private valBuf!: GPUBuffer;
  private stagePol!: GPUBuffer;
  private stageVal!: GPUBuffer;
  private area = 0;
  private policy = 0;
  private batch = 0;
  /** Serializes `forward`: the uniform and staging buffers are shared. */
  private queue: Promise<unknown> = Promise.resolve();

  /** The device's loss signal — callers drop cached instances on it. */
  get lost(): Promise<GPUDeviceLostInfo> {
    return this.dev.lost;
  }

  destroy(): void {
    this.dev.destroy();
  }

  static async init(modelBuf: ArrayBuffer): Promise<GoGpu> {
    const g = new GoGpu();
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU unavailable');
    g.dev = await adapter.requestDevice();
    g.model = parseModel(modelBuf);
    const d = g.dev;
    const C = g.model.C;
    const area = g.model.size * g.model.size;
    const policy = area + 1;
    g.area = area;
    g.policy = policy;

    g.convPipe = d.createComputePipeline({
      layout: 'auto',
      compute: { module: d.createShaderModule({ code: CONV_WGSL }), entryPoint: 'main' },
    });
    g.linPipe = d.createComputePipeline({
      layout: 'auto',
      compute: { module: d.createShaderModule({ code: LINEAR_WGSL }), entryPoint: 'main' },
    });

    const sbuf = (arr: Float32Array<ArrayBuffer>): GPUBuffer => {
      const b = d.createBuffer({
        size: arr.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      d.queue.writeBuffer(b, 0, arr);
      return b;
    };
    const abuf = (n: number): GPUBuffer =>
      d.createBuffer({
        size: n * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });

    const actX = abuf(MAX_BATCH * C * area);
    const actY = abuf(MAX_BATCH * C * area);
    const actT = abuf(MAX_BATCH * C * area);
    const dummy = abuf(16);
    g.inBuf = abuf(MAX_BATCH * PLANES * area);
    const pTwo = abuf(MAX_BATCH * 2 * area);
    const vTwo = abuf(MAX_BATCH * 2 * area);
    const v128 = abuf(MAX_BATCH * 128);
    g.polBuf = abuf(MAX_BATCH * policy);
    g.valBuf = abuf(MAX_BATCH);
    g.stagePol = d.createBuffer({
      size: MAX_BATCH * policy * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    g.stageVal = d.createBuffer({
      size: MAX_BATCH * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const convLayer = (
      cv: Conv,
      xb: GPUBuffer,
      yb: GPUBuffer,
      relu: number,
      resBuf: GPUBuffer | null,
    ): ConvLayer => {
      const u = d.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const bg = d.createBindGroup({
        layout: g.convPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: xb } },
          { binding: 1, resource: { buffer: sbuf(cv.w) } },
          { binding: 2, resource: { buffer: sbuf(cv.b) } },
          { binding: 3, resource: { buffer: resBuf ?? dummy } },
          { binding: 4, resource: { buffer: yb } },
          { binding: 5, resource: { buffer: u } },
        ],
      });
      return { kind: 'conv', u, bg, cv, relu, residual: !!resBuf };
    };
    const linLayer = (l: Linear, xb: GPUBuffer, yb: GPUBuffer, act: number): LinLayer => {
      const u = d.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      const bg = d.createBindGroup({
        layout: g.linPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: xb } },
          { binding: 1, resource: { buffer: sbuf(l.w) } },
          { binding: 2, resource: { buffer: sbuf(l.b) } },
          { binding: 3, resource: { buffer: yb } },
          { binding: 4, resource: { buffer: u } },
        ],
      });
      return { kind: 'lin', u, bg, l, act };
    };

    g.layers.push(convLayer(g.model.stem, g.inBuf, actX, 1, null));
    let X = actX;
    let Y = actY;
    for (const [c1, c2] of g.model.tower) {
      g.layers.push(convLayer(c1, X, actT, 1, null));
      g.layers.push(convLayer(c2, actT, Y, 1, X));
      [X, Y] = [Y, X];
    }
    // Policy head: p1 conv (C→2, relu) then a linear to size²+1 logits.
    g.layers.push(convLayer(g.model.p1, X, pTwo, 1, null));
    g.layers.push(linLayer(g.model.pf, pTwo, g.polBuf, 0));
    // Value head: v1 conv (C→2, relu) then 128-wide relu then tanh scalar.
    g.layers.push(convLayer(g.model.v1, X, vTwo, 1, null));
    g.layers.push(linLayer(g.model.vf1, vTwo, v128, 1));
    g.layers.push(linLayer(g.model.vf2, v128, g.valBuf, 2));
    return g;
  }

  private setBatch(B: number): void {
    if (B === this.batch) return;
    this.batch = B;
    for (const L of this.layers) {
      if (L.kind === 'conv') {
        this.dev.queue.writeBuffer(
          L.u,
          0,
          new Uint32Array([
            L.cv.ci,
            L.cv.co,
            L.cv.k,
            L.relu,
            L.residual ? 1 : 0,
            B,
            this.model.size,
            this.area,
          ]),
        );
      } else {
        this.dev.queue.writeBuffer(L.u, 0, new Uint32Array([L.l.ni, L.l.no, L.act, B]));
      }
    }
  }

  /** planes `[B × 9·area]` → logits `[B × (area+1)]` and values `[B]`. */
  forward(
    planes: Float32Array<ArrayBuffer>,
    B: number,
  ): Promise<{ logits: Float32Array; values: Float32Array }> {
    const run = this.queue.then(() => this.forwardNow(planes, B));
    this.queue = run.catch(() => {});
    return run;
  }

  private async forwardNow(
    planes: Float32Array<ArrayBuffer>,
    B: number,
  ): Promise<{ logits: Float32Array; values: Float32Array }> {
    if (B < 1 || B > MAX_BATCH) throw new Error(`batch ${B} out of range 1..${MAX_BATCH}`);
    this.setBatch(B);
    this.dev.queue.writeBuffer(this.inBuf, 0, planes);
    const enc = this.dev.createCommandEncoder();
    for (const L of this.layers) {
      const pass = enc.beginComputePass();
      if (L.kind === 'conv') {
        pass.setPipeline(this.convPipe);
        pass.setBindGroup(0, L.bg);
        pass.dispatchWorkgroups(L.cv.co, B);
      } else {
        pass.setPipeline(this.linPipe);
        pass.setBindGroup(0, L.bg);
        pass.dispatchWorkgroups(Math.ceil((L.l.no * B) / 64));
      }
      pass.end();
    }
    enc.copyBufferToBuffer(this.polBuf, 0, this.stagePol, 0, B * this.policy * 4);
    enc.copyBufferToBuffer(this.valBuf, 0, this.stageVal, 0, B * 4);
    this.dev.queue.submit([enc.finish()]);

    try {
      await Promise.all([
        this.stagePol.mapAsync(GPUMapMode.READ, 0, B * this.policy * 4),
        this.stageVal.mapAsync(GPUMapMode.READ, 0, B * 4),
      ]);
    } catch (e) {
      // unmap aborts a pending map, so a failed pair never bricks the buffers.
      this.stagePol.unmap();
      this.stageVal.unmap();
      throw e;
    }
    let logits: Float32Array;
    let values: Float32Array;
    try {
      logits = new Float32Array(this.stagePol.getMappedRange(0, B * this.policy * 4).slice(0));
      values = new Float32Array(this.stageVal.getMappedRange(0, B * 4).slice(0));
    } finally {
      this.stagePol.unmap();
      this.stageVal.unmap();
    }
    return { logits, values };
  }
}

/** Softmax restricted to the legal `support` indices into `logits`. */
export function softmaxOver(
  logits: Float32Array,
  support: ArrayLike<number>,
  base = 0,
): number[] {
  const raw = Array.from(support, (s) => logits[base + s]);
  const mx = Math.max(...raw);
  const ex = raw.map((v) => Math.exp(v - mx));
  const sum = ex.reduce((a, v) => a + v, 0);
  return ex.map((v) => v / sum);
}
