// WebGPU evaluator for AZSNK1 exports: the 18-plane residual policy-value net
// with global-pooling heads (the snake AlphaZero net). The residual trunk (stem
// + tower) and the head's 1×1 convs (p1, v1) run on the GPU via a single
// padded-conv kernel — the same kernel go's GoGpu uses; only the heads differ.
// The global pooling and the small linear heads run on the CPU after one
// readback (they're tiny, and CPU keeps them exactly in step with snakeinfer's
// reference fp32 forward). Validated against snakeinfer headless in Deno — keep
// them agreeing to ~1e-3.
//
// Snake's heads are simpler than go's: policy is a 3C→C→4 MLP over the global
// pool (the four absolute headings), value a 3C→128→1 MLP then tanh — no
// placement conv and no pass head. The board is a fixed 20×20.
//
// TODO(shared-trunk): this file is the third azgpu evaluator, and the conv
// kernel + buffer/pipeline scaffolding + globalPool/linearFwd/softmaxOver are
// byte-for-byte identical to go's. go and snake genuinely share a core — both
// run the trunk + head 1×1 convs on the GPU and the global-pool MLP heads on
// the CPU — and should collapse onto one `ConvNet` base that takes a parseModel
// + heads hook (go: placement-conv + pass; snake: 4-way MLP). It is deliberately
// NOT done here: the extraction must touch the shipped go bot and be
// re-validated headless before landing, which is out of scope for getting snake
// onto the GPU. chess does NOT fit the same base — its head runs flat convs and
// its linears on the GPU, not a CPU global pool — so any shared core is go+snake,
// not all three.

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

/** Input planes for the AZSNK1 encoder (mirrors snake::encode::PLANES). */
export const PLANES = 18;
/** The four absolute headings the policy head scores (Up/Right/Down/Left). */
export const ACTIONS = 4;
/** `19.0` centers the global-pool size-scale; matches the trainer/snakeinfer. */
const POOL_SIZE_REF = 19;
/** Largest batch the pre-allocated buffers accept per `forward` call. */
export const MAX_BATCH = 32;

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

export interface SnakeModel {
  blocks: number;
  C: number;
  size: number;
  stem: Conv;
  tower: [Conv, Conv][];
  // Policy head: 1×1 conv → global pool → MLP (3C→C→4).
  p1: Conv;
  pf1: Linear;
  pf2: Linear;
  // Value head: 1×1 conv → global pool → MLP (3C→128→1) → tanh.
  v1: Conv;
  vf1: Linear;
  vf2: Linear;
}

/** Parses an AZSNK1 export (see azsnake's export and snakeinfer's reference). */
export function parseModel(buf: ArrayBuffer): SnakeModel {
  const magic = new TextDecoder().decode(buf.slice(0, 6));
  if (magic !== 'AZSNK1') throw new Error('bad magic: ' + magic);
  const dv = new DataView(buf);
  const blocks = dv.getUint32(6, true);
  const C = dv.getUint32(10, true);
  const size = dv.getUint32(14, true);
  let pos = 18;
  // The 18-byte header leaves the float region 2-byte (not 4-byte) aligned, so
  // a zero-copy `Float32Array(buf, pos, n)` view is illegal — copy each region
  // into its own aligned buffer instead. (go's 20-byte header avoids this.)
  const floats = (n: number): Float32Array<ArrayBuffer> => {
    const v = new Float32Array(buf.slice(pos, pos + n * 4));
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
  const linear = (ni: number, no: number): Linear => ({
    w: floats(no * ni),
    b: floats(no),
    ni,
    no,
  });
  const stem = conv(PLANES, C, 3);
  const tower: [Conv, Conv][] = [];
  for (let i = 0; i < blocks; i++) tower.push([conv(C, C, 3), conv(C, C, 3)]);
  const p1 = conv(C, C, 1);
  const pf1 = linear(3 * C, C);
  const pf2 = linear(C, ACTIONS);
  const v1 = conv(C, C, 1);
  const vf1 = linear(3 * C, 128);
  const vf2 = linear(128, 1);
  if (pos !== buf.byteLength) throw new Error('trailing bytes: ' + (buf.byteLength - pos));
  return { blocks, C, size, stem, tower, p1, pf1, pf2, v1, vf1, vf2 };
}

/** Global pool of a `[C, area]` plane block → `[3C]` = mean, size-scaled mean, max. */
function globalPool(x: Float32Array, base: number, C: number, area: number): Float32Array {
  const scale = Math.sqrt(area) / POOL_SIZE_REF;
  const out = new Float32Array(3 * C);
  for (let ch = 0; ch < C; ch++) {
    const p = base + ch * area;
    let sum = 0;
    let mx = -Infinity;
    for (let i = 0; i < area; i++) {
      const v = x[p + i];
      sum += v;
      if (v > mx) mx = v;
    }
    const mean = sum / area;
    out[ch] = mean;
    out[C + ch] = mean * scale;
    out[2 * C + ch] = mx;
  }
  return out;
}

function linearFwd(l: Linear, x: Float32Array, relu: boolean): Float32Array {
  const out = new Float32Array(l.no);
  for (let o = 0; o < l.no; o++) {
    let acc = l.b[o];
    const wb = o * l.ni;
    for (let i = 0; i < l.ni; i++) acc += l.w[wb + i] * x[i];
    out[o] = relu ? Math.max(acc, 0) : acc;
  }
  return out;
}

type ConvLayer = {
  u: GPUBuffer;
  bg: GPUBindGroup;
  cv: Conv;
  relu: number;
  residual: boolean;
};

export class SnakeGpu {
  model!: SnakeModel;
  private dev!: GPUDevice;
  private convPipe!: GPUComputePipeline;
  private layers: ConvLayer[] = [];
  private inBuf!: GPUBuffer;
  private polOut!: GPUBuffer;
  private vOut!: GPUBuffer;
  private stagePol!: GPUBuffer;
  private stageVal!: GPUBuffer;
  private area = 0;
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

  static async init(modelBuf: ArrayBuffer): Promise<SnakeGpu> {
    const g = new SnakeGpu();
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU unavailable');
    g.dev = await adapter.requestDevice();
    g.model = parseModel(modelBuf);
    const d = g.dev;
    const C = g.model.C;
    const area = g.model.size * g.model.size;
    g.area = area;

    g.convPipe = d.createComputePipeline({
      layout: 'auto',
      compute: { module: d.createShaderModule({ code: CONV_WGSL }), entryPoint: 'main' },
    });

    const sbuf = (arr: Float32Array<ArrayBuffer>): GPUBuffer => {
      const b = d.createBuffer({
        size: Math.max(arr.byteLength, 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      if (arr.byteLength) d.queue.writeBuffer(b, 0, arr);
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
    g.polOut = abuf(MAX_BATCH * C * area);
    g.vOut = abuf(MAX_BATCH * C * area);
    g.stagePol = d.createBuffer({
      size: MAX_BATCH * C * area * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    g.stageVal = d.createBuffer({
      size: MAX_BATCH * C * area * 4,
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
      return { u, bg, cv, relu, residual: !!resBuf };
    };

    g.layers.push(convLayer(g.model.stem, g.inBuf, actX, 1, null));
    let X = actX;
    let Y = actY;
    for (const [c1, c2] of g.model.tower) {
      g.layers.push(convLayer(c1, X, actT, 1, null));
      g.layers.push(convLayer(c2, actT, Y, 1, X));
      [X, Y] = [Y, X];
    }
    // Head 1×1 convs read the trunk output X; the rest of each head is CPU-side.
    g.layers.push(convLayer(g.model.p1, X, g.polOut, 1, null));
    g.layers.push(convLayer(g.model.v1, X, g.vOut, 1, null));
    return g;
  }

  private setBatch(B: number): void {
    if (B === this.batch) return;
    this.batch = B;
    const size = this.model.size;
    for (const L of this.layers) {
      this.dev.queue.writeBuffer(
        L.u,
        0,
        new Uint32Array([L.cv.ci, L.cv.co, L.cv.k, L.relu, L.residual ? 1 : 0, B, size, this.area]),
      );
    }
  }

  /** planes `[B × 18·area]` → logits `[B × 4]` (the headings) and values `[B]`. */
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
      pass.setPipeline(this.convPipe);
      pass.setBindGroup(0, L.bg);
      pass.dispatchWorkgroups(L.cv.co, B);
      pass.end();
    }
    const C = this.model.C;
    const blockBytes = B * C * this.area * 4;
    enc.copyBufferToBuffer(this.polOut, 0, this.stagePol, 0, blockBytes);
    enc.copyBufferToBuffer(this.vOut, 0, this.stageVal, 0, blockBytes);
    this.dev.queue.submit([enc.finish()]);

    try {
      await Promise.all([
        this.stagePol.mapAsync(GPUMapMode.READ, 0, blockBytes),
        this.stageVal.mapAsync(GPUMapMode.READ, 0, blockBytes),
      ]);
    } catch (e) {
      this.stagePol.unmap();
      this.stageVal.unmap();
      throw e;
    }
    let polBlock: Float32Array;
    let vBlock: Float32Array;
    try {
      polBlock = new Float32Array(this.stagePol.getMappedRange(0, blockBytes).slice(0));
      vBlock = new Float32Array(this.stageVal.getMappedRange(0, blockBytes).slice(0));
    } finally {
      this.stagePol.unmap();
      this.stageVal.unmap();
    }
    return this.heads(polBlock, vBlock, B);
  }

  /** CPU global-pool heads, mirroring snakeinfer's `Model::forward` exactly. */
  private heads(
    polBlock: Float32Array,
    vBlock: Float32Array,
    B: number,
  ): { logits: Float32Array; values: Float32Array } {
    const { C, pf1, pf2, vf1, vf2 } = this.model;
    const area = this.area;
    const logits = new Float32Array(B * ACTIONS);
    const values = new Float32Array(B);
    for (let b = 0; b < B; b++) {
      const base = b * C * area;
      // Policy: pool the 1×1-conv features → MLP (3C→C relu → 4).
      const polG = globalPool(polBlock, base, C, area);
      const h = linearFwd(pf1, polG, true);
      const out = linearFwd(pf2, h, false);
      logits.set(out, b * ACTIONS);
      // Value: pool the 1×1-conv features → MLP (3C→128 relu → 1) → tanh.
      const vG = globalPool(vBlock, base, C, area);
      const vh = linearFwd(vf1, vG, true);
      values[b] = Math.tanh(linearFwd(vf2, vh, false)[0]);
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
