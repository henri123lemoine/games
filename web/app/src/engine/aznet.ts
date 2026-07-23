// Shared WebGPU evaluator for the AlphaZero resnet family — the one driver the
// chess, go, and snake frontends run their nets through. All three are the same
// network shape (a 3×3 conv stem, a residual tower of paired 3×3 convs with BN
// folded into the conv weights, then policy + value heads); they differ only in
// the head topology and the board size. This module owns everything that is the
// same across them: the padded-conv WGSL kernel (and a dense kernel for chess's
// on-GPU head), the buffer/pipeline scaffolding and the conv→tower→heads
// dispatch loop, the binary parser, and the CPU-side global-pool / linear /
// softmax math the pooled heads use after one readback.
//
// Two head families subclass the base trunk driver:
//   * `PooledHeadNet` (go, snake): the residual trunk and each head's 1×1 conv
//     run on the GPU; the global pooling and the small linear heads run on the
//     CPU after one readback (they're tiny, and CPU keeps them exactly in step
//     with nn-infer's reference fp32 forward). go and snake plug in only their
//     `heads()` math. The conv weights are board-size-agnostic, so a pooled net
//     forwards at any size ≤ the buffers it was sized for.
//   * `FlatHeadNet` (chess): the policy head is a 1×1 conv stack and the value
//     head a small conv then a pair of linears, all on the GPU via the dense
//     kernel; only the final channel-major→square-major reshuffle is CPU-side.
//
// The parser reads the unified AZNET1 header (see nn-infer/src/format.rs); it is
// the only export format the arcade ships.

/** `19.0` centers the global-pool size-scale; matches the trainers/reference. */
const POOL_SIZE_REF = 19;
/** Largest batch the pre-allocated buffers accept per `forward` call. */
export const MAX_BATCH = 32;
/** Chess flat value head: `v1` reduces the trunk to this many channels before
 * the dense MLP, and the MLP's hidden width. Fixed by the chess net. */
export const CHESS_VALUE_CHANNELS = 8;
export const CHESS_VALUE_HIDDEN = 256;
/** Go/snake global-pool value head's hidden width. */
export const POOL_VALUE_HIDDEN = 128;

// The padded-conv kernel, shared by every net: one workgroup per (output
// channel, batch item), 64 threads striding over the board squares. The board
// `size`/`area` are uniforms, so the same kernel runs an 8×8 chess board or a
// 19×19 go board, and a pooled net forwards at any size with one pipeline.
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

// The dense kernel chess's value head uses: one thread per (batch, output)
// element, optional relu / tanh activation.
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

export interface Conv {
  w: Float32Array<ArrayBuffer>;
  b: Float32Array<ArrayBuffer>;
  ci: number;
  co: number;
  k: number;
}

export interface Linear {
  w: Float32Array<ArrayBuffer>;
  b: Float32Array<ArrayBuffer>;
  ni: number;
  no: number;
}

/** The policy/value head topology, read from the header (never a game
 * identity). Mirrors nn-infer's `HeadKind`. */
export enum HeadKind {
  FlatConv = 0,
  GlobalPoolSpatial = 1,
  GlobalPoolDense = 2,
}

/** Optional appended heads, one flag bit each. */
const FLAG_OWNERSHIP = 1;
const FLAG_VALUE_SEATS = 2;

/** The architecture header — everything a parser or the driver needs to lay out
 * the net. Mirrors nn-infer's `Arch`. */
export interface Arch {
  blocks: number;
  /** Channels (named `C` to match the trunk/head math). */
  C: number;
  planes: number;
  size: number;
  scalars: number;
  head: HeadKind;
  /** Flat/dense policy width; `0` for spatial (whose width is `size²+1`). */
  policyLen: number;
  ownership: boolean;
  /** 1 for mover-scalar nets; >1 for raw absolute-seat value logits. */
  valueSeats: number;
}

/** Parses the unified AZNET1 header into the `Arch` plus the byte offset where
 * the weight stream begins. */
export function parseArch(buf: ArrayBuffer): { arch: Arch; body: number } {
  const dv = new DataView(buf);
  const magic = new TextDecoder().decode(buf.slice(0, 8));
  const u32 = (off: number): number => dv.getUint32(off, true);

  if (magic !== 'AZNET1\0\0') throw new Error('bad magic: ' + magic);
  // Fields are ten u32s at byte 8 + 4·i: version, blocks, channels, planes,
  // size, scalars, head_kind, policy_len, flags, reserved (see format.rs).
  const version = u32(8);
  if (version !== 1) throw new Error('unsupported AZNET1 version ' + version);
  const head = u32(32);
  if (head > 2) throw new Error('unknown head_kind ' + head);
  const flags = u32(40);
  if (flags & ~(FLAG_OWNERSHIP | FLAG_VALUE_SEATS))
    throw new Error('unknown head flags ' + flags.toString(16));
  const reserved = u32(44);
  const valueSeats = flags & FLAG_VALUE_SEATS ? reserved : 1;
  if (flags & FLAG_VALUE_SEATS) {
    if (valueSeats < 2 || valueSeats > 8) throw new Error('invalid value seat count ' + valueSeats);
  } else if (reserved !== 0) throw new Error('nonzero reserved header word');
  return {
    arch: {
      blocks: u32(12),
      C: u32(16),
      planes: u32(20),
      size: u32(24),
      scalars: u32(28),
      head: head as HeadKind,
      policyLen: u32(36),
      ownership: !!(flags & FLAG_OWNERSHIP),
      valueSeats,
    },
    body: 48,
  };
}

/** Sequential reader over an export's float region — the TS mirror of nn-infer's
 * `Reader{floats,conv,linear}`. `floats` copies (rather than viewing) so a
 * header that leaves the region misaligned is still safe. */
export class Reader {
  pos: number;
  constructor(
    private buf: ArrayBuffer,
    start: number,
  ) {
    this.pos = start;
  }

  floats(n: number): Float32Array<ArrayBuffer> {
    const v = new Float32Array(this.buf.slice(this.pos, this.pos + n * 4));
    this.pos += n * 4;
    return v;
  }

  conv(ci: number, co: number, k: number): Conv {
    return { w: this.floats(co * ci * k * k), b: this.floats(co), ci, co, k };
  }

  /** A bias-less conv (go's placement / ownership heads): weights only, zero bias. */
  convNoBias(ci: number, co: number, k: number): Conv {
    return { w: this.floats(co * ci * k * k), b: new Float32Array(co), ci, co, k };
  }

  linear(ni: number, no: number): Linear {
    return { w: this.floats(no * ni), b: this.floats(no), ni, no };
  }

  /** Asserts the float region was consumed exactly — the no-trailing-bytes
   * integrity check every export format carries. */
  done(): void {
    if (this.pos !== this.buf.byteLength)
      throw new Error('trailing bytes: ' + (this.buf.byteLength - this.pos));
  }
}

/** The parsed trunk every net shares. */
export interface Trunk {
  arch: Arch;
  stem: Conv;
  tower: [Conv, Conv][];
}

/** Reads the stem + residual tower (the trunk), leaving the reader at the first
 * head weight. The layer order matches nn-infer's `Net::parse`. */
export function parseTrunk(buf: ArrayBuffer): { trunk: Trunk; reader: Reader } {
  const { arch, body } = parseArch(buf);
  const r = new Reader(buf, body);
  const stem = r.conv(arch.planes, arch.C, 3);
  const tower: [Conv, Conv][] = [];
  for (let i = 0; i < arch.blocks; i++) tower.push([r.conv(arch.C, arch.C, 3), r.conv(arch.C, arch.C, 3)]);
  return { trunk: { arch, stem, tower }, reader: r };
}

/** Global pool of a `[C, area]` plane block → `[3C]` = mean, size-scaled mean,
 * max — the reduction the pooled go/snake heads share. */
export function globalPool(x: Float32Array, base: number, C: number, area: number): Float32Array {
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

/** A single dense layer on the CPU, with optional relu — the pooled heads' MLP. */
export function linearFwd(l: Linear, x: Float32Array, relu: boolean): Float32Array {
  const out = new Float32Array(l.no);
  for (let o = 0; o < l.no; o++) {
    let acc = l.b[o];
    const wb = o * l.ni;
    for (let i = 0; i < l.ni; i++) acc += l.w[wb + i] * x[i];
    out[o] = relu ? Math.max(acc, 0) : acc;
  }
  return out;
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

interface ConvLayer {
  kind: 'conv';
  u: GPUBuffer;
  bg: GPUBindGroup;
  cv: Conv;
  relu: number;
  residual: boolean;
}
interface LinLayer {
  kind: 'lin';
  u: GPUBuffer;
  bg: GPUBindGroup;
  l: Linear;
  act: number;
}
type Layer = ConvLayer | LinLayer;

export interface Eval {
  logits: Float32Array;
  values: Float32Array;
}

/** Base trunk driver: owns the device, the conv and dense pipelines, the
 * activation buffers, and the dispatch loop. A subclass parses its export, lays
 * out its layers in `build`, and turns the GPU readback into `{logits, values}`
 * in `finish`. The two kernels, the buffer plumbing, and the serialized
 * `forward` are all here, shared. */
export abstract class AznetBase {
  arch!: Arch;
  protected dev!: GPUDevice;
  protected convPipe!: GPUComputePipeline;
  protected linPipe!: GPUComputePipeline;
  protected layers: Layer[] = [];
  protected inBuf!: GPUBuffer;
  protected dummy!: GPUBuffer;
  private batch = 0;
  private curSize = 0;
  /** Serializes `forward`: the uniform and staging buffers are shared. */
  private queue: Promise<unknown> = Promise.resolve();

  /** A `[blocks]×[C]` summary the harnesses log; back-compat with `.model`. */
  get model(): Arch {
    return this.arch;
  }

  /** The device's loss signal — callers drop cached instances on it. */
  get lost(): Promise<GPUDeviceLostInfo> {
    return this.dev.lost;
  }

  destroy(): void {
    this.dev.destroy();
  }

  /** Parse the export, returning the trunk + a reader positioned at the heads. */
  protected abstract parse(buf: ArrayBuffer): { trunk: Trunk; reader: Reader };
  /** Allocate buffers and push the conv/linear layers, given the parsed trunk. */
  protected abstract build(trunk: Trunk, reader: Reader): void;
  /** Read the GPU outputs back and produce the final logits + values. */
  protected abstract finish(enc: GPUCommandEncoder, B: number, area: number): Promise<Eval>;

  protected async boot(modelBuf: ArrayBuffer): Promise<void> {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU unavailable');
    this.dev = await adapter.requestDevice();
    const { trunk, reader } = this.parse(modelBuf);
    this.arch = trunk.arch;
    const d = this.dev;
    this.convPipe = d.createComputePipeline({
      layout: 'auto',
      compute: { module: d.createShaderModule({ code: CONV_WGSL }), entryPoint: 'main' },
    });
    this.linPipe = d.createComputePipeline({
      layout: 'auto',
      compute: { module: d.createShaderModule({ code: LINEAR_WGSL }), entryPoint: 'main' },
    });
    this.dummy = this.abuf(16);
    this.build(trunk, reader);
  }

  /** A storage buffer holding a weight array, uploaded once. */
  protected sbuf(arr: Float32Array<ArrayBuffer>): GPUBuffer {
    const b = this.dev.createBuffer({
      size: Math.max(arr.byteLength, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (arr.byteLength) this.dev.queue.writeBuffer(b, 0, arr);
    return b;
  }

  /** An `n`-float activation/scratch buffer (storage, copyable both ways). */
  protected abuf(n: number): GPUBuffer {
    return this.dev.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  }

  /** A `bytes`-byte buffer the host can map for readback. */
  protected stageBuf(bytes: number): GPUBuffer {
    return this.dev.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  protected convLayer(
    cv: Conv,
    xb: GPUBuffer,
    yb: GPUBuffer,
    relu: number,
    resBuf: GPUBuffer | null,
  ): void {
    const u = this.dev.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bg = this.dev.createBindGroup({
      layout: this.convPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: xb } },
        { binding: 1, resource: { buffer: this.sbuf(cv.w) } },
        { binding: 2, resource: { buffer: this.sbuf(cv.b) } },
        { binding: 3, resource: { buffer: resBuf ?? this.dummy } },
        { binding: 4, resource: { buffer: yb } },
        { binding: 5, resource: { buffer: u } },
      ],
    });
    this.layers.push({ kind: 'conv', u, bg, cv, relu, residual: !!resBuf });
  }

  protected linLayer(l: Linear, xb: GPUBuffer, yb: GPUBuffer, act: number): void {
    const u = this.dev.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bg = this.dev.createBindGroup({
      layout: this.linPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: xb } },
        { binding: 1, resource: { buffer: this.sbuf(l.w) } },
        { binding: 2, resource: { buffer: this.sbuf(l.b) } },
        { binding: 3, resource: { buffer: yb } },
        { binding: 4, resource: { buffer: u } },
      ],
    });
    this.layers.push({ kind: 'lin', u, bg, l, act });
  }

  /** Lays out stem → tower, returning the trunk-output buffer the head reads
   * and a scratch buffer (the second conv of the last block's free slot, reused
   * by flat heads). Shared by both head families. */
  protected buildTrunk(trunk: Trunk, area: number): { X: GPUBuffer; scratch: GPUBuffer } {
    const C = trunk.arch.C;
    const actX = this.abuf(MAX_BATCH * C * area);
    const actY = this.abuf(MAX_BATCH * C * area);
    const actT = this.abuf(MAX_BATCH * C * area);
    this.convLayer(trunk.stem, this.inBuf, actX, 1, null);
    let X = actX;
    let Y = actY;
    for (const [c1, c2] of trunk.tower) {
      this.convLayer(c1, X, actT, 1, null);
      this.convLayer(c2, actT, Y, 1, X);
      [X, Y] = [Y, X];
    }
    return { X, scratch: actT };
  }

  /** Reprograms the per-layer uniforms when the batch or board size changes. */
  private setShape(B: number, size: number): void {
    if (B === this.batch && size === this.curSize) return;
    this.batch = B;
    this.curSize = size;
    const area = size * size;
    for (const L of this.layers) {
      if (L.kind === 'conv') {
        this.dev.queue.writeBuffer(
          L.u,
          0,
          new Uint32Array([L.cv.ci, L.cv.co, L.cv.k, L.relu, L.residual ? 1 : 0, B, size, area]),
        );
      } else {
        this.dev.queue.writeBuffer(L.u, 0, new Uint32Array([L.l.ni, L.l.no, L.act, B]));
      }
    }
  }

  /** Runs the net on `planes` (`B` items at board `size`), serialized against
   * any in-flight call. Subclasses expose a typed wrapper. */
  protected run(planes: Float32Array<ArrayBuffer>, B: number, size: number): Promise<Eval> {
    const r = this.queue.then(() => this.runNow(planes, B, size));
    this.queue = r.catch(() => {});
    return r;
  }

  private async runNow(planes: Float32Array<ArrayBuffer>, B: number, size: number): Promise<Eval> {
    if (B < 1 || B > MAX_BATCH) throw new Error(`batch ${B} out of range 1..${MAX_BATCH}`);
    const area = size * size;
    this.setShape(B, size);
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
    return this.finish(enc, B, area);
  }

  /** Copies two GPU buffers into mapped staging buffers and reads them back as
   * `Float32Array`s — with the unmap-on-error guard so a failed map never bricks
   * the staging buffers. */
  protected async readback(
    enc: GPUCommandEncoder,
    a: { src: GPUBuffer; stage: GPUBuffer; bytes: number },
    b: { src: GPUBuffer; stage: GPUBuffer; bytes: number },
  ): Promise<[Float32Array, Float32Array]> {
    enc.copyBufferToBuffer(a.src, 0, a.stage, 0, a.bytes);
    enc.copyBufferToBuffer(b.src, 0, b.stage, 0, b.bytes);
    this.dev.queue.submit([enc.finish()]);
    try {
      await Promise.all([
        a.stage.mapAsync(GPUMapMode.READ, 0, a.bytes),
        b.stage.mapAsync(GPUMapMode.READ, 0, b.bytes),
      ]);
    } catch (e) {
      a.stage.unmap();
      b.stage.unmap();
      throw e;
    }
    try {
      return [
        new Float32Array(a.stage.getMappedRange(0, a.bytes).slice(0)),
        new Float32Array(b.stage.getMappedRange(0, b.bytes).slice(0)),
      ];
    } finally {
      a.stage.unmap();
      b.stage.unmap();
    }
  }
}

/** Pooled-head net (go, snake): the trunk and each head's 1×1 conv run on the
 * GPU; the global pool and the linear heads run on the CPU in `heads()`, which a
 * subclass supplies. Board-size-agnostic — `forward(planes, B, size)` runs at
 * any `size` whose area ≤ the export's. */
export abstract class PooledHeadNet extends AznetBase {
  protected polOut!: GPUBuffer;
  protected vOut!: GPUBuffer;
  protected v1!: Linear;
  protected v2!: Linear;
  private stagePol!: GPUBuffer;
  private stageVal!: GPUBuffer;
  /** Buffers are sized for the export's (largest) board; play area ≤ this. */
  private maxArea = 0;

  protected parse(buf: ArrayBuffer): { trunk: Trunk; reader: Reader } {
    return parseTrunk(buf);
  }

  /** Reads this head's weights from the reader (positioned past the trunk) and
   * stashes whatever the subclass's `heads()` needs. Returns the two 1×1 convs
   * (`p1`, `v1`) the GPU runs before readback. */
  protected abstract parseHead(arch: Arch, r: Reader): { p1: Conv; v1: Conv };
  /** CPU global-pool heads, mirroring nn-infer's reference forward exactly. */
  protected abstract heads(
    polBlock: Float32Array,
    vBlock: Float32Array,
    B: number,
    area: number,
  ): Eval;

  protected build(trunk: Trunk, reader: Reader): void {
    const C = trunk.arch.C;
    const area = trunk.arch.size * trunk.arch.size;
    this.maxArea = area;
    this.inBuf = this.abuf(MAX_BATCH * trunk.arch.planes * area);
    this.polOut = this.abuf(MAX_BATCH * C * area);
    this.vOut = this.abuf(MAX_BATCH * C * area);
    this.stagePol = this.stageBuf(MAX_BATCH * C * area * 4);
    this.stageVal = this.stageBuf(MAX_BATCH * C * area * 4);
    const { X } = this.buildTrunk(trunk, area);
    const { p1, v1 } = this.parseHead(trunk.arch, reader);
    reader.done();
    this.convLayer(p1, X, this.polOut, 1, null);
    this.convLayer(v1, X, this.vOut, 1, null);
  }

  protected async finish(enc: GPUCommandEncoder, B: number, area: number): Promise<Eval> {
    const blockBytes = B * this.arch.C * area * 4;
    const [polBlock, vBlock] = await this.readback(
      enc,
      { src: this.polOut, stage: this.stagePol, bytes: blockBytes },
      { src: this.vOut, stage: this.stageVal, bytes: blockBytes },
    );
    return this.heads(polBlock, vBlock, B, area);
  }

  /** planes `[B × planes·area]` → logits + values. `size` defaults to the
   * export's board (snake's fixed size); go passes the requested play size. */
  forward(planes: Float32Array<ArrayBuffer>, B: number, size = this.arch.size): Promise<Eval> {
    if (size * size > this.maxArea) throw new Error(`size ${size} exceeds export max`);
    return this.run(planes, B, size);
  }

  /** The shared global-pool value head (go and snake are identical): pool the
   * `v1` features, MLP (3C→128 relu → 1), tanh. */
  protected poolValue(vBlock: Float32Array, base: number, area: number): number {
    const vG = globalPool(vBlock, base, this.arch.C, area);
    const h = linearFwd(this.v1, vG, true);
    return Math.tanh(linearFwd(this.v2, h, false)[0]);
  }
}
