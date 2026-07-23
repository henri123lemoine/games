// Chess's WebGPU evaluator: the flat-conv head over the shared resnet trunk.
// The generic trunk (conv kernel, GPU stem + tower, buffer/dispatch driver,
// parser) lives in engine/aznet.ts; chess is the outlier head — unlike go/snake
// it runs the whole head on the GPU: a 1×1 policy conv stack (p1 → p2) and a
// value head (a small conv reduced by two dense layers via the shared linear
// kernel), with only the channel-major→square-major policy reshuffle on the
// CPU. Validated against nn-infer's reference fp32 forward (and the wasm CPU
// fallback) to ~1e-3 by /azero-test.html.

import {
  AznetBase,
  CHESS_VALUE_CHANNELS,
  CHESS_VALUE_HIDDEN,
  type Conv,
  type Eval,
  type Linear,
  MAX_BATCH,
  parseTrunk,
  type Reader,
  softmaxOver,
  type Trunk,
} from '../../engine/aznet';

export { softmaxOver, MAX_BATCH };
/** Input planes for the chess encoder (mirrors chess::encode::PLANES). */
export const PLANES = 18;
/** Square-major policy width of the shipped chess net: 64 squares × 73 move
 * planes. The driver derives the actual width from the export's header, but the
 * bot and test harness import this as the fixed stride of the deployed net. */
export const POLICY_LEN = 4672;

const AREA = 64;

export class AzGpu extends AznetBase {
  private polBuf!: GPUBuffer;
  private v1out!: GPUBuffer;
  private stagePol!: GPUBuffer;
  private stageVal!: GPUBuffer;
  /** Move planes (`policy_len / area`) and the flat policy width, from the
   * header — channel-major `[movePlanes, area]` transposed to square-major. */
  private movePlanes = 0;
  private policyLen = 0;

  static async init(modelBuf: ArrayBuffer): Promise<AzGpu> {
    const g = new AzGpu();
    await g.boot(modelBuf);
    return g;
  }

  protected parse(buf: ArrayBuffer): { trunk: Trunk; reader: Reader } {
    return parseTrunk(buf);
  }

  protected build(trunk: Trunk, r: Reader): void {
    const C = trunk.arch.C;
    this.policyLen = trunk.arch.policyLen;
    this.movePlanes = this.policyLen / AREA;
    this.inBuf = this.abuf(MAX_BATCH * trunk.arch.planes * AREA);
    const { X, scratch } = this.buildTrunk(trunk, AREA);

    // Policy: p1 (C→C, relu) → p2 (C→movePlanes, no relu), channel-major.
    const p1: Conv = r.conv(C, C, 1);
    const p2: Conv = r.conv(C, this.movePlanes, 1);
    // Value: v1 (C→8, relu) → vf1 (8·area→256, relu) → vf2 (256→1, tanh).
    const v1: Conv = r.conv(C, CHESS_VALUE_CHANNELS, 1);
    const vf1: Linear = r.linear(CHESS_VALUE_CHANNELS * AREA, CHESS_VALUE_HIDDEN);
    const vf2: Linear = r.linear(CHESS_VALUE_HIDDEN, trunk.arch.valueSeats);
    r.done();

    this.polBuf = this.abuf(MAX_BATCH * this.movePlanes * AREA);
    const v64 = this.abuf(MAX_BATCH * CHESS_VALUE_CHANNELS * AREA);
    const vHidden = this.abuf(MAX_BATCH * CHESS_VALUE_HIDDEN);
    this.v1out = this.abuf(MAX_BATCH * trunk.arch.valueSeats);
    this.stagePol = this.stageBuf(MAX_BATCH * this.movePlanes * AREA * 4);
    this.stageVal = this.stageBuf(MAX_BATCH * trunk.arch.valueSeats * 4);

    this.convLayer(p1, X, scratch, 1, null);
    this.convLayer(p2, scratch, this.polBuf, 0, null);
    this.convLayer(v1, X, v64, 1, null);
    this.linLayer(vf1, v64, vHidden, 1);
    this.linLayer(vf2, vHidden, this.v1out, trunk.arch.valueSeats === 1 ? 2 : 0);
  }

  protected async finish(enc: GPUCommandEncoder, B: number): Promise<Eval> {
    const mp = this.movePlanes;
    const [polCM, values] = await this.readback(
      enc,
      { src: this.polBuf, stage: this.stagePol, bytes: B * mp * AREA * 4 },
      { src: this.v1out, stage: this.stageVal, bytes: B * this.arch.valueSeats * 4 },
    );
    // Channel-major [movePlanes, area] → square-major policy logits.
    const logits = new Float32Array(B * this.policyLen);
    for (let b = 0; b < B; b++)
      for (let p = 0; p < mp; p++)
        for (let sq = 0; sq < AREA; sq++)
          logits[b * this.policyLen + sq * mp + p] = polCM[(b * mp + p) * AREA + sq];
    return { logits, values };
  }

  /** planes `[B × 18·64]` → square-major logits `[B × policyLen]` and values `[B]`. */
  forward(planes: Float32Array<ArrayBuffer>, B: number): Promise<Eval> {
    return this.run(planes, B, 8);
  }
}
