// Four-player chess's fixed 14x14 flat-conv policy and four-seat value head.
// The trunk/kernels/parser are shared with every AZNET1 browser net.

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

export { softmaxOver };
export const PLANES = 71;
export const AREA = 14 * 14;
export const POLICY_LEN = AREA * 112;
export const VALUE_SEATS = 4;

export class FourPlayerAzGpu extends AznetBase {
  private policy!: GPUBuffer;
  private values!: GPUBuffer;
  private stagePolicy!: GPUBuffer;
  private stageValues!: GPUBuffer;
  private movePlanes = 0;
  private policyLen = 0;

  static async init(model: ArrayBuffer): Promise<FourPlayerAzGpu> {
    const gpu = new FourPlayerAzGpu();
    await gpu.boot(model);
    if (
      gpu.arch.size !== 14 ||
      gpu.arch.planes !== PLANES ||
      gpu.arch.policyLen !== POLICY_LEN ||
      gpu.arch.valueSeats !== VALUE_SEATS
    )
      throw new Error('incompatible four-player chess net');
    return gpu;
  }

  protected parse(buf: ArrayBuffer): { trunk: Trunk; reader: Reader } {
    return parseTrunk(buf);
  }

  protected build(trunk: Trunk, reader: Reader): void {
    const C = trunk.arch.C;
    this.policyLen = trunk.arch.policyLen;
    this.movePlanes = this.policyLen / AREA;
    this.inBuf = this.abuf(MAX_BATCH * trunk.arch.planes * AREA);
    const { X, scratch } = this.buildTrunk(trunk, AREA);
    const p1: Conv = reader.conv(C, C, 1);
    const p2: Conv = reader.conv(C, this.movePlanes, 1);
    const v1: Conv = reader.conv(C, CHESS_VALUE_CHANNELS, 1);
    const vf1: Linear = reader.linear(CHESS_VALUE_CHANNELS * AREA, CHESS_VALUE_HIDDEN);
    const vf2: Linear = reader.linear(CHESS_VALUE_HIDDEN, VALUE_SEATS);
    reader.done();

    this.policy = this.abuf(MAX_BATCH * this.movePlanes * AREA);
    const valuePlanes = this.abuf(MAX_BATCH * CHESS_VALUE_CHANNELS * AREA);
    const hidden = this.abuf(MAX_BATCH * CHESS_VALUE_HIDDEN);
    this.values = this.abuf(MAX_BATCH * VALUE_SEATS);
    this.stagePolicy = this.stageBuf(MAX_BATCH * this.movePlanes * AREA * 4);
    this.stageValues = this.stageBuf(MAX_BATCH * VALUE_SEATS * 4);
    this.convLayer(p1, X, scratch, 1, null);
    this.convLayer(p2, scratch, this.policy, 0, null);
    this.convLayer(v1, X, valuePlanes, 1, null);
    this.linLayer(vf1, valuePlanes, hidden, 1);
    this.linLayer(vf2, hidden, this.values, 0);
  }

  protected async finish(encoder: GPUCommandEncoder, batch: number): Promise<Eval> {
    const [channelMajor, values] = await this.readback(
      encoder,
      {
        src: this.policy,
        stage: this.stagePolicy,
        bytes: batch * this.movePlanes * AREA * 4,
      },
      { src: this.values, stage: this.stageValues, bytes: batch * VALUE_SEATS * 4 },
    );
    const logits = new Float32Array(batch * this.policyLen);
    for (let b = 0; b < batch; b++)
      for (let plane = 0; plane < this.movePlanes; plane++)
        for (let square = 0; square < AREA; square++)
          logits[b * this.policyLen + square * this.movePlanes + plane] =
            channelMajor[(b * this.movePlanes + plane) * AREA + square];
    return { logits, values };
  }

  forward(planes: Float32Array<ArrayBuffer>, batch: number): Promise<Eval> {
    return this.run(planes, batch, 14);
  }
}
