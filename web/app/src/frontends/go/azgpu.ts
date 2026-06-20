// Go's WebGPU evaluator: the global-pool-spatial head over the shared resnet
// trunk. Everything generic (the conv kernel, the GPU trunk + head 1×1 convs,
// the buffer/dispatch driver, the parser, and globalPool/linearFwd/softmaxOver)
// lives in engine/aznet.ts; this file is only go's head — a KataGo-style policy
// (1×1 conv biased by the global pool, a bias-less placement conv, and a pooled
// pass logit: one logit per board point plus pass) and the shared global-pool
// value head. The ownership head (GO3) is parsed-and-skipped: leaf scoring uses
// policy+value only. Validated against nn-infer's reference forward (and the
// wasm CPU fallback) to ~1e-3 via /go-azero-test.html.

import {
  type Arch,
  type Conv,
  type Eval,
  globalPool,
  type Linear,
  linearFwd,
  POOL_VALUE_HIDDEN,
  PooledHeadNet,
  type Reader,
  softmaxOver,
} from '../../engine/aznet';

export { softmaxOver };
export { MAX_BATCH } from '../../engine/aznet';
/** Input planes for the go encoder (mirrors go::encode::PLANES). */
export const PLANES = 16;
/** Policy width for a board: one logit per point plus the pass. */
export function policyLen(size: number): number {
  return size * size + 1;
}

export class GoGpu extends PooledHeadNet {
  private pgb!: Linear;
  private pfc!: Conv;
  private ppass!: Linear;

  static async init(modelBuf: ArrayBuffer): Promise<GoGpu> {
    const g = new GoGpu();
    await g.boot(modelBuf);
    return g;
  }

  protected parseHead(arch: Arch, r: Reader): { p1: Conv; v1: Conv } {
    const C = arch.C;
    const p1 = r.conv(C, C, 1);
    this.pgb = r.linear(3 * C, C);
    this.pfc = r.convNoBias(C, 1, 1);
    this.ppass = r.linear(3 * C, 1);
    const v1 = r.conv(C, C, 1);
    this.v1 = r.linear(3 * C, POOL_VALUE_HIDDEN);
    this.v2 = r.linear(POOL_VALUE_HIDDEN, 1);
    // GO3 carries an ownership head (o1: bias-less C→1 conv) after the value
    // head; consume it so the no-trailing-bytes check passes. The GPU path
    // scores leaves with policy+value only and never runs it.
    if (arch.ownership) r.convNoBias(C, 1, 1);
    return { p1, v1 };
  }

  protected heads(polBlock: Float32Array, vBlock: Float32Array, B: number, area: number): Eval {
    const C = this.arch.C;
    const policy = area + 1;
    const logits = new Float32Array(B * policy);
    const values = new Float32Array(B);
    for (let b = 0; b < B; b++) {
      const base = b * C * area;
      // Policy: pool the 1×1-conv features, bias them, bias-less placement conv,
      // pass logit from the pool.
      const polG = globalPool(polBlock, base, C, area);
      const bias = linearFwd(this.pgb, polG, false);
      const lb = b * policy;
      for (let pos = 0; pos < area; pos++) {
        let acc = 0;
        for (let ch = 0; ch < C; ch++) {
          let v = polBlock[base + ch * area + pos] + bias[ch];
          if (v < 0) v = 0;
          acc += this.pfc.w[ch] * v;
        }
        logits[lb + pos] = acc;
      }
      logits[lb + area] = linearFwd(this.ppass, polG, false)[0];
      values[b] = this.poolValue(vBlock, base, area);
    }
    return { logits, values };
  }
}
