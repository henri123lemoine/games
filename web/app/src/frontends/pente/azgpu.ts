// Pente's WebGPU evaluator: the global-pool-spatial head over the shared resnet
// trunk — byte-identical in topology to go's policy head (a KataGo-style policy:
// a 1×1 conv biased by the global pool, a bias-less placement conv, and a pooled
// trailing logit) plus the shared global-pool value head. Pente's net carries no
// ownership head (the flag is unset), so the conditional skip never fires; it is
// kept only so a future ownership-carrying export would still parse. Everything
// generic lives in engine/aznet.ts. Validated against nn-infer's reference
// forward (and the wasm CPU fallback) via the pente-validate harness.

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
/** Input planes for the pente encoder (mirrors pente::encode::PLANES). */
export const PLANES = 8;
/** Policy width for a board: one logit per point plus the (unused) pass slot. */
export function policyLen(size: number): number {
  return size * size + 1;
}

export class PenteGpu extends PooledHeadNet {
  private pgb!: Linear;
  private pfc!: Conv;
  private ppass!: Linear;

  static async init(modelBuf: ArrayBuffer): Promise<PenteGpu> {
    const g = new PenteGpu();
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
    // Pente exports carry no ownership head, but mirror go's parse so an
    // ownership-carrying export would still consume cleanly.
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
