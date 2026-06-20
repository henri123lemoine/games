// Snake's WebGPU evaluator: the global-pool-dense head over the shared resnet
// trunk. Everything generic (the conv kernel, the GPU trunk + head 1×1 convs,
// the buffer/dispatch driver, the parser, and globalPool/linearFwd/softmaxOver)
// lives in engine/aznet.ts; this file is only snake's head — a 3C→C→4 policy
// MLP over the global pool (the four absolute headings) and the shared
// global-pool value head. Validated against nn-infer's reference fp32 forward
// (and the wasm CPU fallback) to ~1e-3 via the browser-validate harness.

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
/** Input planes for the snake encoder (mirrors snake::encode::PLANES). */
export const PLANES = 18;
/** The four absolute headings the policy head scores (Up/Right/Down/Left). */
export const ACTIONS = 4;

export class SnakeGpu extends PooledHeadNet {
  private pf1!: Linear;
  private pf2!: Linear;

  static async init(modelBuf: ArrayBuffer): Promise<SnakeGpu> {
    const g = new SnakeGpu();
    await g.boot(modelBuf);
    return g;
  }

  protected parseHead(arch: Arch, r: Reader): { p1: Conv; v1: Conv } {
    const C = arch.C;
    const p1 = r.conv(C, C, 1);
    this.pf1 = r.linear(3 * C, C);
    this.pf2 = r.linear(C, ACTIONS);
    const v1 = r.conv(C, C, 1);
    this.v1 = r.linear(3 * C, POOL_VALUE_HIDDEN);
    this.v2 = r.linear(POOL_VALUE_HIDDEN, 1);
    return { p1, v1 };
  }

  protected heads(polBlock: Float32Array, vBlock: Float32Array, B: number, area: number): Eval {
    const C = this.arch.C;
    const logits = new Float32Array(B * ACTIONS);
    const values = new Float32Array(B);
    for (let b = 0; b < B; b++) {
      const base = b * C * area;
      // Policy: pool the 1×1-conv features → MLP (3C→C relu → 4 headings).
      const polG = globalPool(polBlock, base, C, area);
      const h = linearFwd(this.pf1, polG, true);
      logits.set(linearFwd(this.pf2, h, false), b * ACTIONS);
      values[b] = this.poolValue(vBlock, base, area);
    }
    return { logits, values };
  }
}
