// tensors is a basic PyTorch-like tensor/deep-learning library
//
//   Tensor -- (value, grad)
//   withGrad() -- scope to enable autograd
//   {view, dot, ...} -- differentiable ops

import { NdArray, assertArrayEquals } from "./ndarray"

export class Tensor {
  data: NdArray
  grad: NdArray
  shape: number[]

  constructor(data: NdArray) {
    this.data = data
    this.grad = new NdArray(data.shape).fill_(0)
    this.shape = data.shape
  }

  ndim(): number {
    return this.data.ndim()
  }
}

// Tape

let _tape: Array<() => void> = []

export function withGrad<T>(fn: () => T): T {
  _tape = []
  const result = fn()
  _tape
    .slice()
    .reverse()
    .forEach((f) => {
      f()
    })
  return result
}

// Ops

export function view(t: Tensor, shape: number[]): Tensor {
  const result = new Tensor(t.data.view_(shape))
  _tape.push(() => {
    t.grad.add_(result.grad.view_(t.shape))
  })
  return result
}

export function gather(t: Tensor, i: Tensor): Tensor {
  const result = new Tensor(t.data.gather(i.data))
  _tape.push(() => {
    t.grad.add_(result.grad.scatter(i.data, t.shape[i.ndim() - 1]))
  })
  return result
}

export function transpose(t: Tensor, dims: number[]): Tensor {
  const result = new Tensor(t.data.transpose(dims))
  _tape.push(() => {
    t.grad.add_(result.grad.untranspose(dims))
  })
  return result
}

export function dot(a: Tensor, b: Tensor): Tensor {
  const result = new Tensor(a.data.dot(b.data))
  _tape.push(() => {
    a.grad.add_(result.grad.dot(b.data.t()))
    b.grad.add_(a.data.t().dot(result.grad))
  })
  return result
}

export function relu(t: Tensor): Tensor {
  const result = new Tensor(t.data.clone().map_((x) => Math.max(x, 0)))
  _tape.push(() => {
    t.grad.add_(result.grad.clone().map_((g, i) => g * +(t.data.data[i] >= 0)))
  })
  return result
}

export function softmaxCrossEntropy(logits: Tensor, target: NdArray): Tensor {
  assertArrayEquals(target.shape, [logits.shape[0], 1])
  const logSoftmax = logits.data.clone().logSoftmax_()
  const losses = new Tensor(logSoftmax.gather(target).map_((x) => -x))
  _tape.push(() => {
    const nClasses = logits.shape[logits.ndim() - 1]
    logits.grad.map_((v, i) => {
      const targetI = Math.floor(i / nClasses)
      const lossGrad = losses.grad.data[targetI]
      const grad =
        lossGrad *
        (Math.exp(logSoftmax.data[i]) -
          +(i % nClasses === target.data[targetI]))
      return v + grad
    })
  })
  return losses
}

export function idxMax(values: number[], start: number, end: number): number {
  let [idx, max] = [start, values[start]]
  for (let i = start + 1; i < end; ++i) {
    if (values[i] > max) {
      idx = i
      max = values[i]
    }
  }
  return idx
}

// Elementwise accuracy
// logits -- (B, N)
// targets -- (B, 1)
// returns -- (B, 1)
export function accuracy(logits: NdArray, targets: NdArray): NdArray {
  const N = logits.shape[1]
  return targets
    .clone()
    .map_((t, i) => +(i * N + t === idxMax(logits.data, i * N, (i + 1) * N)))
}

// Models

export abstract class Parameter extends Tensor {
  abstract update(): void
}

export class SgdParameter extends Parameter {
  lr: number

  constructor(data: NdArray, lr: number) {
    super(data)
    this.lr = lr
  }

  update(): void {
    this.data.map2_(this.grad, (x, grad) => x - this.lr * grad)
  }
}

// Adam (without bias correction)
export class AdamParameter extends Parameter {
  lr: number
  beta: [number, number]
  epsilon: number
  momentum: NdArray
  variance: NdArray

  constructor(
    data: NdArray,
    lr: number,
    beta: [number, number],
    epsilon: number,
  ) {
    super(data)
    this.lr = lr
    this.beta = beta
    this.epsilon = epsilon
    this.momentum = new NdArray(data.shape).fill_(0)
    this.variance = new NdArray(data.shape).fill_(0)
  }

  update(): void {
    const [beta1, beta2] = this.beta
    const data = this.data.data
    const momentum = this.momentum.data
    const variance = this.variance.data
    const grad = this.grad.data
    for (let i = 0; i < data.length; ++i) {
      const g = grad[i]
      momentum[i] = beta1 * momentum[i] + (1 - beta1) * g
      variance[i] = beta2 * variance[i] + (1 - beta2) * g * g
    }
    for (let i = 0; i < data.length; ++i) {
      data[i] -= (this.lr * momentum[i]) / (variance[i] + this.epsilon) ** 0.5
    }
  }
}

export type ParameterFactory = (shape: number[], scale: number) => Parameter

export class Model {
  parameters: Parameter[] = []
  parameterFactory: ParameterFactory

  constructor(parameterFactory: ParameterFactory) {
    this.parameterFactory = parameterFactory
  }

  addParameter(shape: number[], scale: number): Parameter {
    const parameter = this.parameterFactory(shape, scale)
    this.parameters.push(parameter)
    return parameter
  }

  step<T>(fn: () => T): T {
    this.parameters.forEach((p) => p.grad.fill_(0))
    const result = withGrad(fn)
    this.parameters.forEach((p) => {
      p.update()
    })
    return result
  }
}
