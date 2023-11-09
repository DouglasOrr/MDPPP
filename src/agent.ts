// An imitation-learning agent for pingpong

import { NdArray } from "./ndarray"
import type { Game } from "./pingpong"
import * as T from "./tensors"

function getState(game: Game, player: number): number[] {
  // Reflect the ball y-position for player 1 "from their view"
  return [
    game.ball[0],
    game.ball[1] * (1 - 2 * player),
    game.paddles[player][0],
  ]
}

// The replay buffer retains a set of states to use for training
class ReplayBuffer {
  length: number
  writeProbability: number
  state: NdArray
  control: NdArray
  static readonly STATE_SIZE = 3

  constructor(capacity: number, writeProbability: number) {
    this.state = new NdArray([capacity, ReplayBuffer.STATE_SIZE])
    this.control = new NdArray([capacity])
    this.length = 0
    this.writeProbability = writeProbability
  }

  update(game: Game, player: number, control: number): void {
    if (Math.random() < this.writeProbability) {
      let idx: number
      if (this.length < this.state.shape[0]) {
        idx = this.length++
      } else {
        idx = Math.floor(Math.random() * this.length)
      }
      const N = ReplayBuffer.STATE_SIZE
      this.state.data.splice(N * idx, N, ...getState(game, player))
      this.control.data[idx] = control
    }
  }

  sample(batchSize: number): [NdArray, NdArray] {
    const N = ReplayBuffer.STATE_SIZE
    const state = new NdArray([batchSize, N])
    const control = new NdArray([batchSize, 1])
    for (let i = 0; i < batchSize; ++i) {
      const idx = Math.floor(Math.random() * this.length)
      state.data.splice(
        N * i,
        N,
        ...this.state.data.slice(N * idx, N * (idx + 1)),
      )
      control.data[i] = this.control.data[idx]
    }
    return [state, control]
  }
}

export const S = {
  // Buffer
  bufferCapacity: 1000,
  writeProbability: 0.1,
  bufferMinForTraining: 100,
  // Model
  nFeatures: 3,
  nActions: 3,
  nBuckets: 10,
  hiddenSize: 128,
  // Training
  batchSize: 20,
  lr: 0.001,
}

export class Model extends T.Model {
  buffer: ReplayBuffer
  embed: T.Parameter
  W0: T.Parameter
  W1: T.Parameter
  trainLoss: number[]
  trainAccuracy: number[]

  constructor() {
    super(
      (shape, scale) =>
        new T.AdamParameter(
          new NdArray(shape).rand_(-scale, scale),
          S.lr,
          [0.9, 0.999],
          1e-8,
        ),
    )
    this.buffer = new ReplayBuffer(S.bufferCapacity, S.writeProbability)
    this.embed = this.addParameter([S.nFeatures, S.nBuckets, S.hiddenSize], 1.0)
    this.W0 = this.addParameter(
      [S.nFeatures * S.hiddenSize, S.hiddenSize],
      S.hiddenSize ** -0.5,
    )
    this.W1 = this.addParameter([S.hiddenSize, S.nActions], 0)
    this.trainLoss = []
    this.trainAccuracy = []
  }

  observe(game: Game, player: number, control: number): void {
    this.buffer.update(game, player, control)
  }

  bucketise(state: NdArray): NdArray {
    return state
      .clone()
      .map_((v) =>
        Math.max(
          0,
          Math.min(S.nBuckets - 1, Math.round(((v + 1) * S.nBuckets) / 2)),
        ),
      )
  }

  logits(state: NdArray): T.Tensor {
    const batchSize = state.shape[0]
    const buckets = new T.Tensor(this.bucketise(state))
    let hidden = T.transpose(
      T.gather(this.embed, T.transpose(buckets, [1, 0])),
      [1, 0, 2],
    )
    hidden = T.view(hidden, [batchSize, S.nFeatures * S.hiddenSize])
    hidden = T.dot(hidden, this.W0)
    hidden = T.relu(hidden)
    return T.dot(hidden, this.W1)
  }

  train(): void {
    if (S.bufferMinForTraining <= this.buffer.length) {
      this.step(() => {
        const [state, targets] = this.buffer.sample(S.batchSize)
        targets.map_((x) => x + 1)
        const logits = this.logits(state)
        const losses = T.softmaxCrossEntropy(logits, targets)
        losses.grad.fill_(1)
        this.trainLoss.push(losses.data.mean().data[0])
        this.trainAccuracy.push(T.accuracy(logits.data, targets).mean().data[0])
      })
    }
  }

  act(game: Game, player: number, debug: boolean): number {
    const state = new NdArray([1, S.nFeatures], getState(game, player))
    const logits = this.logits(state)
    if (debug) {
      console.log("agent.buffer.length", this.buffer.length)
      console.log("agent.logits", logits.data.data)
    }
    return T.idxMax(logits.data.data, 0, S.nActions) - 1
  }
}
