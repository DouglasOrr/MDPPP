// An imitation-learning agent for pingpong

import { NdArray } from "./ndarray"
import type { Game } from "./pingpong"
import * as T from "./tensors"

const STATE_SIZE = 4

export function getState(game: Game, player: number): number[] {
  // Reflect the ball y-position for player 1 "from their view"
  const ballSpeed = Math.sqrt(game.ball_v[0] ** 2 + game.ball_v[1] ** 2)
  return [
    game.ball[0],
    game.ball[1] * (1 - 2 * player),
    game.ball_v[0] / ballSpeed,
    (game.ball_v[1] / ballSpeed) * (1 - 2 * player),
  ]
}

// The replay buffer retains a set of states to use for training
class ReplayBuffer {
  length: number
  writeProbability: number
  state: NdArray
  position: NdArray

  constructor(capacity: number, writeProbability: number) {
    this.state = new NdArray([capacity, STATE_SIZE])
    this.position = new NdArray([capacity])
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
      const N = STATE_SIZE
      this.state.data.splice(N * idx, N, ...getState(game, player))
      this.position.data[idx] = game.paddles[player][0]
    }
  }

  sample(batchSize: number): [NdArray, NdArray] {
    const N = STATE_SIZE
    const state = new NdArray([batchSize, N])
    const position = new NdArray([batchSize, 1])
    for (let i = 0; i < batchSize; ++i) {
      const idx = Math.floor(Math.random() * this.length)
      state.data.splice(
        N * i,
        N,
        ...this.state.data.slice(N * idx, N * (idx + 1)),
      )
      position.data[i] = this.position.data[idx]
    }
    return [state, position]
  }
}

export const S = {
  // Buffer
  bufferCapacity: 1000,
  writeProbability: 0.1,
  bufferMinForTraining: 100,
  // Model
  nBuckets: 40,
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
    this.embed = this.addParameter([STATE_SIZE, S.nBuckets, S.hiddenSize], 1.0)
    this.W0 = this.addParameter(
      [STATE_SIZE * S.hiddenSize, S.hiddenSize],
      S.hiddenSize ** -0.5,
    )
    this.W1 = this.addParameter([S.hiddenSize, S.nBuckets], 0)
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
    hidden = T.view(hidden, [batchSize, STATE_SIZE * S.hiddenSize])
    hidden = T.dot(hidden, this.W0)
    hidden = T.relu(hidden)
    return T.dot(hidden, this.W1)
  }

  train(): void {
    if (S.bufferMinForTraining <= this.buffer.length) {
      this.step(() => {
        let [state, targets] = this.buffer.sample(S.batchSize)
        targets = this.bucketise(targets)
        const logits = this.logits(state)
        const losses = T.softmaxCrossEntropy(logits, targets)
        losses.grad.fill_(1)
        this.trainLoss.push(losses.data.mean().data[0])
        this.trainAccuracy.push(T.accuracy(logits.data, targets).mean().data[0])
      })
    }
  }

  act(game: Game, player: number, debug: boolean): number {
    const state = new NdArray([1, STATE_SIZE], getState(game, player))
    const logits = this.logits(state)
    const targetBucket = T.idxMax(logits.data.data, 0, S.nBuckets)
    const currentBucket = this.bucketise(
      new NdArray([], [game.paddles[player][0]]),
    ).data[0]
    if (debug) {
      console.log("agent.buffer.length", this.buffer.length)
      console.log("agent.logits", logits.data.data)
      console.log("agent.currentBucket", currentBucket)
      console.log("agent.targetBucket", targetBucket)
    }
    return Math.sign(targetBucket - currentBucket)
  }
}
