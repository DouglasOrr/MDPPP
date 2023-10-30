export const S = {
  dt: 0.01,
  bounds: { left: -0.5, right: 0.5, top: -1, bottom: 1 },
  paddleSpeed: 1.75,
  paddleWidth: 0.2,
  bounceAcceleration: 0.1,
}

function clip(x: number, min: number, max: number): number {
  return Math.min(Math.max(x, min), max)
}

export function simpleAgent(game: Game, player: integer): integer {
  const dx = game.ball[0] - game.paddles[player][0]
  return Math.abs(dx) < S.paddleWidth / 4 ? 0 : Math.sign(dx)
}

export class Game {
  ball: [number, number]
  ball_v: [number, number]
  paddles: Array<[number, number]>

  constructor() {
    this.ball = [0, 0]
    this.ball_v = [2 * (Math.random() - 0.5), Math.sign(Math.random() - 0.5)]
    this.paddles = [
      [0, S.bounds.bottom],
      [0, S.bounds.top],
    ]
  }

  update(control: integer[]): null | 0 | 1 {
    // Paddle
    this.paddles.forEach((paddle, i) => {
      paddle[0] = clip(
        paddle[0] + S.dt * S.paddleSpeed * control[i],
        S.bounds.left + S.paddleWidth / 2,
        S.bounds.right - S.paddleWidth / 2,
      )
    })
    // Ball
    this.ball[0] += S.dt * this.ball_v[0]
    this.ball[1] += S.dt * this.ball_v[1]
    const bounceMultiplier = -1 - S.bounceAcceleration
    if (this.ball[0] < S.bounds.left || S.bounds.right < this.ball[0]) {
      this.ball_v[0] *= bounceMultiplier
    }
    if (this.ball[1] < S.bounds.top) {
      if (Math.abs(this.paddles[1][0] - this.ball[0]) > S.paddleWidth / 2) {
        return 0
      }
      this.ball_v[1] *= bounceMultiplier
    }
    if (this.ball[1] > S.bounds.bottom) {
      if (Math.abs(this.paddles[0][0] - this.ball[0]) > S.paddleWidth / 2) {
        return 1
      }
      this.ball_v[1] *= bounceMultiplier
    }
    this.ball = [
      clip(this.ball[0], S.bounds.left, S.bounds.right),
      clip(this.ball[1], S.bounds.top, S.bounds.bottom),
    ]
    return null
  }
}
