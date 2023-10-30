import * as Phaser from "phaser"

const S = {
  paddlew: 0.2,
}

const WHITE = 0xffffffff

export default class Game extends Phaser.Scene {
  ball: Phaser.GameObjects.Arc | null
  paddles: Phaser.GameObjects.Rectangle[]

  constructor() {
    super("game")
    this.ball = null
    this.paddles = []
  }

  preload(): void {}

  create(): void {
    this.ball = this.add.circle(0, 0, 0.02, WHITE)
    this.paddles = [
      this.add.rectangle(0, -1, S.paddlew, 0.02, WHITE),
      this.add.rectangle(0, 1, S.paddlew, 0.02, WHITE),
    ]
    const camera = this.cameras.main
    camera.setZoom(camera.width, camera.height / 2)
    camera.setScroll(-camera.width / 2, -camera.height / 2)
  }
}

export const game = new Phaser.Game({
  type: Phaser.AUTO,
  backgroundColor: "#000000",
  width: 400,
  height: 800,
  scene: Game,
})
