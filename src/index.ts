import * as Phaser from "phaser"
import * as PingPong from "./pingpong"

const S = PingPong.S
const WHITE = 0xffffffff

export default class Main extends Phaser.Scene {
  state: PingPong.Game | null
  ball: Phaser.GameObjects.Arc | null
  paddles: Phaser.GameObjects.Rectangle[]
  keys: {
    left: Phaser.Input.Keyboard.Key
    right: Phaser.Input.Keyboard.Key
  } | null

  constructor() {
    super("main")
    this.state = new PingPong.Game()
    this.ball = null
    this.paddles = []
    this.keys = null
  }

  preload(): void {}

  create(): void {
    this.state = new PingPong.Game()
    // Objects
    this.ball = this.add.circle(...this.state.ball, 0.02, WHITE)
    this.paddles = this.state.paddles.map(([x, y]) =>
      this.add.rectangle(x, y, S.paddleWidth, 0.02, WHITE),
    )
    // Camera
    const camera = this.cameras.main
    camera.setZoom(
      camera.width / (S.bounds.right - S.bounds.left),
      camera.height / (S.bounds.bottom - S.bounds.top),
    )
    camera.setScroll(-camera.width / 2, -camera.height / 2)
    // Input
    this.keys = {
      left: this.input.keyboard!.addKey("LEFT"),
      right: this.input.keyboard!.addKey("RIGHT"),
    }
    this.input.keyboard!.on("keydown", (e: KeyboardEvent) => {
      if (e.key === " ") {
        this.state = new PingPong.Game()
      }
    })
  }

  update(): void {
    if (this.state != null) {
      const control = +this.keys!.right.isDown - +this.keys!.left.isDown
      const outcome = this.state.update([
        control,
        PingPong.simpleAgent(this.state, 1),
      ])
      if (outcome != null) {
        console.log(`Winner ${outcome}`)
        this.state = null
        return
      }
      this.ball?.setPosition(...this.state.ball)
      this.state.paddles.forEach(([x, y], i) =>
        this.paddles[i].setPosition(x, y),
      )
    }
  }
}

export const game = new Phaser.Game({
  type: Phaser.AUTO,
  backgroundColor: "#000000",
  width: 400,
  height: 800,
  scene: Main,
})
