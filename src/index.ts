import * as Phaser from "phaser"
import * as PingPong from "./pingpong"
import * as Agent from "./agent"

const S = PingPong.S
const WHITE = 0xffffffff

interface Keys {
  left: Phaser.Input.Keyboard.Key
  right: Phaser.Input.Keyboard.Key
}

export default class Main extends Phaser.Scene {
  state: PingPong.Game | null = new PingPong.Game()
  ball: Phaser.GameObjects.Arc | null = null
  paddles: Phaser.GameObjects.Rectangle[] = []
  keys: Keys | null = null
  gameState: "playing" | "paused" | "step" = "playing"
  agent: Agent.Model | null = null

  constructor() {
    super("main")
  }

  preload(): void {}

  create(): void {
    this.state = new PingPong.Game()
    this.agent = new Agent.Model()
    this.gameState = "playing"
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
        this.gameState = "playing"
      }
      if (e.key === "p") {
        this.gameState = "step"
      }
      if (e.key === "o") {
        this.gameState = "playing"
      }
    })
  }

  update(): void {
    if (
      this.state !== null &&
      this.agent !== null &&
      this.gameState !== "paused"
    ) {
      // const playerControl = +this.keys!.right.isDown - +this.keys!.left.isDown
      // const control = [playerControl, PingPong.simpleAgent(this.state, 1)]
      const playerControl = PingPong.simpleAgent(this.state, 0)
      const control = [
        playerControl,
        this.agent.act(this.state, 1, this.gameState === "step"),
      ]
      this.agent.observe(this.state, 0, control[0])
      this.agent.train()
      const outcome = this.state.update(control)

      // Update view state
      this.ball?.setPosition(...this.state.ball)
      this.state.paddles.forEach(([x, y], i) =>
        this.paddles[i].setPosition(x, y),
      )
      if (this.gameState === "step") {
        this.gameState = "paused"
      }
      // End of game
      if (outcome !== null) {
        console.log(`Winner ${outcome}`)
        // this.state = null
        this.state = new PingPong.Game()
      }
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
