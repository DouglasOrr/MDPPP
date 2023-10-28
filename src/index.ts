import * as Phaser from "phaser"

export default class Demo extends Phaser.Scene {
  constructor() {
    super("demo")
  }

  preload(): void {}

  create(): void {
    this.add.rectangle(100, 100, 50, 100, 0xff00ffff)
  }
}

export const game = new Phaser.Game({
  type: Phaser.AUTO,
  backgroundColor: "#000000",
  width: 800,
  height: 600,
  scene: Demo,
})
