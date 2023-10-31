import * as PingPong from "../src/pingpong"

test("game basics", () => {
  const game = new PingPong.Game()
  expect(game.ball).toStrictEqual([0, 0])
  game.update([0, 0])
  expect(game.ball).not.toStrictEqual([0, 0])
})
