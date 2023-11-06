import { NdArray } from "../src/ndarray"

test("new NdArray", () => {
  const a = new NdArray([1, 2, 3]).fill_(100)
  expect(a.shape).toStrictEqual([1, 2, 3])
  expect(a.data).toStrictEqual([100, 100, 100, 100, 100, 100])
  expect(a.ndim()).toBe(3)
  expect(a.strides()).toStrictEqual([6, 3, 1])

  const b = new NdArray([2, 1], [1000, 2000])
  expect(b.shape).toStrictEqual([2, 1])
  expect(b.data).toStrictEqual([1000, 2000])

  expect(() => new NdArray([2, 1], [1000, 2000, 3000])).toThrow("3 === 2")
})

test("clone, map_, map2_", () => {
  const a = new NdArray([1, 3]).fill_(100)
  const b = a.clone().map_((x, i) => 2 * x + i)
  expect(a.data).toStrictEqual([100, 100, 100])
  expect(b.data).toStrictEqual([200, 201, 202])

  const c = a.clone().map2_(new NdArray([1, 3], [0, 10, 20]), (x, y) => x - y)
  expect(c.data).toStrictEqual([100, 90, 80])
})

test("rand_", () => {
  const a = new NdArray([1, 20]).rand_(100, 200)
  expect(a.data.map((x) => 100 <= x && x <= 200)).toStrictEqual(
    Array(20).fill(true),
  )
})

test("view, t, transpose, untranspose", () => {
  const flatA = new NdArray([12]).map_((_, i) => i)
  const a = flatA.clone().view_([2, 1, 3, 2])
  expect(a.shape).toStrictEqual([2, 1, 3, 2])
  expect(a.data).toStrictEqual(flatA.data)

  let aT = a.t()
  expect(aT.shape).toStrictEqual([2, 1, 2, 3])
  const transposedDataA = [0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11]
  expect(aT.data).toStrictEqual(transposedDataA)

  aT = a.transpose([0, 1, 3, 2])
  expect(aT.shape).toStrictEqual([2, 1, 2, 3])
  expect(aT.data).toStrictEqual(transposedDataA)
  expect(aT.untranspose([0, 1, 3, 2])).toMatchObject(a)

  const aT2 = a.transpose([3, 2, 0, 1])
  expect(aT2.shape).toStrictEqual([2, 3, 2, 1])
  expect(aT2.data).toStrictEqual([0, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11])
  expect(aT2.untranspose([3, 2, 0, 1])).toMatchObject(a)
})

test("add_", () => {
  expect(
    new NdArray([2, 3])
      .map_((_, i) => 100 + 2 * i)
      .add_(new NdArray([2, 3]).map_((_, i) => i)),
  ).toMatchObject(new NdArray([2, 3], [100, 103, 106, 109, 112, 115]))

  expect(() => new NdArray([2, 3]).add_(new NdArray([6]))).toThrow(
    "arrayEquals([2,3], [6])",
  )
})

test("logSoftmax_", () => {
  const a = new NdArray(
    [2, 1, 3],
    [1000, 1000, 1000, Math.log(1 / 2), 0, Math.log(1 / 2)],
  )
  const expected = [1 / 3, 1 / 3, 1 / 3, 0.25, 0.5, 0.25]

  const sa = a.clone().logSoftmax_().map_(Math.exp)
  expect(sa.shape).toStrictEqual([2, 1, 3])
  sa.data.forEach((v, i) => {
    expect(v).toBeCloseTo(expected[i], 4)
  })
})

test("mean", () => {
  const a = new NdArray([2, 3], [1, 2, 3, 4, 5, 6])
  const mean = a.mean()
  expect(mean.shape).toStrictEqual([])
  expect(mean.data[0]).toBeCloseTo(3.5)
})

test("dot", () => {
  const a0 = [-1, 0, 2, 0]
  const a1 = [0, 0, 0, 2]
  const a2 = [0, 0, 0, 0]
  const a = new NdArray([2, 3, 4], [...a0, ...a1, ...a2, ...a0, ...a1, ...a2])
  const b = new NdArray([2, 4, 5]).map_((_, i) => i)

  const c = a.dot(b)
  expect(c.shape).toStrictEqual([2, 3, 5])
  const c00 = [20, 21, 22, 23, 24]
  const c01 = [30, 32, 34, 36, 38]
  const c02 = [0, 0, 0, 0, 0]
  const c10 = [40, 41, 42, 43, 44]
  const c11 = [70, 72, 74, 76, 78]
  const c12 = [0, 0, 0, 0, 0]
  expect(c.data).toStrictEqual([...c00, ...c01, ...c02, ...c10, ...c11, ...c12])

  expect(
    new NdArray([5, 4, 3]).fill_(11).dot(new NdArray([5, 3, 7]).fill_(2)),
  ).toMatchObject(new NdArray([5, 4, 7]).fill_(66))
})

test("gather, scatter, scatterWrite_", () => {
  const table = new NdArray([2, 5, 2]).map_((_, i) => i)
  expect(table.gather(new NdArray([2, 3], [4, 0, 4, 4, 1, 1]))).toMatchObject(
    new NdArray([2, 3, 2], [8, 9, 0, 1, 8, 9, 18, 19, 12, 13, 12, 13]),
  )

  const idx = new NdArray([2, 5], [1, 2, 0, 0, 0, 1, 1, 1, 1, 1])
  expect(table.scatter(idx, 3)).toMatchObject(
    new NdArray([2, 3, 2], [18, 21, 0, 1, 2, 3, 0, 0, 70, 75, 0, 0]),
  )
  expect(
    new NdArray([2, 3, 2]).fill_(-1).scatterWrite_(table, idx),
  ).toMatchObject(
    new NdArray([2, 3, 2], [8, 9, 0, 1, 2, 3, -1, -1, 18, 19, -1, -1]),
  )
})
