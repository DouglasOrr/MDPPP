// ndarray is a basic N-dimensional array library, styled on numpy
//
//   NdArray -- C-contiguous array that owns its data buffer (non-view)

// Array utilities

export function arrayEquals<T>(a: T[], b: T[]): boolean {
  return a.length === b.length && a.every((value, i) => value === b[i])
}

export function arrayProduct(array: number[]): number {
  return array.reduce((a, b) => a * b, 1)
}

// Assertions

export type Message = string | (() => string)

export function messageString(message: Message): string {
  return typeof message === "function" ? message() : message
}

export function assert(condition: boolean, message: Message): void {
  if (!condition) {
    throw Error(messageString(message))
  }
}

export function assertEquals(a: any, b: any, message: Message = ""): void {
  assert(a === b, () => `Expected ${a} === ${b};  ${messageString(message)}`)
}

export function assertArrayEquals<T>(
  a: T[],
  b: T[],
  message: Message = "",
): void {
  assert(
    arrayEquals(a, b),
    () =>
      `Expected arrayEquals([${a.toString()}], [${b.toString()}])` +
      `;  ${messageString(message)}`,
  )
}

// NdArray

// C-Contiguous multidimensional array
export class NdArray {
  shape: number[]
  data: number[]

  constructor(shape: number[], data: number[] | null = null) {
    this.shape = shape
    const nelement = arrayProduct(shape)
    if (data === null) {
      this.data = Array(nelement)
    } else {
      assertEquals(
        data.length,
        nelement,
        "NdArray shape (nelement) != data.length",
      )
      this.data = data
    }
  }

  // Basics

  ndim(): number {
    return this.shape.length
  }

  strides(): number[] {
    const result = Array(this.shape.length)
    let stride = 1
    for (let i = result.length - 1; i !== -1; --i) {
      result[i] = stride
      stride *= this.shape[i]
    }
    return result
  }

  clone(): NdArray {
    return new NdArray(this.shape, [...this.data])
  }

  // Initialisation

  fill_(value: number): NdArray {
    this.data.fill(value)
    return this
  }

  rand_(low: number = 0, high: number = 1): NdArray {
    return this.map_(() => Math.random() * (high - low) + low)
  }

  // Transformations

  map_(fn: (x: number, i: number) => number): NdArray {
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] = fn(this.data[i], i)
    }
    return this
  }

  map2_(
    rhs: NdArray,
    fn: (lhs: number, rhs: number, i: number) => number,
  ): NdArray {
    assertArrayEquals(
      this.shape,
      rhs.shape,
      "map2 expects this.shape === rhs.shape",
    )
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] = fn(this.data[i], rhs.data[i], i)
    }
    return this
  }

  view_(shape: number[]): NdArray {
    assertEquals(
      arrayProduct(shape),
      this.data.length,
      "view shape should retain same nelement",
    )
    this.shape = shape
    return this
  }

  t(): NdArray {
    const s = this.shape
    const thisSize1 = s[s.length - 1]
    const thisSize2 = s[s.length - 2]
    const groupSize = thisSize1 * thisSize2
    const result = new NdArray([...s.slice(0, -2), thisSize1, thisSize2])
    this.data.forEach((value, i) => {
      const di =
        Math.floor(i / groupSize) * groupSize +
        (i % thisSize1) * thisSize2 +
        Math.floor((i / thisSize1) % thisSize2)
      result.data[di] = value
    })
    return result
  }

  // Dims maps result dimension to this dimension
  transpose(dims: number[]): NdArray {
    assertArrayEquals(
      dims.slice().sort((a, b) => a - b),
      [...Array(this.ndim()).keys()],
      "transpose dims should cover [0..(ndim-1)]",
    )
    const result = new NdArray(dims.map((i) => this.shape[i]))
    const thisStrides = this.strides()
    const resultStrides = result.strides()
    const thisData = this.data
    function copy(dim: number, resultI: number, thisI: number): void {
      const resultS = resultStrides[dim]
      const thisS = thisStrides[dims[dim]]
      if (dim === result.ndim() - 1) {
        for (let i = 0; i < result.shape[dim]; ++i) {
          result.data[resultI + resultS * i] = thisData[thisI + thisS * i]
        }
      } else {
        for (let i = 0; i < result.shape[dim]; ++i) {
          copy(dim + 1, resultI + resultS * i, thisI + thisS * i)
        }
      }
    }
    copy(0, 0, 0)
    return result
  }

  // Dims maps this dimension to destination dimension
  untranspose(dims: number[]): NdArray {
    const transposeDims = Array(dims.length)
    dims.forEach((dim, i) => (transposeDims[dim] = i))
    return this.transpose(transposeDims)
  }

  add_(rhs: NdArray): NdArray {
    assertArrayEquals(this.shape, rhs.shape, "add this.shape === rhs.shape")
    for (let i = 0; i < this.data.length; ++i) {
      this.data[i] += rhs.data[i]
    }
    return this
  }

  // Always over the final dimension
  logSoftmax_(): NdArray {
    const stride = this.shape[this.shape.length - 1]
    for (let j = 0; j < this.data.length; j += stride) {
      const slice = this.data.slice(j, j + stride)
      const max = slice.reduce((a, b) => Math.max(a, b), -Infinity)
      const logSumExp = Math.log(
        slice.reduce((sum, x) => sum + Math.exp(x - max), 0),
      )
      for (let i = j; i < j + stride; ++i) {
        this.data[i] -= max + logSumExp
      }
    }
    return this
  }

  mean(): NdArray {
    return new NdArray(
      [],
      [this.data.reduce((a, b) => a + b, 0) / this.data.length],
    )
  }

  // (*g, m, k) @ (*g, k, n) -> (*g, m, n)
  dot(rhs: NdArray): NdArray {
    assertArrayEquals(
      this.shape.slice(0, -2),
      rhs.shape.slice(0, -2),
      "dot - group dimensions should match",
    )
    const ndim = this.ndim()
    assertEquals(
      this.shape[ndim - 1],
      rhs.shape[ndim - 2],
      "dot - concat dimension should match (..., :, ?).dot((..., ?, :))",
    )
    const m = this.shape[ndim - 2]
    const k = this.shape[ndim - 1]
    const n = rhs.shape[ndim - 1]
    const result = new NdArray([...this.shape.slice(0, -2), m, n])
    // Naive matmul
    for (let i = 0; i < result.data.length; ++i) {
      let sum = 0.0
      const thisi = Math.floor(i / n) * k
      const rhsi = Math.floor(i / (m * n)) * k * n + (i % n)
      for (let j = 0; j < k; ++j) {
        sum += this.data[thisi + j] * rhs.data[rhsi + n * j]
      }
      result.data[i] = sum
    }
    return result
  }

  // Gather/scatter

  // Grouped on leading dimensions of indices, broadcast on trailing dimensions of this
  // E.g. {shape: [2, 3, 4]}.gather({shape: [2, 5]}) => {shape: [2, 5, 4]}
  gather(indices: NdArray): NdArray {
    assertArrayEquals(
      indices.shape.slice(0, -1),
      this.shape.slice(0, indices.ndim() - 1),
      "gather - leading dimensions of indices should match this.shape",
    )
    const rowShape = this.shape.slice(indices.ndim())
    const rowSize = arrayProduct(rowShape)
    const gatherSize = indices.shape[indices.ndim() - 1]
    const tableSize = this.shape[indices.ndim() - 1]
    const result = new NdArray(indices.shape.concat(rowShape))
    indices.data.forEach((index, i) => {
      const di =
        Math.floor(i / gatherSize) * tableSize * rowSize + index * rowSize
      result.data.splice(
        i * rowSize,
        rowSize,
        ...this.data.slice(di, di + rowSize),
      )
    })
    return result
  }

  // Grouped on leading dimensions of indices, broadcast on trailing dimensions of this
  // E.g. {shape: [2, 5, 4]}.scatter({shape: [2, 5]}, 3) => {shape: [2, 3, 4]}
  scatter(indices: NdArray, size: number): NdArray {
    assertArrayEquals(
      indices.shape,
      this.shape.slice(0, indices.ndim()),
      "scatter - indices shape should match this.shape",
    )
    const rowShape = this.shape.slice(indices.ndim())
    const rowSize = arrayProduct(rowShape)
    const tableSize = indices.shape[indices.ndim() - 1]
    const result = new NdArray([
      ...indices.shape.slice(0, -1),
      size,
      ...rowShape,
    ]).fill_(0)
    indices.data.forEach((index, i) => {
      const ri = Math.floor(i / tableSize) * size * rowSize + index * rowSize
      for (let j = 0; j < rowSize; ++j) {
        result.data[ri + j] += this.data[i * rowSize + j]
      }
    })
    return result
  }

  // Grouped on leading dimensions of indices, broadcast on trailing dimensions of src
  // E.g. {shape: [2, 3, 4]}.scatter_write({shape: [2, 5, 4]}, {shape: [2, 5]})
  scatterWrite_(src: NdArray, indices: NdArray): NdArray {
    assertArrayEquals(
      src.shape.slice(0, indices.ndim()),
      indices.shape,
      "scatterWrite_ - leading dimensions of src should match indices.shape",
    )
    assertArrayEquals(
      indices.shape.slice(0, -1),
      this.shape.slice(0, indices.ndim() - 1),
      "scatterWrite_ - indices shape should match this.shape",
    )
    const rowShape = src.shape.slice(indices.ndim())
    assertArrayEquals(
      this.shape.slice(indices.ndim()),
      rowShape,
      "scatterWrite_ - trailing dimensions of src.shape should match this.shape",
    )
    const rowSize = arrayProduct(rowShape)
    const tableSize = indices.shape[indices.ndim() - 1]
    const scatterSize = this.shape[indices.ndim() - 1]
    indices.data.forEach((index, i) => {
      const ri =
        Math.floor(i / tableSize) * scatterSize * rowSize + index * rowSize
      for (let j = 0; j < rowSize; ++j) {
        this.data[ri + j] = src.data[i * rowSize + j]
      }
    })
    return this
  }
}
