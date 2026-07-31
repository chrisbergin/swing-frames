import { describe, it, expect } from "vitest";
import { rotationFromMatrix } from "./webcodecs";

/** A quarter-turn transform matrix in 16.16 fixed point (1.0 = 65536). */
const ONE = 65536;
const matrices = {
  0: [ONE, 0, 0, 0, ONE, 0, 0, 0, 0],
  90: [0, ONE, 0, -ONE, 0, 0, 0, 0, 0],
  180: [-ONE, 0, 0, 0, -ONE, 0, 0, 0, 0],
  270: [0, -ONE, 0, ONE, 0, 0, 0, 0, 0],
};

describe("rotationFromMatrix", () => {
  it("reads each quarter turn from the container matrix", () => {
    expect(rotationFromMatrix(matrices[0])).toBe(0);
    expect(rotationFromMatrix(matrices[90])).toBe(90);
    expect(rotationFromMatrix(matrices[180])).toBe(180);
    expect(rotationFromMatrix(matrices[270])).toBe(270);
  });

  it("defaults to no rotation when the matrix is missing", () => {
    expect(rotationFromMatrix(undefined)).toBe(0);
    expect(rotationFromMatrix([])).toBe(0);
  });
});
