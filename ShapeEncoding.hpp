#pragma once

#include "Types.hpp"
#include <cassert>
#include <cstdint>

namespace mlir::coord {

static constexpr int64_t ScalarShapeEncoding = 0b10;

/// Returns the number of scalar integers in a coordinate of the given shape encoding.
/// Shape encoding uses 2-bit tokens:
///   - Integer:          0b10
///   - Empty tuple:      0b00
///   - Open parenthesis: 0b01
///   - Close parenthesis:0b11
constexpr int64_t getNumIntegersFromShape(uint64_t shape) {
  if (shape == 0b10)
    return 1;
  if (shape == 0b00)
    return 0;

  // Bit length (round up to even)
  int bitLength = 64 - __builtin_clzll(shape);
  if (bitLength % 2 != 0)
    bitLength += 1;

  // Must have at least open + close parens
  assert(bitLength >= 4 && "Invalid encoding: too short");

  // Check for open prefix and close suffix
  uint64_t prefix = (shape >> (bitLength - 2)) & 0b11;
  uint64_t suffix = shape & 0b11;
  assert(prefix == 0b01 && suffix == 0b11 && "Invalid encoding: missing tuple parens");

  // Count 0b10 tokens
  int64_t count = 0;
  for (int pos = 0; pos < bitLength; pos += 2) {
    uint64_t token = (shape >> (bitLength - pos - 2)) & 0b11;
    if (token == 0b10)
      ++count;
  }

  return count;
}


/// Returns true if the given `shape` is a valid coordinate shape encoding.
///
/// Coordinate shapes are encoded using a recursive, fixed-width binary scheme:
/// - Scalar        → "10"
/// - Empty tuple   → "00"
/// - Open paren    → "01"
/// - Close paren   → "11"
///
/// A tuple is represented as:
///   01 [child encodings] 11
///
/// For example:
/// - Scalar         = 0b10
/// - Empty tuple    = 0b00
/// - Tuple of two scalars (1, 2) = 0b01101011
///
/// This function checks that:
///  - The encoding is a multiple of 2 bits
///  - Non-scalar shapes begin with 0b01 and end with 0b11
///  - All tokens are valid
inline bool isValidShapeEncoding(int64_t shape) {
  if (shape == 0b10 || shape == 0b00)
    return true;

  int bitLength = llvm::bit_width(static_cast<uint64_t>(shape));
  if (bitLength % 2 != 0)
    bitLength += 1;

  if (bitLength < 4)
    return false;

  int prefix = (shape >> (bitLength - 2)) & 0b11;
  int suffix = shape & 0b11;
  if (prefix != 0b01 || suffix != 0b11)
    return false;

  int nesting = 0;
  for (int pos = 0; pos < bitLength; pos += 2) {
    int token = (shape >> (bitLength - pos - 2)) & 0b11;
    switch (token) {
      case 0b01: nesting++; break;       // open paren
      case 0b11: nesting--;              // close paren
               if (nesting < 0) return false;
               break;
      case 0b10: case 0b00: break;       // scalar, empty tuple
      default: return false;
    }
  }

  return nesting == 0;
}


/// Recursively compute the shape encoding of a `Coord`-structured type.
/// The input type must be:
///   - i64 → scalar coordinate, encoding "10"
///   - coord.coord<shape> → returns its shape directly
///   - tuple<T...> → encodes each element recursively with open "01" and close "11"
/// Encoding tokens:
///   - Integer:      "10"
///   - Empty tuple:  "00"
///   - Open tuple:   "01"
///   - Close tuple:  "11"
inline int64_t computeShapeOfCoord(mlir::Type type) {
  using namespace mlir::coord;

  if (type.isInteger(64)) {
    return 0b10; // scalar
  }

  if (auto coordTy = dyn_cast<CoordType>(type)) {
    return coordTy.getShape(); // use encoded shape directly
  }

  auto tupleTy = dyn_cast<TupleType>(type);
  if (!tupleTy)
    llvm::report_fatal_error("expected a coordinate-compatible type (i64, coord.coord, or tuple)");

  auto types = tupleTy.getTypes();
  if (types.empty()) return 0b00; // empty tuple

  int64_t bits = 0b01; // open tuple
  int totalBits = 2;

  for (auto t : types) {
    int64_t childBits = computeShapeOfCoord(t);
    int childLength = llvm::bit_width<uint64_t>(childBits);

    // Clamp to even width of at least 2 bits
    if (childLength < 2) childLength = 2;
    if (childLength % 2 != 0) ++childLength;

    bits = (bits << childLength) | childBits;
    totalBits += childLength;
  }

  bits = (bits << 2) | 0b11; // close tuple
  totalBits += 2;

  return bits;
}


}
