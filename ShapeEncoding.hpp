#pragma once

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


}
