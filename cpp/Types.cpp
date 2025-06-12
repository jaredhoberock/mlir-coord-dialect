#include "Dialect.hpp"
#include "Types.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

using namespace mlir;
using namespace mlir::coord;

#define GET_TYPEDEF_CLASSES
#include "Types.cpp.inc"

namespace mlir::coord {

LogicalResult CoordType::unifyWith(Type ty, ModuleOp, llvm::function_ref<InFlightDiagnostic()> emitError) const {
  if (!isCoordLike(ty)) {
    return emitError() << "'" << ty << "' is not coord-like";
  }
  return success();
}

LogicalResult PolyType::unifyWith(Type ty, ModuleOp, llvm::function_ref<InFlightDiagnostic()> emitError) const {
  if (!isCoordLike(ty)) {
    return emitError() << ty << " is not coord-like";
  }
  return success();
}

void CoordDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Types.cpp.inc"
  >();
}

bool isCoordLike(Type ty) {
  if (isa<CoordType>(ty)) {
    return true;
  }

  if (isa<PolyType>(ty)) {
    return true;
  }

  if (ty.isInteger()) {
    return true;
  }

  if (auto tup = dyn_cast<TupleType>(ty)) {
    return llvm::all_of(tup.getTypes(), isCoordLike);
  }

  return false;
}

bool isConcreteCoordLike(Type ty) {
  return isCoordLike(ty) and !trait::containsSymbolicType(ty);
}

static bool isEmptyTupleType(Type ty) {
  if (auto tupleTy = dyn_cast<TupleType>(ty)) {
    return tupleTy.getTypes().empty();
  }
  return false;
}

static bool isNonEmptyTupleType(Type ty) {
  if (auto tupleTy = dyn_cast<TupleType>(ty)) {
    return !tupleTy.getTypes().empty();
  }
  return false;
}

// see https://github.com/jaredhoberock/coord-rust/blob/4a51a2d372b8184deb83b36841186566dcc63ccf/src/coord.rs#L32
bool isWeaklyCongruentTo(Type a, Type b) {
  // terminal case 0: both are integers
  if (a.isInteger() and a == b)
    return true;

  // terminal case 1: both are empty tuples
  if (isEmptyTupleType(a) and isEmptyTupleType(b))
    return true;

  // recursive case 0: integer vs non-empty tuple
  if (a.isInteger() and isNonEmptyTupleType(b)) {
    auto tupleTy = dyn_cast<TupleType>(b);
    return llvm::all_of(tupleTy.getTypes(), [&](Type elemTy) {
      return isWeaklyCongruentTo(a, elemTy);
    });
  }

  // recursive case 1: tuples with equal non-zero rank
  if (isNonEmptyTupleType(a) and isNonEmptyTupleType(b)) {
    TupleType tupleA = dyn_cast<TupleType>(a);
    TupleType tupleB = dyn_cast<TupleType>(b);
    if (tupleA.getTypes().size() == tupleB.getTypes().size()) {
      auto zipped = llvm::zip(tupleA.getTypes(), tupleB.getTypes());
      return llvm::all_of(zipped, [](auto pair) {
        auto [a, b] = pair;
        return isWeaklyCongruentTo(a, b);
      });
    }
  }

  // terminal case 3: all other cases are not weakly congruent
  return false;
}

LogicalResult verifyIsWeaklyCongruentTo(std::optional<Location> loc, Type a, Type b) {
  if (!isWeaklyCongruentTo(a,b)) {
    if (loc)
      return emitError(*loc) << "expected " << a << " to be weakly congruent to " << b;
    return failure();
  }
  return success();
}

bool isCongruentTo(Type a, Type b) {
  // terminal case 0: both are integers
  if (a.isInteger() and a == b)
    return true;

  // recursive case: both are tuples
  if (isa<TupleType>(a) and isa<TupleType>(b)) {
    auto tupleA = dyn_cast<TupleType>(a);
    auto tupleB = dyn_cast<TupleType>(b);
    if (tupleA.getTypes().size() == tupleB.getTypes().size()) {
      auto zipped = llvm::zip(tupleA.getTypes(), tupleB.getTypes());
      return llvm::all_of(zipped, [](auto pair) {
        auto [a, b] = pair;
        return isCongruentTo(a, b);
      });
    }
  }

  // terminal case 1: all other cases are not congruent
  return false;
}

LogicalResult verifyIsCongruentTo(std::optional<Location> loc, Type a, Type b) {
  if (!isCongruentTo(a, b)) {
    if (loc)
      emitError(*loc) << "expected '" << a << "' to be congruent to '" << b << "'";
    return failure();
  }
  return success();
}

bool areCongruent(TypeRange types) {
  if (types.empty())
    return true;

  Type first = types[0];
  return llvm::all_of(types, [=](Type ty) {
    return isCongruentTo(first, ty);
  });
}

Type inferInnerProductReturnType(std::optional<Location> loc, Type a, Type b) {
  if (failed(verifyIsCongruentTo(loc, a, b)))
    return Type();
  return IntegerType::get(a.getContext(), 64);
}

LogicalResult verifyNonEmptyTuplesWithEqualNonZeroSize(std::optional<Location> loc, Type a, Type b) {
  if (!isNonEmptyTupleType(a)) {
    if (loc)
      emitError(*loc) << "expected '" << a << "' to be a non-empty tuple type";
    return failure();
  }

  if (!isNonEmptyTupleType(b)) {
    if (loc)
      emitError(*loc) << "expected '" << b << "' to be a non-empty tuple type";
    return failure();
  }

  auto tupleA = dyn_cast<TupleType>(a);
  auto tupleB = dyn_cast<TupleType>(b);
  if (tupleA.getTypes().size() != tupleB.getTypes().size()) {
    if (loc)
      emitError(*loc) << "expected '" << a << "' to have the same number of element types as '" << b << "'";
    return failure();
  }

  return success();
}

Type inferWeakProductReturnType(std::optional<Location> loc, Type a, Type b) {
  if (failed(verifyIsWeaklyCongruentTo(loc, a, b)))
    return Type();

  // when a and b are congruent, weak_product is equivalent to inner_product
  if (isCongruentTo(a,b))
    return inferInnerProductReturnType(loc, a,b);

  MLIRContext *ctx = a.getContext();

  // recursive case 0: scalar vs non-empty tuple
  // intN * (a, b, ...) => (intN * a, intN * b, ...)
  if (a.isInteger() and isNonEmptyTupleType(b)) {
    auto tupleTy = dyn_cast<TupleType>(b);
    SmallVector<Type> resultElementTypes;
    for (auto elemTy : tupleTy.getTypes()) {
      resultElementTypes.push_back(inferWeakProductReturnType(loc, a, elemTy));
    }
    return TupleType::get(ctx, resultElementTypes);
  }

  // recursive case 1: tuples with equal non-zero rank
  if (failed(verifyNonEmptyTuplesWithEqualNonZeroSize(loc, a, b)))
    return Type();

  auto tupleA = dyn_cast<TupleType>(a);
  auto tupleB = dyn_cast<TupleType>(b);

  // map inferWeakProductReturnType across the element types of tupleA & tupleB
  SmallVector<Type> childProductTypes;
  for (auto [a,b] : llvm::zip(tupleA.getTypes(), tupleB.getTypes())) {
    childProductTypes.push_back(inferWeakProductReturnType(loc, a,b));
  }

  // if all product types are congruent, then the result type is simply this type
  Type firstChild = childProductTypes[0];
  bool allCongruent = llvm::all_of(childProductTypes, [&](Type child) {
    return isCongruentTo(firstChild, child);
  });

  if (allCongruent) {
    return firstChild;
  }

  // otherwise, the result is a tuple of the children
  return TupleType::get(ctx, childProductTypes);
}

} // end mlir::coord;
