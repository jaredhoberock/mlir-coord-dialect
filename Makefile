#LLVM_BIN_PATH = /home/jhoberock/dev/git/llvm-project-19/build/bin
LLVM_BIN_PATH = /home/jhoberock/dev/git/llvm-project-20/build/bin

LLVM_CONFIG := $(LLVM_BIN_PATH)/llvm-config
TBLGEN := $(LLVM_BIN_PATH)/mlir-tblgen
OPT := $(LLVM_BIN_PATH)/mlir-opt

# Compiler flags
CXX := clang++
CXXFLAGS := -g -fPIC `$(LLVM_CONFIG) --cxxflags`

# LLVM/MLIR libraries
#MLIR_INCLUDE = /usr/lib/llvm-19/include
MLIR_INCLUDE = /home/jhoberock/dev/git/llvm-project-20/install/include

INCLUDES := -I $(MLIR_INCLUDE)

# Dialect library sources (everything except main)
DIALECT_SOURCES := coord_c.cpp Canonicalization.cpp Dialect.cpp Lowering.cpp Ops.cpp Types.cpp
DIALECT_OBJECTS := $(DIALECT_SOURCES:.cpp=.o)

# Generated files
GENERATED := Dialect.hpp.inc Dialect.cpp.inc Ops.hpp.inc Ops.cpp.inc Types.hpp.inc Types.cpp.inc

.PHONY: all clean

all: libcoord_dialect.so

# TableGen rules
Dialect.hpp.inc: Dialect.td
	$(TBLGEN) --gen-dialect-decls $(INCLUDES) $< -o $@

Dialect.cpp.inc: Dialect.td
	$(TBLGEN) --gen-dialect-defs $(INCLUDES) $< -o $@

Ops.hpp.inc: Ops.td
	$(TBLGEN) --gen-op-decls $(INCLUDES) $< -o $@

Ops.cpp.inc: Ops.td
	$(TBLGEN) --gen-op-defs $(INCLUDES) $< -o $@

Types.hpp.inc: Types.td
	$(TBLGEN) --gen-typedef-decls $(INCLUDES) $< -o $@

Types.cpp.inc: Types.td
	$(TBLGEN) --gen-typedef-defs $(INCLUDES) $< -o $@

# Object file rules
%.o: %.cpp $(GENERATED)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

PLUGIN_OBJECTS := $(DIALECT_OBJECTS) Plugin.o

libcoord_dialect.so: $(PLUGIN_OBJECTS)
	$(CXX) -shared $^ -o $@

.PHONY: test
test: libcoord_dialect.so
	@echo "Running coord dialect tests..."
	./venv/bin/lit tests

clean:
	rm -f *.o *.inc *.a *.so
