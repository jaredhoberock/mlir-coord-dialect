import os
import lit.formats

config.name = "Coord Dialect Tests"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)

llvm_bin_dir = '/home/jhoberock/dev/git/llvm-project-20/build/bin'
coord_plugin_path = os.path.join(os.path.dirname(__file__), '..', 'libcoord_dialect.so')
tuple_plugin_path = os.path.join(os.path.dirname(__file__), '/home/jhoberock/dev/git/mlir-tuple-dialect/cpp', 'libtuple_dialect.so')
trait_plugin_path = os.path.join(os.path.dirname(__file__), '/home/jhoberock/dev/git/mlir-trait-dialect/cpp', 'libtrait_dialect.so')

config.substitutions.append(('mlir-opt', f'{os.path.join(llvm_bin_dir, "mlir-opt")} --load-dialect-plugin={trait_plugin_path} --load-dialect-plugin={tuple_plugin_path} --load-dialect-plugin={coord_plugin_path}'))
config.substitutions.append(('FileCheck', os.path.join(llvm_bin_dir, 'FileCheck')))
