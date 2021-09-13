# -*- Python -*-

import os
import platform
import re
import subprocess
import sys

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'Flang'

# testFormat: The test format to use to interpret tests.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.f', '.F', '.ff', '.FOR', '.for', '.f77', '.f90', '.F90',
                   '.ff90', '.f95', '.F95', '.ff95', '.fpp', '.FPP', '.cuf'
                   '.CUF', '.f18', '.F18', '.fir', '.f03', '.F03', '.f08',
                   '.F08']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)


# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.flang_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%llvmshlibdir', config.llvm_shlib_dir))
config.substitutions.append(('%pluginext', config.llvm_plugin_ext))

llvm_config.use_default_substitutions()

# Targets
config.targets = frozenset(config.targets_to_build.split())
for arch in config.targets_to_build.split():
    config.available_features.add(arch.lower() + '-registered-target')

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# If the flang examples are built, add examples to the config
if config.flang_examples:
    config.available_features.add('examples')

# Plugins (loadable modules)
if config.has_plugins:
    config.available_features.add('plugins')

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.flang_obj_root, 'test')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.flang_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
# For out-of-tree builds, path to bbc and tco needs to be added

if config.llvm_tools_dir != config.flang_llvm_tools_dir :
  llvm_config.with_environment('PATH', config.flang_llvm_tools_dir, append_path=True)

if config.flang_standalone_build:
    # For builds with FIR, set path for tco and enable related tests
    if config.flang_llvm_tools_dir != "":
        config.available_features.add('fir')
        if config.llvm_tools_dir != config.flang_llvm_tools_dir:
            llvm_config.with_environment('PATH', config.flang_llvm_tools_dir, append_path=True)

config.substitutions.append(('%B', config.flang_obj_root))
config.substitutions.append(("%L", config.flang_lib_dir))
config.substitutions.append(("%moddir", config.flang_intrinsic_modules_dir)) 
if len(config.macos_sysroot) > 0:
  config.substitutions.append(("%CXX", config.cplusplus_executable + " -isysroot " + config.macos_sysroot))
  config.substitutions.append(("%CC", config.cc + " -isysroot " + config.macos_sysroot))
else:
  config.substitutions.append(("%CXX", config.cplusplus_executable))
  config.substitutions.append(("%CC", config.cc))

# For each occurrence of a flang tool name, replace it with the full path to
# the build directory holding that tool.
tools = [
  ToolSubst('%bbc', command=FindTool('bbc'),
    extra_args=["-intrinsic-module-directory "+config.flang_intrinsic_modules_dir],
    unresolved='fatal')
]

tools.append(ToolSubst('%flang', command=FindTool('flang-new'), unresolved='fatal'))
tools.append(ToolSubst('%flang_fc1', command=FindTool('flang-new'),
 extra_args=['-fc1'], unresolved='fatal'))

# Define some variables to help us test that the flang runtime doesn't depend on
# the C++ runtime libraries. For this we need a C compiler. If for some reason
# we don't have one, we can just disable the test.
if config.cc:
    libruntime = os.path.join(config.flang_lib_dir, 'libFortranRuntime.a')
    libdecimal = os.path.join(config.flang_lib_dir, 'libFortranDecimal.a')
    include = os.path.join(config.flang_src_dir, 'include')

    if os.path.isfile(libruntime) and os.path.isfile(libdecimal) and os.path.isdir(include):
        config.available_features.add('c-compiler')
        tools.append(ToolSubst('%cc', command=config.cc, unresolved='fatal'))
        tools.append(ToolSubst('%libruntime', command=libruntime,
            unresolved='fatal'))
        tools.append(ToolSubst('%libdecimal', command=libdecimal,
            unresolved='fatal'))
        tools.append(ToolSubst('%include', command=include,
            unresolved='fatal'))

if config.flang_standalone_build:
    llvm_config.add_tool_substitutions(tools, [config.flang_llvm_tools_dir])
else:
    llvm_config.add_tool_substitutions(tools, config.llvm_tools_dir)

# Enable libpgmath testing
result = lit_config.params.get("LIBPGMATH")
if result:
    config.environment["LIBPGMATH"] = True

# Preserve the GCC environment for
# Examples/hello.cpp and Lower/end-to-end-character-assignment-driver.cpp
if 'C_INCLUDE_PATH' in os.environ:
    config.environment['C_INCLUDE_PATH'] = os.environ.get('C_INCLUDE_PATH')
if 'CPLUS_INCLUDE_PATH' in os.environ:
    config.environment['CPLUS_INCLUDE_PATH'] = os.environ.get('CPLUS_INCLUDE_PATH')
