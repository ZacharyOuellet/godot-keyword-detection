#!/usr/bin/env python
import os

CacheDir(".scons_cache")

godot_cpp_path = os.environ.get("GODOT_CPP_PATH", "godot-cpp")

env = SConscript(os.path.join(godot_cpp_path, "SConstruct"))

env.Append(CPPPATH=["src/"])

VariantDir("build", "src", duplicate=False)

sources = Glob("build/*.cpp")

output_dir = "addons/mfcc_dtw/bin"

library = env.SharedLibrary(
    target=os.path.join(output_dir, "libmfcc_dtw"),
    source=sources,
)

Default(library)