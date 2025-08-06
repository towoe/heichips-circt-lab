lib_ext := if os() == "macos" { ".dylib" } else { ".so" }

build:
	mkdir build; cd build; \
	cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..; \
	ninja

run-test:
	../circt/build/bin/circt-opt -load-pass-plugin=build/CombAddOptimize{{lib_ext}} -pass-pipeline='builtin.module(reduce-comb-add-width)' test/basic.mlir

create-chisel-simple:
	cd example-add-gen; \
	scala-cli add-simple.scala

run-chisel-simple:
	../circt/build/bin/circt-opt --firrtl-lower-layers --lower-firrtl-to-hw example-add-gen/AddSimple.mlir > example-add-gen/AddSimple-lowered.mlir; \
	../circt/build/bin/circt-opt -load-pass-plugin=build/CombAddOptimize{{lib_ext}} -pass-pipeline='builtin.module(reduce-comb-add-width)' example-add-gen/AddSimple-lowered.mlir > example-add-gen/AddSimple-optimized.mlir

create-chisel-uneven-addition:
	cd example-add-gen; \
	scala-cli uneven-addition.scala

run-chisel-uneven-addition:
	../circt/build/bin/circt-opt --firrtl-lower-layers --lower-firrtl-to-hw example-add-gen/UnevenAddition.mlir > example-add-gen/UnevenAddition-lowered.mlir; \
	../circt/build/bin/circt-opt -load-pass-plugin=build/CombAddOptimize{{lib_ext}} -pass-pipeline='builtin.module(reduce-comb-add-width)' example-add-gen/UnevenAddition-lowered.mlir > example-add-gen/UnevenAddition-optimized.mlir
