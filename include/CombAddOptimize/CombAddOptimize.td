#ifndef COMB_ADD_OPTIMIZE
#define COMB_ADD_OPTIMIZE

include "mlir/Pass/PassBase.td"

def ReduceCombAddPass: Pass<"reduce-comb-add-width"> {
let summary = "Reduce with width of add operation";
let description = [{
	This pass checks the input ranges of add operations and reduces the
	size of the result.
}];
}

#endif // COMB_ADD_OPTIMIZE
