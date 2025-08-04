#include "CombAddOptimize/CombAddOptimize.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/Naming.h"
#include "llvm/Support/Compiler.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir::dataflow;
using namespace circt;
using namespace circt::comb;

namespace mlir {
namespace combaddoptimize {

#define GEN_PASS_DEF_REDUCECOMBADDPASS
#include "CombAddOptimize/CombAddOptimize.h.inc"

// Retrieve the range of a value `v` and store it in `range`. Returns failure
// This uses the `DataFlowSolver` to find the range of the value.
// Compare: llvm/mlir/lib/Dialect/Arith/Transforms/IntRangeOptimizations.cpp
// TODO: Implement this function
static std::optional<ConstantIntRanges> retrieveRange(DataFlowSolver &solver,
                                                      Value value) {
  // TODO: Use the solver to get the value if it is not uninitialized

  // Return the range
  return inferredRange;
}

namespace {
struct AddOpPattern : public OpRewritePattern<AddOp> {
  AddOpPattern(MLIRContext *context, DataFlowSolver &solver)
      : OpRewritePattern(context), solver(solver) {}
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    // TODO: Only work for `AddOp` with two inputs and one result

    Location loc = op.getLoc();
    auto opWidth = op.getType().getIntOrFloatBitWidth();

    // TODO: Get the range for each value in the operation

    // TODO: Get the number of bits which are not needed

    // TODO: Only work for the case that we can reduce the width and that we do
    // not remove the complete operation

    auto newWidth = opWidth - removeWidth;
    // TODO: We want to replace the current AddOp with a series of other
    // operations, all with the aim to have a new AddOp with a smaller width.
    // The return value should stay the same.
    // Operations to be used:
    //   - ExtractOp
    //   - AddOp
    //   - ConstantOp
    //   - ConcatOp

    Value lhs = op.getOperand(0);

    // TODO: Fill in here

    rewriter.replaceOp(op, replaceOp);

    return success();
  }

 private:
  DataFlowSolver &solver;
};

struct CombAddOptimize : public impl::ReduceCombAddPassBase<CombAddOptimize> {
  void runOnOperation() override {
    auto op = getOperation();
    MLIRContext *context = op->getContext();
    DataFlowSolver solver;
    solver.load<DeadCodeAnalysis>();
    solver.load<IntegerRangeAnalysis>();
    if (failed(solver.initializeAndRun(op))) return signalPassFailure();

    RewritePatternSet patterns(context);
    patterns.add<AddOpPattern>(patterns.getContext(), solver);

    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      signalPassFailure();
  }
};
}  // namespace

}  // namespace combaddoptimize
}  // namespace mlir

namespace mlir {
// Pass plugin registration mechanism.
/// Necessary symbol to register the pass plugin.
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "CombAddOptimize", "v0.1",
          []() { mlir::combaddoptimize::registerReduceCombAddPass(); }};
}
}  // namespace mlir
