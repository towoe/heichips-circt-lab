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
static std::optional<ConstantIntRanges> retrieveRange(DataFlowSolver &solver,
                                                      Value value) {
  auto *maybeInferredRange =
      solver.lookupState<IntegerValueRangeLattice>(value);
  if (!maybeInferredRange || maybeInferredRange->getValue().isUninitialized())
    return std::nullopt;
  const ConstantIntRanges &inferredRange =
      maybeInferredRange->getValue().getValue();

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
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "Expected 2 operands and 1 result");
    Location loc = op.getLoc();
    auto opWidth = op.getType().getIntOrFloatBitWidth();

    auto rangeOp0 = retrieveRange(solver, op.getOperand(0));
    auto rangeOp1 = retrieveRange(solver, op.getOperand(1));
    auto rangeRes = retrieveRange(solver, op.getResult());
    if (!rangeOp0 || !rangeOp1 || !rangeRes)
      return rewriter.notifyMatchFailure(op,
                                         "no ranges for operands or result");

    auto removeWidth = rangeRes.value().umax().countLeadingZeros();
    removeWidth =
        std::min(removeWidth, rangeOp0.value().umax().countLeadingZeros());
    removeWidth =
        std::min(removeWidth, rangeOp1.value().umax().countLeadingZeros());

    // not remove the complete operation
    if (removeWidth == 0)
      return rewriter.notifyMatchFailure(op, "no bits to remove");
    if (removeWidth == opWidth)
      return rewriter.notifyMatchFailure(
          op, "all bits to remove - replace by zero");

    auto newWidth = opWidth - removeWidth;

    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);

    // Create a replacement type for the extracted bits
    auto replaceType = rewriter.getIntegerType(newWidth);

    // Extract the lsbs from each operand
    auto extractLhsOp = ExtractOp::create(rewriter, loc, replaceType, lhs, 0);
    auto extractRhsOp = ExtractOp::create(rewriter, loc, replaceType, rhs, 0);
    auto narrowOp = AddOp::create(rewriter, loc, extractLhsOp, extractRhsOp);

    // Concatenate zeros to match the original operator width
    auto zero =
        hw::ConstantOp::create(rewriter, loc, APInt::getZero(removeWidth));
    auto replaceOp = ConcatOp::create(rewriter, loc, op.getType(),
                                      ValueRange{zero, narrowOp});

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
