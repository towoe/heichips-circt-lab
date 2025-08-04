#ifndef COMB_ADD_OPTIMIZE_H
#define COMB_ADD_OPTIMIZE_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace combaddoptimize {
#define GEN_PASS_DECL
#include "CombAddOptimize/CombAddOptimize.h.inc"

#define GEN_PASS_REGISTRATION
#include "CombAddOptimize/CombAddOptimize.h.inc"
}  // namespace combaddoptimize
}  // namespace mlir

#endif
