//===-- Lower/IO.h -- lower I/O statements ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Experimental IO lowering to FIR + runtime. The Runtime design is under
/// design.
///
/// FIXME This interface is also not final. Should it be based on parser::..
/// nodes and lower expressions as needed or should it get every expression
/// already lowered as mlir::Value? (currently second options, not sure it will
/// provide enough information for complex IO statements).
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_IO_H
#define FORTRAN_LOWER_IO_H

#include "llvm/ADT/DenseMap.h"

namespace mlir {
class Value;
} // namespace mlir

namespace Fortran {
namespace parser {
using Label = std::uint64_t;
struct BackspaceStmt;
struct CloseStmt;
struct EndfileStmt;
struct FlushStmt;
struct InquireStmt;
struct OpenStmt;
struct PrintStmt;
struct ReadStmt;
struct RewindStmt;
struct WaitStmt;
struct WriteStmt;
} // namespace parser

namespace lower {

class AbstractConverter;
class BridgeImpl;

namespace pft {
struct Evaluation;
using LabelEvalMap = llvm::DenseMap<Fortran::parser::Label, Evaluation *>;
} // namespace pft

/// Generate IO calls for BACKSPACE.
void genBackspaceStatement(AbstractConverter &,
                           Fortran::lower::pft::Evaluation &,
                           const parser::BackspaceStmt &);

/// Generate IO calls for CLOSE.
void genCloseStatement(AbstractConverter &, Fortran::lower::pft::Evaluation &,
                       const parser::CloseStmt &);

/// Generate IO calls for ENDFILE.
void genEndfileStatement(AbstractConverter &, Fortran::lower::pft::Evaluation &,
                         const parser::EndfileStmt &);

/// Generate IO calls for FLUSH.
void genFlushStatement(AbstractConverter &, Fortran::lower::pft::Evaluation &,
                       const parser::FlushStmt &);

/// Generate IO calls for INQUIRE.
void genInquireStatement(AbstractConverter &, Fortran::lower::pft::Evaluation &,
                         const parser::InquireStmt &);

/// Generate IO calls for OPEN.
void genOpenStatement(AbstractConverter &, Fortran::lower::pft::Evaluation &,
                      const parser::OpenStmt &);

/// Generate IO calls for PRINT.
void genPrintStatement(AbstractConverter &converter,
                       Fortran::lower::pft::Evaluation &,
                       const parser::PrintStmt &stmt);

/// Generate IO calls for READ.
void genReadStatement(AbstractConverter &converter,
                      Fortran::lower::pft::Evaluation &,
                      const parser::ReadStmt &stmt);

/// Generate IO calls for REWIND.
void genRewindStatement(AbstractConverter &, Fortran::lower::pft::Evaluation &,
                        const parser::RewindStmt &);

/// Generate IO calls for WAIT.
void genWaitStatement(AbstractConverter &, Fortran::lower::pft::Evaluation &,
                      const parser::WaitStmt &);

/// Generate IO calls for WRITE.
void genWriteStatement(AbstractConverter &converter,
                       Fortran::lower::pft::Evaluation &,
                       const parser::WriteStmt &stmt);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_IO_H
