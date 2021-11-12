//===- ComplexExprTest.cpp -- ComplexExpr unit tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/Complex.h"
#include "gtest/gtest.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Support/InitFIR.h"
#include "flang/Optimizer/Support/KindMapping.h"

struct ComplexTest : public testing::Test {
public:
  void SetUp() override {
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();

    // Set up a Module with a dummy function operation inside.
    // Set the insertion point in the function entry block.
    mlir::ModuleOp mod = builder.create<mlir::ModuleOp>(loc);
    mlir::FuncOp func = mlir::FuncOp::create(
        loc, "func1", builder.getFunctionType(llvm::None, llvm::None));
    auto *entryBlock = func.addEntryBlock();
    mod.push_back(mod);
    builder.setInsertionPointToStart(entryBlock);

    fir::support::loadDialects(context);
    kindMap = std::make_unique<fir::KindMapping>(&context);
    firBuilder = std::make_unique<fir::FirOpBuilder>(mod, *kindMap);
    helper = std::make_unique<fir::factory::Complex>(*firBuilder, loc);

    // Init commonly used types
    realTy = mlir::FloatType::getF32(&context);
    cType1 = fir::ComplexType::get(&context, 4);

    // Init commonly used numbers
    rOne = firBuilder->createRealConstant(loc, realTy, 1u);
    rTwo = firBuilder->createRealConstant(loc, realTy, 2u);
    rThree = firBuilder->createRealConstant(loc, realTy, 3u);
    rFour = firBuilder->createRealConstant(loc, realTy, 4u);
  }

  mlir::MLIRContext context;
  std::unique_ptr<fir::KindMapping> kindMap;
  std::unique_ptr<fir::FirOpBuilder> firBuilder;
  std::unique_ptr<fir::factory::Complex> helper;

  // Commonly used real/complex types
  mlir::FloatType realTy;
  fir::ComplexType cType1;

  // Commonly used real numbers
  mlir::Value rOne;
  mlir::Value rTwo;
  mlir::Value rThree;
  mlir::Value rFour;
};

TEST_F(ComplexTest, verifyTypes) {
  mlir::Value cVal1 = helper->createComplex(cType1, rOne, rTwo);
  mlir::Value cVal2 = helper->createComplex(4, rOne, rTwo);
  EXPECT_TRUE(fir::isa_complex(cVal1.getType()));
  EXPECT_TRUE(fir::isa_complex(cVal2.getType()));
  EXPECT_TRUE(fir::isa_real(helper->getComplexPartType(cVal1)));
  EXPECT_TRUE(fir::isa_real(helper->getComplexPartType(cVal2)));

  mlir::Value real1 = helper->extractComplexPart(cVal1, /*isImagPart=*/false);
  mlir::Value imag1 = helper->extractComplexPart(cVal1, /*isImagPart=*/true);
  mlir::Value real2 = helper->extractComplexPart(cVal2, /*isImagPart=*/false);
  mlir::Value imag2 = helper->extractComplexPart(cVal2, /*isImagPart=*/true);
  EXPECT_EQ(realTy, real1.getType());
  EXPECT_EQ(realTy, imag1.getType());
  EXPECT_EQ(realTy, real2.getType());
  EXPECT_EQ(realTy, imag2.getType());

  mlir::Value cVal3 =
      helper->insertComplexPart(cVal1, rThree, /*isImagPart=*/false);
  mlir::Value cVal4 =
      helper->insertComplexPart(cVal3, rFour, /*isImagPart=*/true);
  EXPECT_TRUE(fir::isa_complex(cVal4.getType()));
  EXPECT_TRUE(fir::isa_real(helper->getComplexPartType(cVal4)));
}
