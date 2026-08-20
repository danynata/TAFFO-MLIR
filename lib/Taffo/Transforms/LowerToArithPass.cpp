#include "Taffo/Transforms/LowerToArithPass.h"
#include "Taffo/Dialect/Attributes.h"
#include "Taffo/Dialect/Taffo.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Taffo/Dialect/Ops.h"

namespace mlir::taffo {
#define GEN_PASS_DEF_LOWERTOARITHPASS
#include "Taffo/Transforms/Passes.h.inc"
} // namespace mlir::taffo

using namespace ::mlir::taffo;

namespace mlir {
class LowerToArithPass
    : public mlir::taffo::impl::LowerToArithPassBase<LowerToArithPass> {
public:
  using LowerToArithPassBase::LowerToArithPassBase;

  class TaffoToArithTypeConverter : public mlir::TypeConverter {
  public:
    TaffoToArithTypeConverter(MLIRContext *ctx) {
      addConversion([](Type type) { return type; });
      addConversion([ctx](RealType type) -> Type {
        return IntegerType::get(ctx, type.getBitwidth(),
                                IntegerType::SignednessSemantics::Signless);
      });

      addTargetMaterialization(
          [&](mlir::OpBuilder &builder, mlir::TypeRange resultType,
              mlir::ValueRange inputs, mlir::Location loc,
              mlir::Type) -> llvm::SmallVector<mlir::Value> {
            if (inputs.size() != 1) {
              return {};
            }

            auto CastToRealOp =
                builder.create<mlir::UnrealizedConversionCastOp>(
                    loc, resultType, inputs);

            llvm::SmallVector<mlir::Value> result;
            result.push_back(CastToRealOp.getResult(0));
            return result;
          });

      addSourceMaterialization(
          [&](mlir::OpBuilder &builder, mlir::Type resultType,
              mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
            if (inputs.size() != 1) {
              return {};
            }

            auto CastToRealOp =
                builder.create<mlir::UnrealizedConversionCastOp>(
                    loc, resultType, inputs);

            return CastToRealOp.getResult(0);
          });
    }
  };

  static int getExp(Value v) {
    return ::llvm::dyn_cast<RealType>(v.getType()).getExponent();
  }

  static bool getSignd(Value v) {
    return ::llvm::dyn_cast<RealType>(v.getType()).getSignd();
  }

  template <typename T1, typename T2>
  static LogicalResult ConvertAddCommon(T1 op, T2 adaptor,
                                        ConversionPatternRewriter &rewriter) {

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    RealType resType = ::llvm::dyn_cast<RealType>(op->getResult(0).getType());

    const int targetWidth = resType.getBitwidth();

    auto buildIntAttr = [targetWidth](Builder b, int64_t value) -> IntegerAttr {
      return b.getIntegerAttr(b.getIntegerType(targetWidth), value);
    };

    Value ogLhs = op.getLhs();
    Value ogRhs = op.getRhs();
    int lhsExp = getExp(ogLhs);
    int rhsExp = getExp(ogRhs);
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // reconcile arguments of different signedness
    if (resType.getSignd() && (getSignd(ogRhs) != getSignd(ogLhs))) {

      if (!getSignd(ogLhs)) {
        lhsExp += 1;
        lhs = b.create<arith::ShRUIOp>(
                   lhs, b.create<arith::ConstantOp>(buildIntAttr(b, 1)))
                  .getResult();
      }

      if (!getSignd(ogRhs)) {
        rhsExp += 1;
        rhs = b.create<arith::ShRUIOp>(
                   rhs, b.create<arith::ConstantOp>(buildIntAttr(b, 1)))
                  .getResult();
      }
    }

    int expDiff = std::abs(rhsExp - lhsExp);
    // if the difference in exponent is large enough that largest number of
    // the smaller operand cannot be represented by the larger operand,
    // we delete the op
    if (expDiff > targetWidth) {
      Value maxExpArg = rhsExp > lhsExp ? rhs : lhs;
      rewriter.replaceOp(op, maxExpArg);
      return success();
    }

    Value res;

    if (expDiff != 0) {
      Value to_shift = rhsExp < lhsExp ? rhs : lhs;
      Value no_shift = rhsExp > lhsExp ? rhs : lhs;

      arith::ConstantOp shift_amount =
          b.create<arith::ConstantOp>(buildIntAttr(b, expDiff));
      Value ShOp =
          resType.getSignd()
              ? b.create<arith::ShRSIOp>(to_shift, shift_amount).getResult()
              : b.create<arith::ShRUIOp>(to_shift, shift_amount).getResult();
      res = b.create<arith::AddIOp>(no_shift, ShOp);
    } else {
      res = b.create<arith::AddIOp>(lhs, rhs);
    }

    int resExpDiff = resType.getExponent() - std::max(rhsExp, lhsExp);

    if (resExpDiff == 0) {
      rewriter.replaceOp(op, res);
      return success();
    }

    arith::ConstantOp align_res =
        b.create<arith::ConstantOp>(buildIntAttr(b, std::abs(resExpDiff)));
    res = resExpDiff > 0
              ? resType.getSignd()
                    ? b.create<arith::ShRSIOp>(res, align_res).getResult()
                    : b.create<arith::ShRUIOp>(res, align_res).getResult()
              : b.create<arith::ShLIOp>(res, align_res).getResult();
    rewriter.replaceOp(op, res);
    return success();
  }

  struct ConvertAdd : public OpConversionPattern<AddOp> {
    ConvertAdd(mlir::MLIRContext *context)
        : OpConversionPattern<AddOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(AddOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      return ConvertAddCommon<AddOp, OpAdaptor>(op, adaptor, rewriter);
    }
  };

  template <typename T1, typename T2>
  static LogicalResult ConvertSubCommon(T1 op, T2 adaptor,
                                        ConversionPatternRewriter &rewriter) {

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    RealType resType = ::llvm::dyn_cast<RealType>(op->getResult(0).getType());

    const int targetWidth = resType.getBitwidth();

    auto buildIntAttr = [targetWidth](Builder b, int64_t value) -> IntegerAttr {
      return b.getIntegerAttr(b.getIntegerType(targetWidth), value);
    };

    Value ogLhs = op.getLhs();
    Value ogRhs = op.getRhs();
    int lhsExp = getExp(ogLhs);
    int rhsExp = getExp(ogRhs);
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // reconcile arguments of different signedness (identical to ConvertAdd)
    if (resType.getSignd() && (getSignd(ogRhs) != getSignd(ogLhs))) {

      if (!getSignd(ogLhs)) {
        lhsExp += 1;
        lhs = b.create<arith::ShRUIOp>(
                   lhs, b.create<arith::ConstantOp>(buildIntAttr(b, 1)))
                  .getResult();
      }

      if (!getSignd(ogRhs)) {
        rhsExp += 1;
        rhs = b.create<arith::ShRUIOp>(
                   rhs, b.create<arith::ConstantOp>(buildIntAttr(b, 1)))
                  .getResult();
      }
    }

    int expDiff = std::abs(rhsExp - lhsExp);

    // If one operand's exponent dwarfs the other's, the smaller-magnitude
    // operand is negligible at this bitwidth. Unlike addition, subtraction
    // is NOT symmetric here: if rhs dominates, the result is approximately
    // -rhs (not +rhs), since lhs contributes ~0.
    if (expDiff > targetWidth) {
      if (rhsExp > lhsExp) {
        Value zero = b.create<arith::ConstantOp>(buildIntAttr(b, 0));
        Value negRhs = b.create<arith::SubIOp>(zero, rhs).getResult();
        rewriter.replaceOp(op, negRhs);
      } else {
        rewriter.replaceOp(op, lhs);
      }
      return success();
    }

    // Align both operands to the same (coarser/larger) exponent by shifting
    // whichever has the SMALLER exponent right by expDiff. Unlike
    // ConvertAdd's "no_shift"/"to_shift" naming (safe there only because
    // addition is commutative), we track alignedLhs/alignedRhs explicitly
    // so the final SubIOp's operand order always matches the original
    // lhs - rhs, regardless of which one needed shifting.
    Value alignedLhs = lhs;
    Value alignedRhs = rhs;

    if (expDiff != 0) {
      arith::ConstantOp shift_amount =
          b.create<arith::ConstantOp>(buildIntAttr(b, expDiff));
      if (lhsExp < rhsExp) {
        alignedLhs =
            resType.getSignd()
                ? b.create<arith::ShRSIOp>(lhs, shift_amount).getResult()
                : b.create<arith::ShRUIOp>(lhs, shift_amount).getResult();
      } else {
        alignedRhs =
            resType.getSignd()
                ? b.create<arith::ShRSIOp>(rhs, shift_amount).getResult()
                : b.create<arith::ShRUIOp>(rhs, shift_amount).getResult();
      }
    }

    Value res = b.create<arith::SubIOp>(alignedLhs, alignedRhs);

    int resExpDiff = resType.getExponent() - std::max(rhsExp, lhsExp);

    if (resExpDiff == 0) {
      rewriter.replaceOp(op, res);
      return success();
    }

    arith::ConstantOp align_res =
        b.create<arith::ConstantOp>(buildIntAttr(b, std::abs(resExpDiff)));
    res = resExpDiff > 0
              ? resType.getSignd()
                    ? b.create<arith::ShRSIOp>(res, align_res).getResult()
                    : b.create<arith::ShRUIOp>(res, align_res).getResult()
              : b.create<arith::ShLIOp>(res, align_res).getResult();
    rewriter.replaceOp(op, res);
    return success();
  }

  struct ConvertSub : public OpConversionPattern<SubOp> {
    ConvertSub(mlir::MLIRContext *context)
        : OpConversionPattern<SubOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(SubOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      return ConvertSubCommon<SubOp, OpAdaptor>(op, adaptor, rewriter);
    }
  };

  template <typename T1, typename T2>
  static LogicalResult ConvertMultCommon(T1 op, T2 adaptor,
                                         ConversionPatternRewriter &rewriter) {

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    RealType resType = ::llvm::dyn_cast<RealType>(op->getResult(0).getType());

    const int resWidth = resType.getBitwidth();
    const int ogWidth = resWidth;
    const int extWidth = resWidth * 2;

    auto buildNarrowAttr = [ogWidth](Builder b, int64_t value) -> IntegerAttr {
      return b.getIntegerAttr(b.getIntegerType(ogWidth), value);
    };

    auto buildWideAttr = [extWidth](Builder b, int64_t value) -> IntegerAttr {
      return b.getIntegerAttr(b.getIntegerType(extWidth), value);
    };

    Value ogLhs = op.getLhs();
    Value ogRhs = op.getRhs();
    int lhsExp = getExp(ogLhs);
    int rhsExp = getExp(ogRhs);
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // reconcile arguments of different signedness
    if (resType.getSignd() && (getSignd(ogRhs) != getSignd(ogLhs))) {

      if (!getSignd(ogLhs)) {
        lhsExp += 1;
        lhs = b.create<arith::ShRUIOp>(
                   lhs, b.create<arith::ConstantOp>(buildNarrowAttr(b, 1)))
                  .getResult();
      }

      if (!getSignd(ogRhs)) {
        rhsExp += 1;
        rhs = b.create<arith::ShRUIOp>(
                   rhs, b.create<arith::ConstantOp>(buildNarrowAttr(b, 1)))
                  .getResult();
      }
    }

    int implicitExp = rhsExp + lhsExp + resType.getBitwidth();
    int expDiff = resType.getExponent() - implicitExp;

    // extend operands
    lhs = resType.getSignd()
              ? b.create<arith::ExtSIOp>(b.getIntegerType(extWidth), lhs)
                    .getResult()
              : b.create<arith::ExtUIOp>(b.getIntegerType(extWidth), lhs)
                    .getResult();
    rhs = resType.getSignd()
              ? b.create<arith::ExtSIOp>(b.getIntegerType(extWidth), rhs)
                    .getResult()
              : b.create<arith::ExtUIOp>(b.getIntegerType(extWidth), rhs)
                    .getResult();

    Value res = b.create<arith::MulIOp>(lhs, rhs).getResult();

    // this is probably unnecessary as I expect shifts by 0 get folded away
    // TODO: check if this check is necessary
    if (expDiff == 0) {
      rewriter.replaceOp(op, res);
      return success();
    }

    int correctionFactor = expDiff + ogWidth;
    arith::ConstantOp align_res = b.create<arith::ConstantOp>(
        buildWideAttr(b, std::abs(correctionFactor)));

    // this should never be < 0, but it's best to check (if it is, it means
    // that VRA has been broken)
    res = correctionFactor > 0
              ? resType.getSignd()
                    ? b.create<arith::ShRSIOp>(res, align_res).getResult()
                    : b.create<arith::ShRUIOp>(res, align_res).getResult()
              : b.create<arith::ShLIOp>(res, align_res).getResult();

    res = b.create<arith::TruncIOp>(b.getIntegerType(ogWidth), res).getResult();
    rewriter.replaceOp(op, res);
    return success();
  }

  struct ConvertMult : public OpConversionPattern<MultOp> {
    ConvertMult(mlir::MLIRContext *context)
        : OpConversionPattern<MultOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(MultOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      return ConvertMultCommon<MultOp, OpAdaptor>(op, adaptor, rewriter);
    }
  };

  template <typename T1, typename T2>
  static LogicalResult ConvertDivCommon(T1 op, T2 adaptor,
                                        ConversionPatternRewriter &rewriter) {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    RealType resType = ::llvm::dyn_cast<RealType>(op->getResult(0).getType());
    const int resWidth = resType.getBitwidth();
    const int ogWidth = resWidth;
    const int extWidth = resWidth * 2;

    auto buildNarrowAttr = [ogWidth](Builder b, int64_t value) -> IntegerAttr {
      return b.getIntegerAttr(b.getIntegerType(ogWidth), value);
    };

    auto buildWideAttr = [extWidth](Builder b, int64_t value) -> IntegerAttr {
      return b.getIntegerAttr(b.getIntegerType(extWidth), value);
    };

    Value ogNumerator = op.getLhs();
    Value ogDenom = op.getRhs();
    int numExp = getExp(ogNumerator);
    int denomExp = getExp(ogDenom);
    Value numerator = adaptor.getLhs();
    Value denominator = adaptor.getRhs();

    // reconcile arguments of different signedness
    if (resType.getSignd() && (getSignd(ogNumerator) != getSignd(ogDenom))) {
      if (!getSignd(ogNumerator)) {
        numExp += 1;
        numerator =
            b.create<arith::ShRUIOp>(
                 numerator, b.create<arith::ConstantOp>(buildNarrowAttr(b, 1)))
                .getResult();
      }
      if (!getSignd(ogDenom)) {
        denomExp += 1;
        denominator = b.create<arith::ShRUIOp>(
                           denominator,
                           b.create<arith::ConstantOp>(buildNarrowAttr(b, 1)))
                          .getResult();
      }
    }

    // Extend operands to wide type.
    numerator =
        resType.getSignd()
            ? b.create<arith::ExtSIOp>(b.getIntegerType(extWidth), numerator)
                  .getResult()
            : b.create<arith::ExtUIOp>(b.getIntegerType(extWidth), numerator)
                  .getResult();
    denominator =
        resType.getSignd()
            ? b.create<arith::ExtSIOp>(b.getIntegerType(extWidth), denominator)
                  .getResult()
            : b.create<arith::ExtUIOp>(b.getIntegerType(extWidth), denominator)
                  .getResult();

    // Scale numerator by shifting left to preserve fractional precision.
    arith::ConstantOp scale_const =
        b.create<arith::ConstantOp>(buildWideAttr(b, resWidth));
    Value numerator_scaled =
        b.create<arith::ShLIOp>(numerator, scale_const).getResult();

    // Perform division.
    Value res = resType.getSignd()
                    ? b.create<arith::DivSIOp>(numerator_scaled, denominator)
                          .getResult()
                    : b.create<arith::DivUIOp>(numerator_scaled, denominator)
                          .getResult();

    // Compute correction factor.
    // Expected effective exponent after division: (numExp - denomExp -
    // resWidth) We correct by: resType.getExponent() - (numExp - denomExp -
    // resWidth)
    int correctionFactor = resType.getExponent() + resWidth - numExp + denomExp;

    if (correctionFactor != 0) {
      arith::ConstantOp correction_const = b.create<arith::ConstantOp>(
          buildWideAttr(b, std::abs(correctionFactor)));
      res = correctionFactor > 0
                ? (resType.getSignd()
                       ? b.create<arith::ShRSIOp>(res, correction_const)
                             .getResult()
                       : b.create<arith::ShRUIOp>(res, correction_const)
                             .getResult())
                : b.create<arith::ShLIOp>(res, correction_const).getResult();
    }

    // Truncate the result back to narrow type.
    res = b.create<arith::TruncIOp>(b.getIntegerType(ogWidth), res).getResult();
    rewriter.replaceOp(op, res);
    return success();
  }

  struct ConvertDiv : public OpConversionPattern<DivOp> {
    ConvertDiv(mlir::MLIRContext *context)
        : OpConversionPattern<DivOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(DivOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      return ConvertDivCommon<DivOp, OpAdaptor>(op, adaptor, rewriter);
    }
  };

  struct ConvertCastToReal : public OpConversionPattern<CastToRealOp> {
    ConvertCastToReal(mlir::MLIRContext *context)
        : OpConversionPattern<CastToRealOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(CastToRealOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      ImplicitLocOpBuilder b(op.getLoc(), rewriter);

      FloatType fType = ::llvm::dyn_cast<FloatType>(op.getFrom().getType());
      int ogWidth = fType.getWidth();

      if (ogWidth > 64) {
        op->emitOpError()
            << "Conversion from floats bigger than f64 is not yet supported";
        return failure();
      }

      RealType resType = ::llvm::dyn_cast<RealType>(op->getResult(0).getType());

      if (resType.getBitwidth() > 64) {
        op->emitOpError()
            << "Conversion to fixpoints bigger than 64 is not yet supported";
        return failure();
      }

      auto buildIntAttr = [ogWidth](Builder b, int64_t value) -> IntegerAttr {
        return b.getIntegerAttr(b.getIntegerType(ogWidth), value);
      };

      // imagine a world where you are a programmer who decides the method
      // "getFPMantissaWidth" returns the width of the mantissa field + 1
      // because it accounts for the implicit bit for some godforsaken reason
      int mantissaBitwidth = fType.getFPMantissaWidth() - 1;

      IntegerType destType = b.getIntegerType(resType.getBitwidth());

      arith::BitcastOp bitcast =
          b.create<arith::BitcastOp>(b.getIntegerType(ogWidth), op.getFrom());

      int normalized_exp = resType.getExponent();

      // compute actual exponent in place
      IntegerAttr dtExp = buildIntAttr(
          b, (uint64_t)(std::abs(normalized_exp) << mantissaBitwidth));
      arith::ConstantOp dtExp_const = b.create<arith::ConstantOp>(dtExp);
      Value actual_exp =
          normalized_exp > 0
              ? b.create<arith::SubIOp>(bitcast, dtExp_const).getResult()
              : b.create<arith::AddIOp>(bitcast, dtExp_const).getResult();

      arith::BitcastOp bitcast_back =
          b.create<arith::BitcastOp>(fType, actual_exp);

      Value res =
          (resType.getSignd())
              ? b.create<arith::FPToSIOp>(destType, bitcast_back).getResult()
              : b.create<arith::FPToUIOp>(destType, bitcast_back).getResult();

      rewriter.replaceOp(op, res);
      return success();
    }
  };

  struct ConvertCastToFloat : public OpConversionPattern<CastToFloatOp> {
    ConvertCastToFloat(mlir::MLIRContext *context)
        : OpConversionPattern<CastToFloatOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(CastToFloatOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      ImplicitLocOpBuilder b(op.getLoc(), rewriter);

      FloatType fType = ::llvm::dyn_cast<FloatType>(op.getRes().getType());

      int targetWidth = fType.getWidth();

      RealType fromType =
          ::llvm::dyn_cast<RealType>(op->getOperand(0).getType());

      if (fromType.getBitwidth() > 64) {
        op->emitOpError()
            << "Conversion from fixpoints bigger than 64 is not yet supported";
        return failure();
      }

      if (targetWidth > 64) {
        op->emitOpError()
            << "Conversion to floats bigger than f64 is not yet supported";
        return failure();
      }

      if (fromType.getExponent() >
          llvm::APFloat::semanticsMaxExponent(fType.getFloatSemantics())) {
        // maybe add an option for saturating conversion in the future?
        op->emitOpError()
            << "Target float type too small to represent real value";
        return failure();
      }

      auto buildIntAttr = [targetWidth](Builder b,
                                        int64_t value) -> IntegerAttr {
        return b.getIntegerAttr(b.getIntegerType(targetWidth), value);
      };

      // imagine a world where you are a programmer who decides the method
      // "getFPMantissaWidth" returns the width of the mantissa field + 1
      // because it accounts for the implicit bit for some godforsaken reason
      int mantissaBitwidth = fType.getFPMantissaWidth() - 1;

      Value conv =
          (fromType.getSignd())
              ? b.create<arith::SIToFPOp>(fType, adaptor.getFrom()).getResult()
              : b.create<arith::UIToFPOp>(fType, adaptor.getFrom()).getResult();

      // NOTE: there is no need to account for exponent bias here, as it's
      // already implicitly accounted for

      arith::BitcastOp bitcast =
          b.create<arith::BitcastOp>(b.getIntegerType(targetWidth), conv);

      // compute actual exponent in place
      IntegerAttr dtExp = buildIntAttr(
          b, (uint64_t)(std::abs(fromType.getExponent()) << mantissaBitwidth));
      arith::ConstantOp dtExp_const = b.create<arith::ConstantOp>(dtExp);
      Value actual_exp =
          fromType.getExponent() > 0
              ? b.create<arith::AddIOp>(bitcast, dtExp_const).getResult()
              : b.create<arith::SubIOp>(bitcast, dtExp_const).getResult();

      // bitcast back
      arith::BitcastOp bitcast_back =
          b.create<arith::BitcastOp>(fType, actual_exp);

      // FIX: the "add to the exponent bits directly" trick above only
      // correctly scales NON-ZERO values. If the fixed-point integer is
      // exactly 0, its float bit pattern (from SIToFP/UIToFP) is all
      // zeros; adding a non-zero constant to THAT doesn't produce 0.0
      // scaled by 2^exponent -- it corrupts an all-zero pattern into a
      // spurious non-zero float with a real, garbage exponent field.
      // Special-case it: if the source integer is zero, produce a real
      // 0.0 directly instead of going through the bit-trick.
      Value zeroInt = b.create<arith::ConstantOp>(
          b.getIntegerAttr(b.getIntegerType(fromType.getBitwidth()), 0));
      Value isZero = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                             adaptor.getFrom(), zeroInt);
      Value zeroFloat = b.create<arith::ConstantOp>(b.getFloatAttr(fType, 0.0));
      Value res = b.create<arith::SelectOp>(isZero, zeroFloat,
                                            bitcast_back.getResult());

      rewriter.replaceOp(op, res);
      return success();
    }
  };

  // Handles the case where raise-to-taffo inserts a GENERIC
  // builtin.unrealized_conversion_cast (not a real taffo.cast2float op)
  // to bridge an already-annotated !taffo.real value into an unraised
  // consumer that still expects plain float (e.g. arith.maxnumf, which has
  // no raising rule at all). ConvertCastToFloat only matches real
  // CastToFloatOp ops, so it can never fire on these -- this pattern
  // reuses its exact numeric logic, applied to the generic op instead.
  //
  // Deliberately ONE-DIRECTIONAL (RealType -> FloatType only): the reverse
  // direction (float -> real) genuinely needs external range/precision
  // information that can't be synthesized generically, which is why those
  // cases are handled via explicit taffo.cast2real annotations elsewhere
  // in the pipeline, not by a pattern like this one.
  struct ConvertGenericCastToFloat
      : public OpConversionPattern<UnrealizedConversionCastOp> {
    ConvertGenericCastToFloat(mlir::MLIRContext *context)
        : OpConversionPattern<UnrealizedConversionCastOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      if (op->getNumOperands() != 1 || op->getNumResults() != 1)
        return failure();

      RealType fromType =
          ::llvm::dyn_cast<RealType>(op->getOperand(0).getType());
      FloatType fType = ::llvm::dyn_cast<FloatType>(op->getResult(0).getType());

      // Only claim the specific real->float bridging case; let every other
      // unrealized_conversion_cast (including the reverse direction, and
      // any unrelated type pairs) pass through untouched for other
      // patterns / reconcile-unrealized-casts to handle.
      if (!fromType || !fType)
        return failure();

      ImplicitLocOpBuilder b(op.getLoc(), rewriter);

      int targetWidth = fType.getWidth();

      if (fromType.getBitwidth() > 64) {
        op->emitOpError()
            << "Conversion from fixpoints bigger than 64 is not yet supported";
        return failure();
      }

      if (targetWidth > 64) {
        op->emitOpError()
            << "Conversion to floats bigger than f64 is not yet supported";
        return failure();
      }

      if (fromType.getExponent() >
          llvm::APFloat::semanticsMaxExponent(fType.getFloatSemantics())) {
        op->emitOpError()
            << "Target float type too small to represent real value";
        return failure();
      }

      auto buildIntAttr = [targetWidth](Builder b,
                                        int64_t value) -> IntegerAttr {
        return b.getIntegerAttr(b.getIntegerType(targetWidth), value);
      };

      int mantissaBitwidth = fType.getFPMantissaWidth() - 1;

      Value conv =
          (fromType.getSignd())
              ? b.create<arith::SIToFPOp>(fType, adaptor.getOperands()[0])
                    .getResult()
              : b.create<arith::UIToFPOp>(fType, adaptor.getOperands()[0])
                    .getResult();

      arith::BitcastOp bitcast =
          b.create<arith::BitcastOp>(b.getIntegerType(targetWidth), conv);

      IntegerAttr dtExp = buildIntAttr(
          b, (uint64_t)(std::abs(fromType.getExponent()) << mantissaBitwidth));
      arith::ConstantOp dtExp_const = b.create<arith::ConstantOp>(dtExp);
      Value actual_exp =
          fromType.getExponent() > 0
              ? b.create<arith::AddIOp>(bitcast, dtExp_const).getResult()
              : b.create<arith::SubIOp>(bitcast, dtExp_const).getResult();

      arith::BitcastOp bitcast_back =
          b.create<arith::BitcastOp>(fType, actual_exp);

      // The "add to the exponent bits directly" trick above only
      // correctly scales NON-ZERO values. If the fixed-point integer is
      // exactly 0, its float bit pattern (from SIToFP/UIToFP) is all
      // zeros; adding a non-zero constant to THAT doesn't produce 0.0
      // scaled by 2^exponent -- it corrupts an all-zero pattern into a
      // spurious non-zero float with a real, garbage exponent field.
      // Special-case it: if the source integer is zero, produce a real
      // 0.0 directly instead of going through the bit-trick.
      Value zeroInt = b.create<arith::ConstantOp>(
          b.getIntegerAttr(b.getIntegerType(fromType.getBitwidth()), 0));
      Value isZero = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                             adaptor.getOperands()[0], zeroInt);
      Value zeroFloat = b.create<arith::ConstantOp>(b.getFloatAttr(fType, 0.0));
      Value res = b.create<arith::SelectOp>(isZero, zeroFloat,
                                            bitcast_back.getResult());

      rewriter.replaceOp(op, res);
      return success();
    }
  };

  struct ConvertBitcastToReal : public OpConversionPattern<BitcastToRealOp> {
    ConvertBitcastToReal(mlir::MLIRContext *context)
        : OpConversionPattern<BitcastToRealOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(BitcastToRealOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      rewriter.replaceOp(op, adaptor.getFrom());
      return success();
    }
  };

  struct ConvertBitcastToInt : public OpConversionPattern<BitcastToIntOp> {
    ConvertBitcastToInt(mlir::MLIRContext *context)
        : OpConversionPattern<BitcastToIntOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(BitcastToIntOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      rewriter.replaceOp(op, adaptor.getFrom());
      return success();
    }
  };

  struct ConvertAlign : public OpConversionPattern<AlignOp> {
    ConvertAlign(mlir::MLIRContext *context)
        : OpConversionPattern<AlignOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(AlignOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      ImplicitLocOpBuilder b(op.getLoc(), rewriter);

      RealType source = op.getFrom().getType();
      RealType target = op.getRes().getType();

      if (source.getBitwidth() > target.getBitwidth()) {
        // shift first
        int expDiff = target.getExponent() - source.getExponent();
        arith::ConstantOp align_res = b.create<arith::ConstantOp>(
            b.getIntegerAttr(b.getIntegerType(source.getBitwidth()), expDiff));
        Value shifted =
            source.getSignd()
                ? b.create<arith::ShRSIOp>(adaptor.getFrom(), align_res)
                      .getResult()
                : b.create<arith::ShRUIOp>(adaptor.getFrom(), align_res)
                      .getResult();
        // then trunc
        Value res = b.create<arith::TruncIOp>(
            b.getIntegerType(target.getBitwidth()), shifted);
        rewriter.replaceOp(op, res);
      } else {
        Value source_val = adaptor.getFrom();
        // we need this check because ext to same datatype isn't a valid op
        // (idk why they don't simply have a folder for this)
        if (source.getBitwidth() < target.getBitwidth()) {
          // ext first
          source_val = source.getSignd()
                           ? b.create<arith::ExtSIOp>(
                                  b.getIntegerType(target.getBitwidth()),
                                  adaptor.getFrom())
                                 .getResult()
                           : b.create<arith::ExtUIOp>(
                                  b.getIntegerType(target.getBitwidth()),
                                  adaptor.getFrom())
                                 .getResult();
        }

        // then shift
        int expDiff = target.getExponent() - source.getExponent();
        arith::ConstantOp align_res =
            b.create<arith::ConstantOp>(b.getIntegerAttr(
                b.getIntegerType(target.getBitwidth()), std::abs(expDiff)));
        Value res =
            expDiff > 0
                ? source.getSignd()
                      ? b.create<arith::ShRSIOp>(source_val, align_res)
                            .getResult()
                      : b.create<arith::ShRUIOp>(source_val, align_res)
                            .getResult()
                : b.create<arith::ShLIOp>(source_val, align_res).getResult();
        rewriter.replaceOp(op, res);
      }
      return success();
    }
  };

  struct ConvertLoopRegionTypes : public OpConversionPattern<scf::ForOp> {
    ConvertLoopRegionTypes(mlir::MLIRContext *context)
        : OpConversionPattern<scf::ForOp>(context) {}

    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {

      Block *body = op.getBody();

      Region &region = op->getRegion(0);

      rewriter.startOpModification(op);
      auto terminator = cast<scf::YieldOp>(body->getTerminator());
      SmallVector<Value> terminatorRes;
      if (failed(rewriter.getRemappedValues(terminator->getOperands(),
                                            terminatorRes)))
        return failure();
      rewriter.modifyOpInPlace(terminator,
                               [&] { terminator->setOperands(terminatorRes); });

      rewriter.finalizeOpModification(op);

      if (failed(rewriter.convertRegionTypes(&region, *getTypeConverter())))
        return failure();

      ImplicitLocOpBuilder b(op.getLoc(), rewriter);

      auto newOp =
          b.create<scf::ForOp>(adaptor.getLowerBound(), adaptor.getUpperBound(),
                               adaptor.getStep(), adaptor.getInitArgs());

      // We do not need the empty block created by rewriter.
      rewriter.eraseBlock(newOp.getBody(0));
      // Inline the type converted region from the original operation.
      rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                  newOp.getRegion().end());

      rewriter.replaceOp(op, newOp);
      return success();
    }
  };

  // The matmul reduction loops don't use scf.for iter_args at all (confirmed
  // via debug instrumentation: numRegionIterArgs == 0 for every one of
  // them). They use a MEMORY-based accumulator instead:
  //
  //   %alloca = memref.alloca() : memref<f32>
  //   affine.store %cst, %alloca[] : memref<f32>        -- init, before loop
  //   affine.for %i = 0 to N {
  //     ...compute %delta (a !taffo.real value)...
  //     %acc_f32  = affine.load %alloca[] : memref<f32>
  //     %acc_real = <cast f32 to real>(%acc_f32)          -- per-iteration
  //     %new_real = taffo.add %acc_real, %delta
  //     %new_f32  = <cast real to f32>(%new_real)         -- per-iteration
  //     affine.store %new_f32, %alloca[] : memref<f32>
  //   }
  //   %final = affine.load %alloca[] : memref<f32>        -- after loop
  //
  // Same idea as before -- keep the accumulator natively typed for the
  // loop's whole duration instead of round-tripping through casts every
  // iteration -- but retargeted at this memref-based structure: we change
  // the ALLOCA's element type instead of a block argument's type.
  struct RewriteMemAccumulator : public OpRewritePattern<memref::AllocaOp> {
    const TaffoToArithTypeConverter &typeConverter;

    RewriteMemAccumulator(const TaffoToArithTypeConverter &tc,
                          MLIRContext *context)
        : OpRewritePattern<memref::AllocaOp>(context), typeConverter(tc) {}

    // Walks up from `user` looking for an affine.for/scf.for ancestor that
    // is nested INSIDE the alloca's own block (i.e. the loop containing
    // this use). Returns nullptr if `user` is directly in the alloca's own
    // block (not inside any loop relative to the alloca).
    Operation *findEnclosingLoop(Operation *user, Block *allocaBlock) const {
      Operation *cur = user;
      while (cur && cur->getBlock() != allocaBlock) {
        if (isa<scf::ForOp>(cur->getParentOp()))
          return cur->getParentOp();
        cur = cur->getParentOp();
      }
      return nullptr;
    }

    LogicalResult matchAndRewrite(memref::AllocaOp allocaOp,
                                  PatternRewriter &rewriter) const override {
      auto memrefType = dyn_cast<MemRefType>(allocaOp.getType());
      if (!memrefType || !memrefType.getElementType().isF32() ||
          memrefType.getRank() != 0) {
        return failure();
      }

      Block *allocaBlock = allocaOp->getBlock();

      memref::StoreOp initStore;
      Operation *loop = nullptr;
      memref::LoadOp bodyLoad;
      memref::StoreOp bodyStore;
      SmallVector<memref::LoadOp> postLoopLoads;

      for (Operation *user : allocaOp->getUsers()) {
        Operation *enclosingLoop = findEnclosingLoop(user, allocaBlock);

        if (!enclosingLoop) {
          // Directly in the alloca's own block: either the init store, or
          // a post-loop load.
          if (auto store = dyn_cast<memref::StoreOp>(user)) {
            if (initStore) {
              return failure();
            }
            initStore = store;
          } else if (auto load = dyn_cast<memref::LoadOp>(user)) {
            postLoopLoads.push_back(load);
          } else {
            return failure();
          }
          continue;
        }

        // Inside a loop.
        if (loop && loop != enclosingLoop) {
          return failure();
        }
        loop = enclosingLoop;

        if (auto load = dyn_cast<memref::LoadOp>(user)) {
          if (bodyLoad) {
            return failure();
          }
          bodyLoad = load;
        } else if (auto store = dyn_cast<memref::StoreOp>(user)) {
          if (bodyStore) {
            return failure();
          }
          bodyStore = store;
        } else {
          return failure();
        }
      }

      if (!initStore || !loop || !bodyLoad || !bodyStore) {
        return failure();
      }

      // Verify the body's load -> cast -> accumulate -> cast -> store chain.
      if (!bodyLoad->getResult(0).hasOneUse()) {
        return failure();
      }
      Operation *afterLoad = *bodyLoad->getResult(0).getUsers().begin();
      // Accept either a generic bridging cast OR a real, already-annotated
      // taffo.cast2real -- functionally equivalent at this point in the
      // pipeline (VRA/dt-optimization already consumed the annotation's
      // bounds; by lower-to-arith it's just a type-changing op either way).
      // Every scalar accumulator load gets a real cast2real here because
      // insert_annotations.py's --accumulator-range flag pre-annotates all
      // of them, not just some.
      Operation *castToReal = nullptr;
      if (isa<UnrealizedConversionCastOp>(afterLoad) ||
          isa<CastToRealOp>(afterLoad)) {
        if (afterLoad->getNumResults() == 1 &&
            isa<RealType>(afterLoad->getResult(0).getType()))
          castToReal = afterLoad;
      }
      if (!castToReal) {
        return failure();
      }

      if (!castToReal->getResult(0).hasOneUse()) {
        return failure();
      }
      Operation *afterCast = *castToReal->getResult(0).getUsers().begin();

      // Some accumulators (e.g. fc3) have an intermediate taffo.align
      // (exponent realignment) between the cast and the actual add --
      // recognize and skip through it, but note it for the type-equality
      // check below: align can genuinely change the exponent, and if it
      // does, the load-side and store-side would need DIFFERENT native
      // types, which this rewrite (one native type for the whole alloca)
      // cannot safely handle. Better to correctly decline than to guess.
      Operation *alignOp = nullptr;
      Operation *accumOp = afterCast;
      if (isa<AlignOp>(afterCast)) {
        alignOp = afterCast;
        if (!alignOp->getResult(0).hasOneUse()) {
          return failure();
        }
        accumOp = *alignOp->getResult(0).getUsers().begin();
      }

      if (!isa<AddOp>(accumOp)) {
        return failure();
      }

      // The native representation used throughout this rewrite is always
      // accumOp's own result type (what's actually produced and persisted
      // each iteration) -- NOT castToReal's declared type. When an align
      // op is present, its whole job is reconciling castToReal's exponent
      // with what accumOp actually needs; storing/loading natively at
      // accumOp's exponent from the start makes that reconciliation
      // unnecessary rather than skipped, so both castToReal and align can
      // be eliminated together.

      if (!accumOp->getResult(0).hasOneUse()) {
        return failure();
      }
      Operation *afterAccum = *accumOp->getResult(0).getUsers().begin();
      // Same widening for the exit side: accept either cast form.
      Operation *castToFloat = nullptr;
      if (isa<UnrealizedConversionCastOp>(afterAccum) ||
          isa<CastToFloatOp>(afterAccum)) {
        if (afterAccum->getNumResults() == 1 &&
            afterAccum->getResult(0).getType().isF32())
          castToFloat = afterAccum;
      }
      if (!castToFloat) {
        return failure();
      }

      if (bodyStore.getValueToStore() != castToFloat->getResult(0)) {
        return failure();
      }

      RealType accumType = cast<RealType>(accumOp->getResult(0).getType());
      Type nativeType = typeConverter.convertType(accumType);
      if (!nativeType)
        return failure();

      // --- All checks passed: rewrite. ---
      ImplicitLocOpBuilder b(allocaOp.getLoc(), rewriter);

      // 1. New alloca with the native element type.
      auto newMemrefType = MemRefType::get({}, nativeType);
      b.setInsertionPoint(allocaOp);
      auto newAlloca = b.create<memref::AllocaOp>(newMemrefType);

      // 2. Init store: every accumulator init value seen in this codebase
      // is the literal constant 0.0 -- special-case it directly (0 is
      // trivially representable in both float and fixed-point, no
      // scaling/rounding needed) rather than attempting a generic f32 ->
      // native conversion, which has no lowering pattern here by design
      // (that direction needs external range info we can't invent).
      // If the init value is ever NOT provably a zero constant, bail out
      // of the whole transformation rather than guess.
      b.setInsertionPoint(initStore);
      Value initF32 = initStore.getValueToStore();
      auto initConstOp = initF32.getDefiningOp<arith::ConstantOp>();
      bool initIsZero = false;
      if (initConstOp) {
        if (auto fAttr = dyn_cast<FloatAttr>(initConstOp.getValue()))
          initIsZero = fAttr.getValue().isZero();
      }
      if (!initIsZero) {
        return failure();
      }
      Value nativeZero =
          b.create<arith::ConstantOp>(rewriter.getZeroAttr(nativeType));
      b.create<memref::StoreOp>(nativeZero, newAlloca, initStore.getIndices());
      rewriter.eraseOp(initStore);

      // 3. Body: load/store natively. The loaded native value and the
      // accumulate op's real-typed result represent the SAME underlying
      // bits -- bridge between them with BitcastToRealOp/BitcastToIntOp
      // (confirmed to lower to a literal zero-cost pass-through, see
      // ConvertBitcastToReal/ConvertBitcastToInt above) instead of feeding
      // the raw native value directly into taffo.add, which requires a
      // genuinely !taffo.real-typed operand and crashes otherwise (this
      // is exactly what happened on the first attempt at this rewrite).
      b.setInsertionPoint(bodyLoad);
      auto newBodyLoad =
          b.create<memref::LoadOp>(newAlloca, bodyLoad.getIndices());
      auto reboxToReal =
          b.create<BitcastToRealOp>(accumType, newBodyLoad.getResult());
      // Redirect the correct op's uses: if align was present, its rescale
      // is what made castToReal's exponent match what accumOp needs --
      // since we now load NATIVELY at accumOp's own exponent already,
      // that rescale is no longer needed, so align's uses (not
      // castToReal's) get redirected, and both intermediate ops are
      // erased. Without align, castToReal's uses are redirected directly,
      // same as before.
      if (alignOp) {
        rewriter.replaceOp(alignOp, reboxToReal.getResult());
        rewriter.eraseOp(castToReal);
      } else {
        rewriter.replaceOp(castToReal, reboxToReal.getResult());
      }
      rewriter.eraseOp(bodyLoad);

      b.setInsertionPoint(bodyStore);
      auto reboxToInt =
          b.create<BitcastToIntOp>(nativeType, accumOp->getResult(0));
      b.create<memref::StoreOp>(reboxToInt.getResult(), newAlloca,
                                bodyStore.getIndices());
      rewriter.eraseOp(bodyStore);
      rewriter.eraseOp(castToFloat);

      // 4. Post-loop loads: load natively, bridge to !taffo.real (zero-cost
      // bitcast), then to f32 via the EXISTING generic real->float cast
      // (ConvertGenericCastToFloat), which does the genuine numeric
      // conversion. A direct native-int -> f32 cast has no lowering
      // pattern (ConvertGenericCastToFloat requires a RealType operand).
      for (auto load : postLoopLoads) {
        b.setInsertionPoint(load);
        auto newLoad = b.create<memref::LoadOp>(newAlloca, load.getIndices());
        auto reboxToReal =
            b.create<BitcastToRealOp>(accumType, newLoad.getResult());
        auto resultCast = b.create<UnrealizedConversionCastOp>(
            TypeRange{rewriter.getF32Type()},
            ValueRange{reboxToReal.getResult()});
        rewriter.replaceOp(load, resultCast.getResults());
      }

      rewriter.eraseOp(allocaOp);
      return success();
    }
  };

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::Operation *module = getOperation();

    TaffoToArithTypeConverter typeConverter(context);

    // Run this FIRST, as a separate, plain greedy rewrite -- not as part
    // of the dialect conversion below. These accumulator loops are judged
    // "already legal" by the conversion framework (f32->f32 is a trivial
    // no-op under the type converter), so a conversion pattern would never
    // even be attempted on them; a greedy rewrite has no such restriction.
    {
      RewritePatternSet accumPatterns(context);
      accumPatterns.add<RewriteMemAccumulator>(typeConverter, context);
      if (failed(applyPatternsGreedily(module, std::move(accumPatterns)))) {
        signalPassFailure();
        return;
      }
    }

    ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal([](Operation *op) { return true; });
    target.addIllegalDialect<TaffoDialect>();

    // builtin.unrealized_conversion_cast isn't part of TaffoDialect, so it
    // falls under markUnknownOpDynamicallyLegal's catch-all (legal by
    // default) -- meaning ConvertGenericCastToFloat would otherwise never
    // even be attempted by the framework, regardless of whether it's
    // correctly written. Explicitly mark it illegal ONLY for the specific
    // !taffo.real -> float pattern we want converted; every other use of
    // this op (including the type converter's own legitimate materialization
    // bridges elsewhere) remains legal and untouched.
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [](UnrealizedConversionCastOp op) {
          if (op->getNumOperands() != 1 || op->getNumResults() != 1)
            return true; // legal: not the pattern we care about
          bool isRealToFloat =
              ::llvm::isa<RealType>(op->getOperand(0).getType()) &&
              ::llvm::isa<FloatType>(op->getResult(0).getType());
          return !isRealToFloat; // illegal (needs conversion) only when true
        });

    RewritePatternSet patterns(context);

    patterns
        .add<ConvertAdd, ConvertSub, ConvertMult, ConvertDiv, ConvertCastToReal,
             ConvertCastToFloat, ConvertGenericCastToFloat, ConvertBitcastToInt,
             ConvertBitcastToReal, ConvertAlign>(typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
    target.addDynamicallyLegalOp<scf::ForOp>(
        [&](scf::ForOp op) { return typeConverter.isLegal(op); });

    RewritePatternSet loopPatterns(context);
    loopPatterns.add<ConvertLoopRegionTypes>(typeConverter, context);
    if (failed(
            applyPartialConversion(module, target, std::move(loopPatterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace mlir