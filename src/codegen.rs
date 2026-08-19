use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroU32;

use inkwell::basic_block::BasicBlock;
use inkwell::builder::BuilderError;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::{BasicType, BasicTypeEnum, StringRadix};
use inkwell::values::{
    BasicMetadataValueEnum, BasicValueEnum, FunctionValue, IntValue, PointerValue,
};
use inkwell::{FloatPredicate, IntPredicate};

use crate::ast::Program;
use crate::ast::{BinaryOp, UnaryOp};
use crate::lexer::Span;
use crate::semantic;
use crate::typed::{
    IntegerWidth, ResolvedType, TypedBlock, TypedExpr, TypedFunction, TypedProgram, TypedStmt,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodegenError {
    pub message: String,
    pub span: Option<Span>,
}

impl CodegenError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            span: None,
        }
    }
    fn at(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span: Some(span),
        }
    }
}

impl fmt::Display for CodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.span {
            Some(span) => write!(f, "{} at {}..{}", self.message, span.start, span.end),
            None => f.write_str(&self.message),
        }
    }
}
impl std::error::Error for CodegenError {}
impl From<BuilderError> for CodegenError {
    fn from(error: BuilderError) -> Self {
        Self::new(format!("LLVM builder error: {error:?}"))
    }
}

#[derive(Clone)]
struct Local<'ctx> {
    pointer: PointerValue<'ctx>,
    llvm_type: BasicTypeEnum<'ctx>,
    ty: ResolvedType,
    mutable: bool,
}

#[derive(Clone, Copy)]
struct LoopTargets<'ctx> {
    continue_block: BasicBlock<'ctx>,
    break_block: BasicBlock<'ctx>,
}

#[derive(Clone, Copy)]
struct Flow(u8);
impl Flow {
    const NORMAL: Self = Self(1);
    const RETURN: Self = Self(2);
    const BREAK: Self = Self(4);
    const CONTINUE: Self = Self(8);
    fn contains(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
    fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }
}

/// LLVM lowering consumes only resolved typed IR.  In particular, signedness,
/// widths, call targets, and local references are never inferred from syntax.
pub struct CodeGenerator<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: inkwell::builder::Builder<'ctx>,
    locals: HashMap<usize, Local<'ctx>>,
    functions: HashMap<usize, FunctionValue<'ctx>>,
    current_function: Option<FunctionValue<'ctx>>,
    current_return_type: ResolvedType,
    pointer_width: u32,
    loop_targets: Vec<LoopTargets<'ctx>>,
}

impl<'ctx> CodeGenerator<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        Self::with_pointer_width(context, module_name, usize::BITS)
    }
    pub fn with_pointer_width(
        context: &'ctx Context,
        module_name: &str,
        pointer_width: u32,
    ) -> Self {
        Self {
            context,
            module: context.create_module(module_name),
            builder: context.create_builder(),
            locals: HashMap::new(),
            functions: HashMap::new(),
            current_function: None,
            current_return_type: ResolvedType::Unit,
            pointer_width,
            loop_targets: Vec::new(),
        }
    }

    /// Compatibility entry point for users that still have an AST.  Semantic
    /// analysis and typed lowering happen before this backend is entered.
    pub fn generate(self, program: &Program) -> Result<Module<'ctx>, CodegenError> {
        let typed = semantic::analyze_typed(program)
            .map_err(|error| CodegenError::new(format!("semantic error: {error}")))?;
        self.generate_typed(&typed)
    }

    pub fn generate_typed(mut self, program: &TypedProgram) -> Result<Module<'ctx>, CodegenError> {
        self.declare_functions(program)?;
        for function in &program.functions {
            self.generate_function(function)?;
        }
        self.module
            .verify()
            .map_err(|error| CodegenError::new(error.to_string()))?;
        Ok(self.module)
    }

    fn declare_functions(&mut self, program: &TypedProgram) -> Result<(), CodegenError> {
        for function in &program.functions {
            if self.module.get_function(&function.name).is_some() {
                return Err(CodegenError::at(
                    format!("function `{}` is declared more than once", function.name),
                    function.span,
                ));
            }
            let mut params = Vec::new();
            for parameter in &function.params {
                params.push(self.basic_type(parameter.ty)?.into());
            }
            let function_type = if function.return_type == ResolvedType::Unit {
                self.context.void_type().fn_type(&params, false)
            } else {
                self.basic_type(function.return_type)?
                    .fn_type(&params, false)
            };
            let value = self
                .module
                .add_function(&function.name, function_type, None);
            self.functions.insert(function.id, value);
        }
        Ok(())
    }

    fn generate_function(&mut self, function: &TypedFunction) -> Result<(), CodegenError> {
        let llvm_function = self.functions[&function.id];
        self.current_function = Some(llvm_function);
        self.current_return_type = function.return_type;
        self.locals.clear();
        self.loop_targets.clear();
        let entry = self.context.append_basic_block(llvm_function, "entry");
        self.builder.position_at_end(entry);

        for (index, parameter) in function.params.iter().enumerate() {
            let value = llvm_function
                .get_nth_param(index as u32)
                .expect("typed parameter count matches function type");
            let llvm_type = self.basic_type(parameter.ty)?;
            let pointer = self
                .builder
                .build_alloca(llvm_type, &format!("{}.addr", parameter.name))?;
            self.builder.build_store(pointer, value)?;
            self.locals.insert(
                parameter.id,
                Local {
                    pointer,
                    llvm_type,
                    ty: parameter.ty,
                    mutable: true,
                },
            );
        }

        let flow = self.generate_block(&function.body)?;
        if flow.contains(Flow::NORMAL) {
            if self.current_return_type != ResolvedType::Unit {
                return Err(CodegenError::at(
                    format!(
                        "function `{}` does not return a value on every path",
                        function.name
                    ),
                    function.body.span,
                ));
            }
            self.builder.build_return(None)?;
        }
        if !llvm_function.verify(true) {
            return Err(CodegenError::new(format!(
                "LLVM verification failed for function `{}`",
                function.name
            )));
        }
        Ok(())
    }

    fn generate_block(&mut self, block: &TypedBlock) -> Result<Flow, CodegenError> {
        let saved = self.locals.clone();
        let result = (|| {
            let mut flow = Flow::NORMAL;
            for statement in &block.statements {
                if flow.contains(Flow::NORMAL) {
                    flow = flow
                        .without(Flow::NORMAL)
                        .union(self.generate_statement(statement)?);
                }
            }
            Ok(flow)
        })();
        self.locals = saved;
        result
    }

    fn generate_statement(&mut self, statement: &TypedStmt) -> Result<Flow, CodegenError> {
        match statement {
            TypedStmt::Declare {
                id,
                name,
                ty,
                mutable,
                value,
                ..
            } => {
                let llvm_type = self.basic_type(*ty)?;
                let value = self.generate_expression(value)?;
                if value.get_type() != llvm_type {
                    return Err(CodegenError::at(
                        "initializer has the wrong resolved type",
                        value_span(value, statement_span(statement)),
                    ));
                }
                let pointer = self
                    .builder
                    .build_alloca(llvm_type, &format!("{name}.addr"))?;
                self.builder.build_store(pointer, value)?;
                self.locals.insert(
                    *id,
                    Local {
                        pointer,
                        llvm_type,
                        ty: *ty,
                        mutable: *mutable,
                    },
                );
                Ok(Flow::NORMAL)
            }
            TypedStmt::Store {
                id,
                value,
                ty,
                span,
            } => {
                let local = self
                    .locals
                    .get(id)
                    .cloned()
                    .ok_or_else(|| CodegenError::at("unknown resolved local", *span))?;
                if !local.mutable {
                    return Err(CodegenError::at(
                        "cannot assign to immutable variable",
                        *span,
                    ));
                }
                let value = self.generate_expression(value)?;
                if value.get_type() != local.llvm_type {
                    return Err(CodegenError::at(
                        "assigned value has the wrong type",
                        value_span(value, *span),
                    ));
                }
                if *ty != local.ty {
                    return Err(CodegenError::at("resolved store type mismatch", *span));
                }
                self.builder.build_store(local.pointer, value)?;
                Ok(Flow::NORMAL)
            }
            TypedStmt::Return { value, span } => {
                match (self.current_return_type, value) {
                    (ResolvedType::Unit, None) => {
                        self.builder.build_return(None)?;
                    }
                    (ResolvedType::Unit, Some(_)) => {
                        return Err(CodegenError::at(
                            "void function cannot return a value",
                            *span,
                        ));
                    }
                    (_, None) => {
                        return Err(CodegenError::at(
                            "non-void function must return a value",
                            *span,
                        ));
                    }
                    (_, Some(expression)) => {
                        let value = self.generate_expression(expression)?;
                        self.builder.build_return(Some(&value))?;
                    }
                }
                Ok(Flow::RETURN)
            }
            TypedStmt::Expr { expression, .. } => {
                if let TypedExpr::Call { .. } = expression {
                    self.generate_call(expression)?;
                } else {
                    self.generate_expression(expression)?;
                }
                Ok(Flow::NORMAL)
            }
            TypedStmt::Break { span } => {
                let targets = self
                    .loop_targets
                    .last()
                    .copied()
                    .ok_or_else(|| CodegenError::at("break is outside a loop", *span))?;
                self.builder
                    .build_unconditional_branch(targets.break_block)?;
                Ok(Flow::BREAK)
            }
            TypedStmt::Continue { span } => {
                let targets = self
                    .loop_targets
                    .last()
                    .copied()
                    .ok_or_else(|| CodegenError::at("continue is outside a loop", *span))?;
                self.builder
                    .build_unconditional_branch(targets.continue_block)?;
                Ok(Flow::CONTINUE)
            }
            TypedStmt::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => self.generate_if(condition, then_branch, else_branch.as_ref()),
            TypedStmt::While {
                condition, body, ..
            } => self.generate_while(condition, body),
        }
    }

    fn generate_if(
        &mut self,
        condition: &TypedExpr,
        then_branch: &TypedBlock,
        else_branch: Option<&TypedBlock>,
    ) -> Result<Flow, CodegenError> {
        let condition_value = self.generate_expression(condition)?;
        let condition = self.as_condition(condition_value, condition.span())?;
        let function = self
            .current_function
            .ok_or_else(|| CodegenError::new("conditional outside function"))?;
        let then_block = self.context.append_basic_block(function, "if.then");
        let else_block = self.context.append_basic_block(function, "if.else");
        let merge = self.context.append_basic_block(function, "if.end");
        self.builder
            .build_conditional_branch(condition, then_block, else_block)?;
        self.builder.position_at_end(then_block);
        let then_flow = self.generate_block(then_branch)?;
        if then_flow.contains(Flow::NORMAL) {
            self.builder.build_unconditional_branch(merge)?;
        }
        self.builder.position_at_end(else_block);
        let else_flow = match else_branch {
            Some(block) => self.generate_block(block)?,
            None => Flow::NORMAL,
        };
        if else_flow.contains(Flow::NORMAL) {
            self.builder.build_unconditional_branch(merge)?;
        }
        let flow = then_flow.union(else_flow);
        self.builder.position_at_end(merge);
        if !flow.contains(Flow::NORMAL) {
            self.builder.build_unreachable()?;
        }
        Ok(flow)
    }

    fn generate_while(
        &mut self,
        condition: &TypedExpr,
        body: &TypedBlock,
    ) -> Result<Flow, CodegenError> {
        let function = self
            .current_function
            .ok_or_else(|| CodegenError::new("loop outside function"))?;
        let cond_block = self.context.append_basic_block(function, "while.cond");
        let body_block = self.context.append_basic_block(function, "while.body");
        let end_block = self.context.append_basic_block(function, "while.end");
        self.builder.build_unconditional_branch(cond_block)?;
        self.builder.position_at_end(cond_block);
        let condition_raw = self.generate_expression(condition)?;
        let condition_value = self.as_condition(condition_raw, condition.span())?;
        let always = matches!(condition, TypedExpr::Bool { value: true, .. });
        if always {
            self.builder.build_unconditional_branch(body_block)?;
        } else {
            self.builder
                .build_conditional_branch(condition_value, body_block, end_block)?;
        }
        self.loop_targets.push(LoopTargets {
            continue_block: cond_block,
            break_block: end_block,
        });
        self.builder.position_at_end(body_block);
        let body_flow = self.generate_block(body)?;
        self.loop_targets.pop();
        if body_flow.contains(Flow::NORMAL) {
            self.builder.build_unconditional_branch(cond_block)?;
        }
        self.builder.position_at_end(end_block);
        let has_exit = !always || body_flow.contains(Flow::BREAK);
        if always && !has_exit {
            self.builder.build_unreachable()?;
        }
        let mut flow = if body_flow.contains(Flow::RETURN) {
            Flow::RETURN
        } else {
            Flow(0)
        };
        if has_exit {
            flow = flow.union(Flow::NORMAL);
        }
        Ok(flow)
    }

    fn generate_expression(
        &mut self,
        expression: &TypedExpr,
    ) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        match expression {
            TypedExpr::Integer { value, ty, span } => {
                let BasicTypeEnum::IntType(integer_type) = self.basic_type(*ty)? else {
                    return Err(CodegenError::at(
                        "integer has a non-integer resolved type",
                        *span,
                    ));
                };
                let value = integer_type
                    .const_int_from_string(&value.to_string(), StringRadix::Decimal)
                    .ok_or_else(|| CodegenError::at("integer literal is out of range", *span))?;
                Ok(value.into())
            }
            TypedExpr::Float { value, ty, span } => {
                let BasicTypeEnum::FloatType(float_type) = self.basic_type(*ty)? else {
                    return Err(CodegenError::at(
                        "float has a non-float resolved type",
                        *span,
                    ));
                };
                Ok(float_type.const_float(*value).into())
            }
            TypedExpr::Bool { value, .. } => Ok(self
                .context
                .bool_type()
                .const_int(u64::from(*value), false)
                .into()),
            TypedExpr::Load { id, span, .. } => {
                let local = self
                    .locals
                    .get(id)
                    .cloned()
                    .ok_or_else(|| CodegenError::at("unknown resolved local", *span))?;
                Ok(self
                    .builder
                    .build_load(local.llvm_type, local.pointer, "loadtmp")?)
            }
            TypedExpr::Unary {
                operator,
                operand,
                ty,
                span,
            } => {
                let value = self.generate_expression(operand)?;
                match (operator, value) {
                    (UnaryOp::Negate, BasicValueEnum::IntValue(value)) => {
                        Ok(self.builder.build_int_neg(value, "negtmp")?.into())
                    }
                    (UnaryOp::Negate, BasicValueEnum::FloatValue(value)) => {
                        Ok(self.builder.build_float_neg(value, "negtmp")?.into())
                    }
                    (UnaryOp::Not, BasicValueEnum::IntValue(value))
                    | (UnaryOp::BitwiseNot, BasicValueEnum::IntValue(value)) => {
                        Ok(self.builder.build_not(value, "nottmp")?.into())
                    }
                    _ => Err(CodegenError::at(
                        format!("unsupported unary operand of type {ty:?}"),
                        *span,
                    )),
                }
            }
            TypedExpr::Binary {
                left,
                operator,
                right,
                operand_type,
                span,
                ..
            } => self.generate_binary(left, *operator, right, *operand_type, *span),
            TypedExpr::Call { .. } => self.generate_call(expression)?.ok_or_else(|| {
                CodegenError::at("void function call has no value", expression.span())
            }),
        }
    }

    fn generate_binary(
        &mut self,
        left: &TypedExpr,
        operator: BinaryOp,
        right: &TypedExpr,
        operand_type: ResolvedType,
        span: Span,
    ) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        if matches!(operator, BinaryOp::LogicalAnd | BinaryOp::LogicalOr) {
            return self.generate_logical(left, operator, right, span);
        }
        let left = self.generate_expression(left)?;
        let right = self.generate_expression(right)?;
        match (left, right) {
            (BasicValueEnum::IntValue(left), BasicValueEnum::IntValue(right)) => {
                self.generate_integer(left, operator, right, operand_type, span)
            }
            (BasicValueEnum::FloatValue(left), BasicValueEnum::FloatValue(right)) => {
                self.generate_float(left, operator, right, span)
            }
            _ => Err(CodegenError::at(
                "binary operands have incompatible resolved types",
                span,
            )),
        }
    }

    fn generate_logical(
        &mut self,
        left: &TypedExpr,
        operator: BinaryOp,
        right: &TypedExpr,
        span: Span,
    ) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        let left_raw = self.generate_expression(left)?;
        let left = self.as_condition(left_raw, left.span())?;
        let function = self
            .current_function
            .ok_or_else(|| CodegenError::at("logical expression outside function", span))?;
        let right_block = self.context.append_basic_block(function, "logical.right");
        let short_block = self.context.append_basic_block(function, "logical.short");
        let merge = self.context.append_basic_block(function, "logical.end");
        if matches!(operator, BinaryOp::LogicalAnd) {
            self.builder
                .build_conditional_branch(left, right_block, short_block)?;
        } else {
            self.builder
                .build_conditional_branch(left, short_block, right_block)?;
        }
        self.builder.position_at_end(right_block);
        let right_raw = self.generate_expression(right)?;
        let right = self.as_condition(right_raw, right.span())?;
        self.builder.build_unconditional_branch(merge)?;
        self.builder.position_at_end(short_block);
        let short = self
            .context
            .bool_type()
            .const_int(u64::from(matches!(operator, BinaryOp::LogicalOr)), false);
        self.builder.build_unconditional_branch(merge)?;
        self.builder.position_at_end(merge);
        let phi = self
            .builder
            .build_phi(self.context.bool_type(), "logicaltmp")?;
        phi.add_incoming(&[(&right, right_block), (&short, short_block)]);
        Ok(phi.as_basic_value())
    }

    fn generate_integer(
        &mut self,
        left: IntValue<'ctx>,
        operator: BinaryOp,
        right: IntValue<'ctx>,
        ty: ResolvedType,
        span: Span,
    ) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        if ty == ResolvedType::Bool {
            return match operator {
                BinaryOp::Equal => Ok(self
                    .builder
                    .build_int_compare(IntPredicate::EQ, left, right, "eqtmp")?
                    .into()),
                BinaryOp::NotEqual => Ok(self
                    .builder
                    .build_int_compare(IntPredicate::NE, left, right, "netmp")?
                    .into()),
                _ => Err(CodegenError::at("invalid boolean operator", span)),
            };
        }
        let ResolvedType::Integer { width, signed } = ty else {
            return Err(CodegenError::at(
                "integer operation has non-integer resolved type",
                span,
            ));
        };
        let unsigned = !signed;
        let result = match operator {
            BinaryOp::Add => self.builder.build_int_add(left, right, "addtmp")?.into(),
            BinaryOp::Subtract => self.builder.build_int_sub(left, right, "subtmp")?.into(),
            BinaryOp::Multiply => self.builder.build_int_mul(left, right, "multmp")?.into(),
            BinaryOp::Divide | BinaryOp::Modulo => {
                self.guard_integer_division(left, right, signed, span)?;
                if operator == BinaryOp::Divide {
                    if unsigned {
                        self.builder
                            .build_int_unsigned_div(left, right, "divtmp")?
                            .into()
                    } else {
                        self.builder
                            .build_int_signed_div(left, right, "divtmp")?
                            .into()
                    }
                } else if unsigned {
                    self.builder
                        .build_int_unsigned_rem(left, right, "modtmp")?
                        .into()
                } else {
                    self.builder
                        .build_int_signed_rem(left, right, "modtmp")?
                        .into()
                }
            }
            BinaryOp::Equal => self
                .builder
                .build_int_compare(IntPredicate::EQ, left, right, "eqtmp")?
                .into(),
            BinaryOp::NotEqual => self
                .builder
                .build_int_compare(IntPredicate::NE, left, right, "netmp")?
                .into(),
            BinaryOp::Less => self
                .builder
                .build_int_compare(
                    if unsigned {
                        IntPredicate::ULT
                    } else {
                        IntPredicate::SLT
                    },
                    left,
                    right,
                    "lttmp",
                )?
                .into(),
            BinaryOp::LessEqual => self
                .builder
                .build_int_compare(
                    if unsigned {
                        IntPredicate::ULE
                    } else {
                        IntPredicate::SLE
                    },
                    left,
                    right,
                    "letmp",
                )?
                .into(),
            BinaryOp::Greater => self
                .builder
                .build_int_compare(
                    if unsigned {
                        IntPredicate::UGT
                    } else {
                        IntPredicate::SGT
                    },
                    left,
                    right,
                    "gttmp",
                )?
                .into(),
            BinaryOp::GreaterEqual => self
                .builder
                .build_int_compare(
                    if unsigned {
                        IntPredicate::UGE
                    } else {
                        IntPredicate::SGE
                    },
                    left,
                    right,
                    "getmp",
                )?
                .into(),
            BinaryOp::BitwiseAnd => self.builder.build_and(left, right, "andtmp")?.into(),
            BinaryOp::BitwiseOr => self.builder.build_or(left, right, "ortmp")?.into(),
            BinaryOp::BitwiseXor => self.builder.build_xor(left, right, "xortmp")?.into(),
            BinaryOp::ShiftLeft | BinaryOp::ShiftRight => {
                let bits = match width {
                    IntegerWidth::Bits(bits) => bits as u64,
                    IntegerWidth::Pointer => self.pointer_width as u64,
                };
                let limit = left.get_type().const_int(bits, false);
                let valid = self.builder.build_int_compare(
                    IntPredicate::ULT,
                    right,
                    limit,
                    "shift.valid",
                )?;
                self.guard(valid, span)?;
                if operator == BinaryOp::ShiftLeft {
                    self.builder.build_left_shift(left, right, "shltmp")?.into()
                } else {
                    self.builder
                        .build_right_shift(left, right, signed, "shrtmp")?
                        .into()
                }
            }
            BinaryOp::LogicalAnd | BinaryOp::LogicalOr => {
                return Err(CodegenError::at(
                    "logical operators require boolean operands",
                    span,
                ));
            }
        };
        Ok(result)
    }

    fn guard_integer_division(
        &mut self,
        left: IntValue<'ctx>,
        right: IntValue<'ctx>,
        signed: bool,
        span: Span,
    ) -> Result<(), CodegenError> {
        let zero = right.get_type().const_zero();
        let nonzero =
            self.builder
                .build_int_compare(IntPredicate::NE, right, zero, "div.nonzero")?;
        self.guard(nonzero, span)?;
        if signed {
            let bits = left.get_type().get_bit_width();
            let one = left.get_type().const_int(1, false);
            let shift = left
                .get_type()
                .const_int(u64::from(bits.saturating_sub(1)), false);
            let min = self.builder.build_left_shift(one, shift, "div.min.value")?;
            let minus_one = left.get_type().const_all_ones();
            let is_min = self
                .builder
                .build_int_compare(IntPredicate::EQ, left, min, "div.min")?;
            let is_minus_one = self.builder.build_int_compare(
                IntPredicate::EQ,
                right,
                minus_one,
                "div.minusone",
            )?;
            let overflow = self
                .builder
                .build_and(is_min, is_minus_one, "div.overflow")?;
            let safe = self.builder.build_not(overflow, "div.safe")?;
            self.guard(safe, span)?;
        }
        Ok(())
    }

    fn guard(&mut self, condition: IntValue<'ctx>, span: Span) -> Result<(), CodegenError> {
        let function = self
            .current_function
            .ok_or_else(|| CodegenError::at("trap outside function", span))?;
        let trap = self.context.append_basic_block(function, "trap");
        let continue_block = self.context.append_basic_block(function, "after.trap");
        self.builder
            .build_conditional_branch(condition, continue_block, trap)?;
        self.builder.position_at_end(trap);
        let trap_fn = self.module.get_function("llvm.trap").unwrap_or_else(|| {
            self.module.add_function(
                "llvm.trap",
                self.context.void_type().fn_type(&[], false),
                None,
            )
        });
        self.builder.build_call(trap_fn, &[], "trap")?;
        self.builder.build_unreachable()?;
        self.builder.position_at_end(continue_block);
        Ok(())
    }

    fn generate_float(
        &self,
        left: inkwell::values::FloatValue<'ctx>,
        operator: BinaryOp,
        right: inkwell::values::FloatValue<'ctx>,
        span: Span,
    ) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        Ok(match operator {
            BinaryOp::Add => self.builder.build_float_add(left, right, "addtmp")?.into(),
            BinaryOp::Subtract => self.builder.build_float_sub(left, right, "subtmp")?.into(),
            BinaryOp::Multiply => self.builder.build_float_mul(left, right, "multmp")?.into(),
            BinaryOp::Divide => self.builder.build_float_div(left, right, "divtmp")?.into(),
            BinaryOp::Modulo => self.builder.build_float_rem(left, right, "modtmp")?.into(),
            BinaryOp::Equal => self
                .builder
                .build_float_compare(FloatPredicate::OEQ, left, right, "eqtmp")?
                .into(),
            BinaryOp::NotEqual => self
                .builder
                .build_float_compare(FloatPredicate::ONE, left, right, "netmp")?
                .into(),
            BinaryOp::Less => self
                .builder
                .build_float_compare(FloatPredicate::OLT, left, right, "lttmp")?
                .into(),
            BinaryOp::LessEqual => self
                .builder
                .build_float_compare(FloatPredicate::OLE, left, right, "letmp")?
                .into(),
            BinaryOp::Greater => self
                .builder
                .build_float_compare(FloatPredicate::OGT, left, right, "gttmp")?
                .into(),
            BinaryOp::GreaterEqual => self
                .builder
                .build_float_compare(FloatPredicate::OGE, left, right, "getmp")?
                .into(),
            _ => {
                return Err(CodegenError::at(
                    "operator is not valid for floating point values",
                    span,
                ));
            }
        })
    }

    fn generate_call(
        &mut self,
        expression: &TypedExpr,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let TypedExpr::Call {
            function: id,
            arguments,
            span,
            ..
        } = expression
        else {
            return Err(CodegenError::new("internal error: expected call"));
        };
        let function = self
            .functions
            .get(id)
            .copied()
            .ok_or_else(|| CodegenError::at("unknown resolved call target", *span))?;
        let mut values = Vec::new();
        for argument in arguments {
            values.push(BasicMetadataValueEnum::from(
                self.generate_expression(argument)?,
            ));
        }
        let call = self.builder.build_call(function, &values, "calltmp")?;
        Ok(call.try_as_basic_value().basic())
    }

    fn as_condition(
        &self,
        value: BasicValueEnum<'ctx>,
        span: Span,
    ) -> Result<IntValue<'ctx>, CodegenError> {
        match value {
            BasicValueEnum::IntValue(value) if value.get_type().get_bit_width() == 1 => Ok(value),
            _ => Err(CodegenError::at("condition must be a bool", span)),
        }
    }

    fn basic_type(&self, ty: ResolvedType) -> Result<BasicTypeEnum<'ctx>, CodegenError> {
        Ok(match ty {
            ResolvedType::Bool => self.context.bool_type().into(),
            ResolvedType::Integer { width, .. } => match width {
                IntegerWidth::Bits(8) => self.context.i8_type().into(),
                IntegerWidth::Bits(16) => self.context.i16_type().into(),
                IntegerWidth::Bits(32) => self.context.i32_type().into(),
                IntegerWidth::Bits(64) => self.context.i64_type().into(),
                IntegerWidth::Bits(128) => self.context.i128_type().into(),
                IntegerWidth::Bits(bits) => {
                    return Err(CodegenError::new(format!(
                        "unsupported integer width {bits}"
                    )));
                }
                IntegerWidth::Pointer => self
                    .context
                    .custom_width_int_type(
                        NonZeroU32::new(self.pointer_width)
                            .ok_or_else(|| CodegenError::new("target pointer width is zero"))?,
                    )
                    .map_err(CodegenError::new)?
                    .into(),
            },
            ResolvedType::Float { bits: 32 } => self.context.f32_type().into(),
            ResolvedType::Float { bits: 64 } => self.context.f64_type().into(),
            ResolvedType::Float { bits } => {
                return Err(CodegenError::new(format!("unsupported float width {bits}")));
            }
            ResolvedType::Unit => return Err(CodegenError::new("unit is not a value type")),
        })
    }
}

fn statement_span(statement: &TypedStmt) -> Span {
    match statement {
        TypedStmt::Declare { span, .. }
        | TypedStmt::Store { span, .. }
        | TypedStmt::Return { span, .. }
        | TypedStmt::Expr { span, .. }
        | TypedStmt::If { span, .. }
        | TypedStmt::While { span, .. }
        | TypedStmt::Break { span }
        | TypedStmt::Continue { span } => *span,
    }
}

fn value_span(_value: BasicValueEnum<'_>, span: Span) -> Span {
    span
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::analyze_source;

    fn generate(source: &str) -> String {
        let typed = analyze_source(source).expect("analysis failed");
        let context = Context::create();
        CodeGenerator::new(&context, "test")
            .generate_typed(&typed)
            .unwrap()
            .print_to_string()
            .to_string()
    }

    #[test]
    fn generates_minimal_main() {
        let ir = generate("main :: () -> i32 { return 42; }");
        assert!(ir.contains("define i32 @main()"));
        assert!(ir.contains("ret i32 42"));
    }

    #[test]
    fn uses_resolved_unsigned_operations() {
        let ir = generate(
            "less :: (a: u32, b: u32) -> bool { return a < b; } divide :: (a: u32, b: u32) -> u32 { return a / b; }",
        );
        assert!(ir.contains("icmp ult i32"));
        assert!(ir.contains("udiv i32"));
        assert!(ir.contains("llvm.trap"));
    }

    #[test]
    fn emits_traps_for_invalid_division_and_shifts() {
        let ir = generate("main :: () -> i32 { x := 0; return 1 / x; }");
        assert!(ir.contains("div.nonzero"));
        assert!(ir.contains("llvm.trap"));

        let ir = generate("main :: () -> i32 { x := 1; y := 32; return x << y; }");
        assert!(ir.contains("icmp ult i32"));
        assert!(ir.contains("llvm.trap"));
    }

    #[test]
    fn wrapping_arithmetic_has_no_overflow_flags() {
        let ir = generate("main :: () -> i32 { x := 2147483647; return x + 1; }");
        assert!(ir.contains("add i32") && !ir.contains("add nsw") && !ir.contains("add nuw"));
    }
}
