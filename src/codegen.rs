use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroU32;

use inkwell::FloatPredicate;
use inkwell::IntPredicate;
use inkwell::builder::BuilderError;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, FunctionType, StringRadix};
use inkwell::values::{
    BasicMetadataValueEnum, BasicValueEnum, FunctionValue, IntValue, PointerValue,
};

use crate::ast::{BinaryOp, Block, Decl, Expr, FunctionDecl, Program, Stmt, Type, UnaryOp};
use crate::lexer::Span;

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
            None => write!(f, "{}", self.message),
        }
    }
}

impl std::error::Error for CodegenError {}

impl From<BuilderError> for CodegenError {
    fn from(error: BuilderError) -> Self {
        builder_error(error)
    }
}

#[derive(Clone)]
struct Local<'ctx> {
    pointer: PointerValue<'ctx>,
    ty: BasicTypeEnum<'ctx>,
    source_ty: Type,
    mutable: bool,
}

/// Generates LLVM IR directly from the AST
pub struct CodeGenerator<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: inkwell::builder::Builder<'ctx>,
    locals: HashMap<String, Local<'ctx>>,
    function_return_types: HashMap<String, Type>,
    current_return_type: Option<BasicTypeEnum<'ctx>>,
    pointer_width: u32,
}

impl<'ctx> CodeGenerator<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        Self::with_pointer_width(context, module_name, usize::BITS)
    }

    /// Create a generator for a target with the supplied pointer width.
    /// `new` uses the host width, which is suitable for the IR-only command;
    /// native builds pass the LLVM target's actual pointer width.
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
            function_return_types: HashMap::new(),
            current_return_type: None,
            pointer_width,
        }
    }

    pub fn generate(mut self, program: &Program) -> Result<Module<'ctx>, CodegenError> {
        self.declare_functions(program)?;

        for declaration in &program.declarations {
            match declaration {
                Decl::Function(function) => self.generate_function(function)?,
                Decl::Variable(variable) => {
                    return Err(CodegenError::at(
                        "top-level variables are not supported by the LLVM backend yet",
                        variable.span,
                    ));
                }
            }
        }

        self.module
            .verify()
            .map_err(|error| CodegenError::new(error.to_string()))?;

        Ok(self.module)
    }

    fn declare_functions(&mut self, program: &Program) -> Result<(), CodegenError> {
        for declaration in &program.declarations {
            let Decl::Function(function) = declaration else {
                continue;
            };

            if self.module.get_function(&function.name).is_some() {
                return Err(CodegenError::at(
                    format!("function `{}` is declared more than once", function.name),
                    function.span,
                ));
            }

            let function_type = self.function_type(function)?;
            self.function_return_types
                .insert(function.name.clone(), function.return_type.clone());
            self.module
                .add_function(&function.name, function_type, None);
        }

        Ok(())
    }

    fn function_type(&self, function: &FunctionDecl) -> Result<FunctionType<'ctx>, CodegenError> {
        let mut parameters = Vec::with_capacity(function.params.len());

        for parameter in &function.params {
            parameters.push(
                self.basic_type(&parameter.ty)
                    .map(BasicMetadataTypeEnum::from)?,
            );
        }

        if is_void(&function.return_type) {
            Ok(self.context.void_type().fn_type(&parameters, false))
        } else {
            Ok(self
                .basic_type(&function.return_type)?
                .fn_type(&parameters, false))
        }
    }

    fn generate_function(&mut self, function: &FunctionDecl) -> Result<(), CodegenError> {
        let llvm_function = self
            .module
            .get_function(&function.name)
            .expect("all functions are declared before bodies are generated");

        let entry = self.context.append_basic_block(llvm_function, "entry");
        self.builder.position_at_end(entry);
        self.locals.clear();
        self.current_return_type = if is_void(&function.return_type) {
            None
        } else {
            Some(self.basic_type(&function.return_type)?)
        };

        for (index, parameter) in function.params.iter().enumerate() {
            let parameter_value = llvm_function
                .get_nth_param(index as u32)
                .expect("function parameter count matches its declaration");
            let parameter_type = self.basic_type(&parameter.ty)?;
            let pointer = self
                .builder
                .build_alloca(parameter_type, &format!("{}.addr", parameter.name))
                .map_err(builder_error)?;
            self.builder
                .build_store(pointer, parameter_value)
                .map_err(builder_error)?;

            self.locals.insert(
                parameter.name.clone(),
                Local {
                    pointer,
                    ty: parameter_type,
                    source_ty: parameter.ty.clone(),
                    mutable: true,
                },
            );
        }

        let terminated = self.generate_block(&function.body)?;

        if !terminated {
            if self.current_return_type.is_some() {
                return Err(CodegenError::at(
                    format!(
                        "function `{}` does not return a value on every path",
                        function.name
                    ),
                    function.body.span,
                ));
            }

            self.builder.build_return(None).map_err(builder_error)?;
        }

        if !llvm_function.verify(true) {
            return Err(CodegenError::new(format!(
                "LLVM verification failed for function `{}`",
                function.name
            )));
        }

        Ok(())
    }

    /// Returns true when every path through the block has terminated.
    fn generate_block(&mut self, block: &Block) -> Result<bool, CodegenError> {
        // Keep lexical bindings local to this block. Stores through a binding
        // from an outer scope still affect the same alloca, while a shadowing
        // declaration disappears when the block ends.
        let saved_locals = self.locals.clone();
        let result = (|| {
            for statement in &block.statements {
                if self.generate_statement(statement)? {
                    return Ok(true);
                }
            }

            Ok(false)
        })();
        self.locals = saved_locals;
        result
    }

    /// Returns true for a statement that terminates its current basic block.
    fn generate_statement(&mut self, statement: &Stmt) -> Result<bool, CodegenError> {
        match statement {
            Stmt::Return { value, span } => {
                match (self.current_return_type, value) {
                    (None, None) => {
                        self.builder.build_return(None).map_err(builder_error)?;
                    }

                    (None, Some(_)) => {
                        return Err(CodegenError::at(
                            "void function cannot return a value",
                            *span,
                        ));
                    }

                    (Some(_), None) => {
                        return Err(CodegenError::at(
                            "non-void function must return a value",
                            *span,
                        ));
                    }

                    (Some(return_type), Some(expression)) => {
                        let value = self.generate_expression(expression, Some(return_type))?;
                        self.builder
                            .build_return(Some(&value))
                            .map_err(builder_error)?;
                    }
                }

                Ok(true)
            }

            Stmt::Variable(variable) => {
                let expected_type = variable
                    .ty
                    .as_ref()
                    .map(|ty| self.basic_type(ty))
                    .transpose()?;
                let value = self.generate_expression(&variable.value, expected_type)?;
                let value_type = value.get_type();

                if let Some(expected_type) = expected_type
                    && value_type != expected_type
                {
                    return Err(CodegenError::at(
                        format!(
                            "initializer for `{}` has type {}, expected {}",
                            variable.name, value_type, expected_type
                        ),
                        variable.value.span(),
                    ));
                }

                let pointer = self
                    .builder
                    .build_alloca(value_type, &format!("{}.addr", variable.name))
                    .map_err(builder_error)?;
                self.builder
                    .build_store(pointer, value)
                    .map_err(builder_error)?;

                self.locals.insert(
                    variable.name.clone(),
                    Local {
                        pointer,
                        ty: value_type,
                        source_ty: variable
                            .ty
                            .clone()
                            .or_else(|| self.source_type_of_expression(&variable.value))
                            .unwrap_or_else(|| Type::Named("i32".to_string())),
                        mutable: !matches!(variable.kind, crate::ast::VariableKind::Immutable),
                    },
                );

                Ok(false)
            }

            Stmt::Assignment {
                target,
                value,
                span,
            } => {
                let Expr::Identifier { name, .. } = target else {
                    return Err(CodegenError::at(
                        "assignment target must be an identifier",
                        *span,
                    ));
                };

                let local = self.locals.get(name).cloned().ok_or_else(|| {
                    CodegenError::at(format!("unknown variable `{name}`"), target.span())
                })?;

                if !local.mutable {
                    return Err(CodegenError::at(
                        format!("cannot assign to immutable variable `{name}`"),
                        target.span(),
                    ));
                }

                let new_value = self.generate_expression(value, Some(local.ty))?;
                if new_value.get_type() != local.ty {
                    return Err(CodegenError::at(
                        format!(
                            "assigned value has type {}, expected {}",
                            new_value.get_type(),
                            local.ty
                        ),
                        value.span(),
                    ));
                }

                self.builder
                    .build_store(local.pointer, new_value)
                    .map_err(builder_error)?;

                Ok(false)
            }

            Stmt::Expr { expression, .. } => {
                if let Expr::Call { .. } = expression {
                    self.generate_call(expression)?;
                } else {
                    self.generate_expression(expression, None)?;
                }

                Ok(false)
            }

            Stmt::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => self.generate_if(condition, then_branch, else_branch.as_ref()),
        }
    }

    fn generate_if(
        &mut self,
        condition: &Expr,
        then_branch: &Block,
        else_branch: Option<&Block>,
    ) -> Result<bool, CodegenError> {
        let condition_span = condition.span();
        let condition = self.generate_expression(condition, None)?;
        let condition = self.as_condition(condition, condition_span)?;
        let function = self
            .current_function()
            .ok_or_else(|| CodegenError::new("conditional generated outside a function"))?;

        let then_block = self.context.append_basic_block(function, "if.then");
        let else_block = self.context.append_basic_block(function, "if.else");
        let merge_block = self.context.append_basic_block(function, "if.end");

        self.builder
            .build_conditional_branch(condition, then_block, else_block)
            .map_err(builder_error)?;

        self.builder.position_at_end(then_block);
        let then_terminated = self.generate_block(then_branch)?;
        if !then_terminated {
            self.builder
                .build_unconditional_branch(merge_block)
                .map_err(builder_error)?;
        }

        self.builder.position_at_end(else_block);
        let else_terminated = match else_branch {
            Some(block) => self.generate_block(block)?,
            None => false,
        };
        if !else_terminated {
            self.builder
                .build_unconditional_branch(merge_block)
                .map_err(builder_error)?;
        }

        let all_paths_terminate = then_terminated && else_branch.is_some() && else_terminated;
        self.builder.position_at_end(merge_block);

        if all_paths_terminate {
            self.builder.build_unreachable().map_err(builder_error)?;
        }

        Ok(all_paths_terminate)
    }

    fn generate_expression(
        &mut self,
        expression: &Expr,
        expected_type: Option<BasicTypeEnum<'ctx>>,
    ) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        match expression {
            Expr::Integer { value, span } => {
                let integer_type = match expected_type {
                    Some(BasicTypeEnum::IntType(integer_type)) => integer_type,
                    Some(other) => {
                        return Err(CodegenError::at(
                            format!("integer literal cannot have type {other}"),
                            *span,
                        ));
                    }
                    None => self.context.i32_type(),
                };

                let value = integer_type
                    .const_int_from_string(&value.to_string(), StringRadix::Decimal)
                    .ok_or_else(|| CodegenError::at("integer literal is out of range", *span))?;

                Ok(value.into())
            }

            Expr::Float { value, span } => {
                let float_type = match expected_type {
                    Some(BasicTypeEnum::FloatType(float_type)) => float_type,
                    Some(other) => {
                        return Err(CodegenError::at(
                            format!("floating-point literal cannot have type {other}"),
                            *span,
                        ));
                    }
                    None => self.context.f64_type(),
                };

                Ok(float_type.const_float(*value).into())
            }

            Expr::Bool { value, span } => {
                if let Some(expected) = expected_type
                    && expected != self.context.bool_type().into()
                {
                    return Err(CodegenError::at(
                        format!("boolean literal cannot have type {expected}"),
                        *span,
                    ));
                }
                Ok(self
                    .context
                    .bool_type()
                    .const_int(u64::from(*value), false)
                    .into())
            }

            Expr::Identifier { name, span } => {
                let local =
                    self.locals.get(name).cloned().ok_or_else(|| {
                        CodegenError::at(format!("unknown variable `{name}`"), *span)
                    })?;

                let value = self
                    .builder
                    .build_load(local.ty, local.pointer, name)
                    .map_err(builder_error)?;

                if let Some(expected_type) = expected_type
                    && value.get_type() != expected_type
                {
                    return Err(CodegenError::at(
                        format!(
                            "variable `{name}` has type {}, expected {expected_type}",
                            value.get_type()
                        ),
                        *span,
                    ));
                }

                Ok(value)
            }

            Expr::Unary {
                operator,
                operand,
                span,
            } => {
                let value = self.generate_expression(operand, expected_type)?;

                match (operator, value) {
                    (UnaryOp::Negate, BasicValueEnum::IntValue(value)) => Ok(self
                        .builder
                        .build_int_neg(value, "negtmp")
                        .map_err(builder_error)?
                        .into()),
                    (UnaryOp::Negate, BasicValueEnum::FloatValue(value)) => Ok(self
                        .builder
                        .build_float_neg(value, "negtmp")
                        .map_err(builder_error)?
                        .into()),
                    (UnaryOp::Not, BasicValueEnum::IntValue(value)) => Ok(self
                        .builder
                        .build_not(value, "nottmp")
                        .map_err(builder_error)?
                        .into()),
                    (UnaryOp::BitwiseNot, BasicValueEnum::IntValue(value)) => Ok(self
                        .builder
                        .build_not(value, "bwnottmp")
                        .map_err(builder_error)?
                        .into()),
                    _ => Err(CodegenError::at(
                        "unsupported operand type for unary operator",
                        *span,
                    )),
                }
            }

            Expr::Binary {
                left,
                operator,
                right,
                span,
            } => self.generate_binary(left, *operator, right, *span, expected_type),

            Expr::Call { .. } => self.generate_call(expression)?.ok_or_else(|| {
                CodegenError::at("void function call has no value", expression.span())
            }),
        }
    }

    fn generate_binary(
        &mut self,
        left: &Expr,
        operator: BinaryOp,
        right: &Expr,
        span: Span,
        expected_type: Option<BasicTypeEnum<'ctx>>,
    ) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        if matches!(operator, BinaryOp::LogicalAnd | BinaryOp::LogicalOr) {
            return self.generate_logical_binary(left, operator, right, span);
        }

        let operand_type = match operator {
            BinaryOp::LogicalAnd | BinaryOp::LogicalOr => Some(self.context.bool_type().into()),
            BinaryOp::Equal
            | BinaryOp::NotEqual
            | BinaryOp::Less
            | BinaryOp::LessEqual
            | BinaryOp::Greater
            | BinaryOp::GreaterEqual => None,
            _ => expected_type,
        };

        let unsigned = self.expression_is_unsigned(left) || self.expression_is_unsigned(right);
        let left = self.generate_expression(left, operand_type)?;
        let right = self.generate_expression(right, Some(left.get_type()))?;

        match (left, right) {
            (BasicValueEnum::IntValue(left), BasicValueEnum::IntValue(right)) => {
                self.generate_integer_binary(left, operator, right, span, unsigned)
            }
            (BasicValueEnum::FloatValue(left), BasicValueEnum::FloatValue(right)) => {
                self.generate_float_binary(left, operator, right, span)
            }
            _ => Err(CodegenError::at(
                "binary operands must have the same numeric type",
                span,
            )),
        }
    }

    fn generate_logical_binary(
        &mut self,
        left: &Expr,
        operator: BinaryOp,
        right: &Expr,
        span: Span,
    ) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        let bool_type = self.context.bool_type();
        let left_value = self.generate_expression(left, Some(bool_type.into()))?;
        let left_condition = self.as_condition(left_value, left.span())?;
        let function = self.current_function().ok_or_else(|| {
            CodegenError::at("logical expression generated outside a function", span)
        })?;

        let right_block = self.context.append_basic_block(function, "logical.right");
        let short_block = self.context.append_basic_block(function, "logical.short");
        let merge_block = self.context.append_basic_block(function, "logical.end");
        let is_and = matches!(operator, BinaryOp::LogicalAnd);

        if is_and {
            self.builder
                .build_conditional_branch(left_condition, right_block, short_block)
                .map_err(builder_error)?;
        } else {
            self.builder
                .build_conditional_branch(left_condition, short_block, right_block)
                .map_err(builder_error)?;
        }

        self.builder.position_at_end(right_block);
        let right_value = self.generate_expression(right, Some(bool_type.into()))?;
        let right_condition = self.as_condition(right_value, right.span())?;
        self.builder
            .build_unconditional_branch(merge_block)
            .map_err(builder_error)?;

        self.builder.position_at_end(short_block);
        let short_value = bool_type.const_int(u64::from(!is_and), false);
        self.builder
            .build_unconditional_branch(merge_block)
            .map_err(builder_error)?;

        self.builder.position_at_end(merge_block);
        let phi = self
            .builder
            .build_phi(bool_type, "logicaltmp")
            .map_err(builder_error)?;
        phi.add_incoming(&[(&right_condition, right_block), (&short_value, short_block)]);

        Ok(phi.as_basic_value())
    }

    fn generate_integer_binary(
        &self,
        left: IntValue<'ctx>,
        operator: BinaryOp,
        right: IntValue<'ctx>,
        span: Span,
        unsigned: bool,
    ) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        let value = match operator {
            BinaryOp::Add => self.builder.build_int_add(left, right, "addtmp")?.into(),
            BinaryOp::Subtract => self.builder.build_int_sub(left, right, "subtmp")?.into(),
            BinaryOp::Multiply => self.builder.build_int_mul(left, right, "multmp")?.into(),
            BinaryOp::Divide => {
                if unsigned {
                    self.builder
                        .build_int_unsigned_div(left, right, "divtmp")?
                        .into()
                } else {
                    self.builder
                        .build_int_signed_div(left, right, "divtmp")?
                        .into()
                }
            }
            BinaryOp::Modulo => {
                if unsigned {
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
            BinaryOp::LogicalAnd => self.builder.build_and(left, right, "andtmp")?.into(),
            BinaryOp::LogicalOr => self.builder.build_or(left, right, "ortmp")?.into(),
            BinaryOp::BitwiseAnd => self.builder.build_and(left, right, "andtmp")?.into(),
            BinaryOp::BitwiseOr => self.builder.build_or(left, right, "ortmp")?.into(),
            BinaryOp::BitwiseXor => self.builder.build_xor(left, right, "xortmp")?.into(),
            BinaryOp::ShiftLeft => self.builder.build_left_shift(left, right, "shltmp")?.into(),
            BinaryOp::ShiftRight => self
                .builder
                .build_right_shift(left, right, !unsigned, "shrtmp")?
                .into(),
        };

        if matches!(operator, BinaryOp::LogicalAnd | BinaryOp::LogicalOr)
            && left.get_type().get_bit_width() != 1
        {
            return Err(CodegenError::at(
                "logical operators require boolean operands",
                span,
            ));
        }

        Ok(value)
    }

    fn generate_float_binary(
        &self,
        left: inkwell::values::FloatValue<'ctx>,
        operator: BinaryOp,
        right: inkwell::values::FloatValue<'ctx>,
        span: Span,
    ) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        let value = match operator {
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
                    "bitwise and logical operators require integer operands",
                    span,
                ));
            }
        };

        Ok(value)
    }

    fn generate_call(
        &mut self,
        expression: &Expr,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let Expr::Call {
            callee,
            arguments,
            span,
        } = expression
        else {
            return Err(CodegenError::new(
                "internal error: expected a call expression",
            ));
        };

        let Expr::Identifier { name, .. } = callee.as_ref() else {
            return Err(CodegenError::at(
                "only named functions can be called currently",
                callee.span(),
            ));
        };

        let function = self
            .module
            .get_function(name)
            .ok_or_else(|| CodegenError::at(format!("unknown function `{name}`"), *span))?;
        let parameter_types = function.get_type().get_param_types();

        if parameter_types.len() != arguments.len() {
            return Err(CodegenError::at(
                format!(
                    "function `{name}` expects {} arguments, got {}",
                    parameter_types.len(),
                    arguments.len()
                ),
                *span,
            ));
        }

        let mut values = Vec::with_capacity(arguments.len());
        for (argument, parameter_type) in arguments.iter().zip(parameter_types) {
            let parameter_type: BasicTypeEnum = parameter_type.try_into().map_err(|_| {
                CodegenError::at("function parameter is not a value type", argument.span())
            })?;
            let value = self.generate_expression(argument, Some(parameter_type))?;
            values.push(BasicMetadataValueEnum::from(value));
        }

        let call = self
            .builder
            .build_call(function, &values, "calltmp")
            .map_err(builder_error)?;

        Ok(call.try_as_basic_value().basic())
    }

    fn as_condition(
        &self,
        value: BasicValueEnum<'ctx>,
        span: Span,
    ) -> Result<IntValue<'ctx>, CodegenError> {
        match value {
            BasicValueEnum::IntValue(value) if value.get_type().get_bit_width() == 1 => Ok(value),
            BasicValueEnum::IntValue(value) => self
                .builder
                .build_int_compare(
                    IntPredicate::NE,
                    value,
                    value.get_type().const_zero(),
                    "ifcond",
                )
                .map_err(builder_error),
            BasicValueEnum::FloatValue(value) => self
                .builder
                .build_float_compare(
                    FloatPredicate::ONE,
                    value,
                    value.get_type().const_zero(),
                    "ifcond",
                )
                .map_err(builder_error),
            _ => Err(CodegenError::at(
                "if condition must be an integer or floating-point value",
                span,
            )),
        }
    }

    fn current_function(&self) -> Option<FunctionValue<'ctx>> {
        let block = self.builder.get_insert_block()?;
        block.get_parent()
    }

    fn source_type_of_expression(&self, expression: &Expr) -> Option<Type> {
        match expression {
            Expr::Integer { .. } => Some(Type::Named("i32".to_string())),
            Expr::Float { .. } => Some(Type::Named("f64".to_string())),
            Expr::Bool { .. } => Some(Type::Named("bool".to_string())),
            Expr::Identifier { name, .. } => {
                self.locals.get(name).map(|local| local.source_ty.clone())
            }
            Expr::Unary { operand, .. } => self.source_type_of_expression(operand),
            Expr::Binary { left, right, .. } => self
                .source_type_of_expression(left)
                .or_else(|| self.source_type_of_expression(right)),
            Expr::Call { callee, .. } => match callee.as_ref() {
                Expr::Identifier { name, .. } => self.function_return_types.get(name).cloned(),
                _ => None,
            },
        }
    }

    fn expression_is_unsigned(&self, expression: &Expr) -> bool {
        matches!(
            self.source_type_of_expression(expression),
            Some(Type::Named(name)) if matches!(name.as_str(), "u8" | "u16" | "u32" | "u64" | "u128" | "usize")
        )
    }

    fn basic_type(&self, ty: &Type) -> Result<BasicTypeEnum<'ctx>, CodegenError> {
        let Type::Named(name) = ty else {
            return Err(CodegenError::new("unit is not a value type"));
        };

        let value = match name.as_str() {
            "bool" => self.context.bool_type().into(),
            "i8" => self.context.i8_type().into(),
            "i16" => self.context.i16_type().into(),
            "i32" => self.context.i32_type().into(),
            "i64" => self.context.i64_type().into(),
            "i128" => self.context.i128_type().into(),
            "u8" => self.context.i8_type().into(),
            "u16" => self.context.i16_type().into(),
            "u32" => self.context.i32_type().into(),
            "u64" => self.context.i64_type().into(),
            "u128" => self.context.i128_type().into(),
            "f32" => self.context.f32_type().into(),
            "f64" => self.context.f64_type().into(),
            "usize" | "isize" => self
                .context
                .custom_width_int_type(
                    NonZeroU32::new(self.pointer_width)
                        .ok_or_else(|| CodegenError::new("target pointer width is zero"))?,
                )
                .map_err(CodegenError::new)?
                .into(),
            "void" => return Err(CodegenError::new("void is not a value type")),
            _ => return Err(CodegenError::new(format!("unknown type `{name}`"))),
        };

        Ok(value)
    }
}

fn is_void(ty: &Type) -> bool {
    matches!(ty, Type::Unit) || matches!(ty, Type::Named(name) if name == "void")
}

fn builder_error(error: BuilderError) -> CodegenError {
    CodegenError::new(format!("LLVM builder error: {error:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn generate(source: &str) -> String {
        let mut lexer = Lexer::new(source);
        let mut tokens = Vec::new();

        loop {
            let token = lexer.next_token().expect("lexing failed");
            let eof = token.kind == crate::lexer::TokenKind::Eof;
            tokens.push(token);
            if eof {
                break;
            }
        }

        let program = Parser::new(tokens).parse().expect("parsing failed");
        let context = Context::create();
        CodeGenerator::new(&context, "test")
            .generate(&program)
            .unwrap()
            .print_to_string()
            .to_string()
    }

    #[test]
    fn generates_minimal_main() {
        let ir = generate(
            r#"
            main :: () -> i32 {
                return 42;
            }
            "#,
        );

        assert!(ir.contains("define i32 @main()"));
        assert!(ir.contains("ret i32 42"));
    }

    #[test]
    fn generates_integer_arithmetic_and_calls() {
        let ir = generate(
            r#"
            add :: (a: i32, b: i32) -> i32 {
                return a + b;
            }

            main :: () -> i32 {
                return add(10, 20) * 2;
            }
            "#,
        );

        assert!(ir.contains("define i32 @add"));
        assert!(ir.contains("call i32 @add"));
        assert!(ir.contains("mul i32"));
    }

    #[test]
    fn generates_local_variables() {
        let ir = generate(
            r#"
            main :: () -> i32 {
                x := 10;
                x = x + 2;
                return x;
            }
            "#,
        );

        assert!(ir.contains("alloca i32"));
        assert!(ir.contains("store i32"));
        assert!(ir.contains("load i32"));
    }

    #[test]
    fn uses_unsigned_integer_operations() {
        let ir = generate(
            r#"
            less :: (a: u32, b: u32) -> bool {
                return a < b;
            }

            divide :: (a: u32, b: u32) -> u32 {
                return a / b;
            }

            main :: () -> i32 {
                return 0;
            }
            "#,
        );

        assert!(ir.contains("icmp ult i32"));
        assert!(ir.contains("udiv i32"));
    }

    #[test]
    fn lowers_logical_operators_with_short_circuit_blocks() {
        let ir = generate(
            r#"
            side_effect :: () -> bool {
                return true;
            }

            main :: () -> i32 {
                if false && side_effect() {
                    return 1;
                }
                return 0;
            }
            "#,
        );

        assert!(ir.contains("logical.right"));
        assert!(ir.contains("logical.short"));
        assert!(ir.contains("phi i1"));
        assert!(!ir.contains("and i1"));
    }
}
