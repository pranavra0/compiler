use crate::ast::Program;
use crate::ast::{BinaryOp, UnaryOp};
use crate::lexer::Span;
use crate::semantic;
use crate::typed::{
    IntegerWidth, LayoutKind, ResolvedType, TypedBlock, TypedExpr, TypedFunction, TypedPlace,
    TypedProgram, TypedStmt,
};
use inkwell::AddressSpace;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::BuilderError;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::targets::TargetData;
use inkwell::types::{BasicType, BasicTypeEnum, StringRadix, StructType};
use inkwell::values::{
    ArrayValue, BasicMetadataValueEnum, BasicValueEnum, FunctionValue, IntValue, PointerValue,
};
use inkwell::{FloatPredicate, IntPredicate};
use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroU32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodegenError {
    pub message: String,
    pub span: Option<Span>,
}
impl CodegenError {
    fn new(x: impl Into<String>) -> Self {
        Self {
            message: x.into(),
            span: None,
        }
    }
    fn at(x: impl Into<String>, s: Span) -> Self {
        Self {
            message: x.into(),
            span: Some(s),
        }
    }
}
impl fmt::Display for CodegenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.span {
            Some(s) => write!(f, "{} at {}..{}", self.message, s.start, s.end),
            None => f.write_str(&self.message),
        }
    }
}
impl std::error::Error for CodegenError {}
impl From<BuilderError> for CodegenError {
    fn from(e: BuilderError) -> Self {
        Self::new(format!("LLVM builder error: {e:?}"))
    }
}
#[derive(Clone)]
struct Local<'ctx> {
    pointer: PointerValue<'ctx>,
    llvm_type: BasicTypeEnum<'ctx>,
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
    fn contains(self, x: Self) -> bool {
        self.0 & x.0 != 0
    }
    fn union(self, x: Self) -> Self {
        Self(self.0 | x.0)
    }
    fn without(self, x: Self) -> Self {
        Self(self.0 & !x.0)
    }
}

pub struct CodeGenerator<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: inkwell::builder::Builder<'ctx>,
    locals: HashMap<usize, Local<'ctx>>,
    globals: HashMap<String, (PointerValue<'ctx>, BasicTypeEnum<'ctx>, ResolvedType)>,
    functions: HashMap<usize, FunctionValue<'ctx>>,
    structs: HashMap<String, StructType<'ctx>>,
    struct_fields: HashMap<String, HashMap<String, u32>>,
    current_function: Option<FunctionValue<'ctx>>,
    current_return_type: ResolvedType,
    pointer_width: u32,
    target_data: TargetData,
    loop_targets: Vec<LoopTargets<'ctx>>,
}
impl<'ctx> CodeGenerator<'ctx> {
    pub fn new(c: &'ctx Context, n: &str) -> Self {
        Self::with_pointer_width(c, n, usize::BITS)
    }
    pub fn with_target_data(c: &'ctx Context, n: &str, data: &TargetData) -> Self {
        let layout = data.get_data_layout();
        let layout_text = layout.as_str().to_string_lossy();
        let target_data = TargetData::create(&layout_text);
        let pointer_width = data.get_pointer_byte_size(None) * 8;
        let module = c.create_module(n);
        module.set_data_layout(&target_data.get_data_layout());
        Self {
            context: c,
            module,
            builder: c.create_builder(),
            locals: HashMap::new(),
            globals: HashMap::new(),
            functions: HashMap::new(),
            structs: HashMap::new(),
            struct_fields: HashMap::new(),
            current_function: None,
            current_return_type: ResolvedType::Unit,
            pointer_width,
            target_data,
            loop_targets: Vec::new(),
        }
    }
    pub fn with_pointer_width(c: &'ctx Context, n: &str, w: u32) -> Self {
        let target_data = TargetData::create(&format!("e-p:{w}:{w}"));
        let module = c.create_module(n);
        module.set_data_layout(&target_data.get_data_layout());
        Self {
            context: c,
            module,
            builder: c.create_builder(),
            locals: HashMap::new(),
            globals: HashMap::new(),
            functions: HashMap::new(),
            structs: HashMap::new(),
            struct_fields: HashMap::new(),
            current_function: None,
            current_return_type: ResolvedType::Unit,
            pointer_width: w,
            target_data,
            loop_targets: Vec::new(),
        }
    }
    pub fn generate(self, p: &Program) -> Result<Module<'ctx>, CodegenError> {
        let t = semantic::analyze_typed_with_pointer_width(p, self.pointer_width)
            .map_err(|e| CodegenError::new(format!("semantic error: {e}")))?;
        self.generate_typed(&t)
    }
    pub fn generate_typed(mut self, p: &TypedProgram) -> Result<Module<'ctx>, CodegenError> {
        self.declare_structs(p)?;
        self.declare_globals(p)?;
        self.declare_functions(p)?;
        for f in &p.functions {
            self.generate_function(f)?
        }
        self.module
            .verify()
            .map_err(|e| CodegenError::new(e.to_string()))?;
        Ok(self.module)
    }
    fn declare_structs(&mut self, p: &TypedProgram) -> Result<(), CodegenError> {
        for s in &p.structs {
            self.structs
                .insert(s.name.clone(), self.context.opaque_struct_type(&s.name));
        }
        for s in &p.structs {
            let st = self.structs[&s.name];
            self.struct_fields.insert(
                s.name.clone(),
                s.fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| (f.name.clone(), i as u32))
                    .collect(),
            );
            let fields = s
                .fields
                .iter()
                .map(|f| self.basic_type(f.ty.clone()))
                .collect::<Result<Vec<_>, _>>()?;
            st.set_body(&fields, false);
        }
        Ok(())
    }
    fn declare_globals(&mut self, p: &TypedProgram) -> Result<(), CodegenError> {
        for g in p
            .globals
            .iter()
            .map(|g| (&g.name, &g.ty, &g.value))
            .chain(p.constants.iter().map(|c| (&c.name, &c.ty, &c.value)))
        {
            let ty = self.basic_type(g.1.clone())?;
            let gv = self.module.add_global(ty, None, g.0);
            let init = self.generate_constant(g.2)?;
            gv.set_initializer(&init);
            if p.constants.iter().any(|constant| constant.name == *g.0) {
                gv.set_constant(true);
            }
            self.globals
                .insert(g.0.clone(), (gv.as_pointer_value(), ty, g.1.clone()));
        }
        Ok(())
    }
    fn declare_functions(&mut self, p: &TypedProgram) -> Result<(), CodegenError> {
        for f in &p.functions {
            let params = f
                .params
                .iter()
                .map(|x| self.basic_type(x.ty.clone()).map(Into::into))
                .collect::<Result<Vec<_>, _>>()?;
            let ft = if f.return_type == ResolvedType::Unit {
                self.context.void_type().fn_type(&params, false)
            } else {
                self.basic_type(f.return_type.clone())?
                    .fn_type(&params, false)
            };
            if self.module.get_function(&f.name).is_some() {
                return Err(CodegenError::at(
                    format!("function `{}` is declared more than once", f.name),
                    f.span,
                ));
            }
            self.functions
                .insert(f.id, self.module.add_function(&f.name, ft, None));
        }
        Ok(())
    }
    fn generate_function(&mut self, f: &TypedFunction) -> Result<(), CodegenError> {
        let fun = self.functions[&f.id];
        self.current_function = Some(fun);
        self.current_return_type = f.return_type.clone();
        self.locals.clear();
        self.loop_targets.clear();
        let entry = self.context.append_basic_block(fun, "entry");
        self.builder.position_at_end(entry);
        for (i, p) in f.params.iter().enumerate() {
            let v = fun.get_nth_param(i as u32).unwrap();
            let lt = self.basic_type(p.ty.clone())?;
            let ptr = self.builder.build_alloca(lt, &format!("{}.addr", p.name))?;
            self.builder.build_store(ptr, v)?;
            self.locals.insert(
                p.id,
                Local {
                    pointer: ptr,
                    llvm_type: lt,
                },
            );
        }
        let flow = self.generate_block(&f.body)?;
        if flow.contains(Flow::NORMAL) {
            if self.current_return_type != ResolvedType::Unit {
                return Err(CodegenError::at(
                    format!(
                        "function `{}` does not return a value on every path",
                        f.name
                    ),
                    f.body.span,
                ));
            }
            self.builder.build_return(None)?;
        }
        if !fun.verify(true) {
            return Err(CodegenError::new(format!(
                "LLVM verification failed for function `{}`",
                f.name
            )));
        }
        Ok(())
    }
    fn generate_block(&mut self, b: &TypedBlock) -> Result<Flow, CodegenError> {
        let saved = self.locals.clone();
        let mut flow = Flow::NORMAL;
        for s in &b.statements {
            if flow.contains(Flow::NORMAL) {
                flow = flow
                    .without(Flow::NORMAL)
                    .union(self.generate_statement(s)?);
            }
        }
        self.locals = saved;
        Ok(flow)
    }
    fn generate_statement(&mut self, s: &TypedStmt) -> Result<Flow, CodegenError> {
        match s {
            TypedStmt::Declare {
                id,
                name,
                ty,
                mutable: _,
                value,
                ..
            } => {
                let lt = self.basic_type(ty.clone())?;
                let v = self.generate_expression(value)?;
                if v.get_type() != lt {
                    return Err(CodegenError::at(
                        "initializer has the wrong resolved type",
                        s_span(s),
                    ));
                }
                let p = self.builder.build_alloca(lt, &format!("{name}.addr"))?;
                self.builder.build_store(p, v)?;
                self.locals.insert(
                    *id,
                    Local {
                        pointer: p,
                        llvm_type: lt,
                    },
                );
                Ok(Flow::NORMAL)
            }
            TypedStmt::Store {
                target,
                value,
                ty,
                span,
            } => {
                let p = self.place_pointer(target, *span)?;
                let v = self.generate_expression(value)?;
                if v.get_type() != self.basic_type(ty.clone())? {
                    return Err(CodegenError::at("assigned value has the wrong type", *span));
                }
                self.builder.build_store(p, v)?;
                Ok(Flow::NORMAL)
            }
            TypedStmt::Return { value, span } => {
                match (self.current_return_type.clone(), value) {
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
                    (_, Some(x)) => {
                        let v = self.generate_expression(x)?;
                        self.builder.build_return(Some(&v))?;
                    }
                }
                Ok(Flow::RETURN)
            }
            TypedStmt::Expr { expression, .. } => {
                self.generate_expression(expression)?;
                Ok(Flow::NORMAL)
            }
            TypedStmt::Break { span } => {
                let t = self
                    .loop_targets
                    .last()
                    .copied()
                    .ok_or_else(|| CodegenError::at("break is outside a loop", *span))?;
                self.builder.build_unconditional_branch(t.break_block)?;
                Ok(Flow::BREAK)
            }
            TypedStmt::Continue { span } => {
                let t = self
                    .loop_targets
                    .last()
                    .copied()
                    .ok_or_else(|| CodegenError::at("continue is outside a loop", *span))?;
                self.builder.build_unconditional_branch(t.continue_block)?;
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
        c: &TypedExpr,
        t: &TypedBlock,
        e: Option<&TypedBlock>,
    ) -> Result<Flow, CodegenError> {
        let raw = self.generate_expression(c)?;
        let cv = self.as_condition(raw, c.span())?;
        let f = self.current_function.unwrap();
        let tb = self.context.append_basic_block(f, "if.then");
        let eb = self.context.append_basic_block(f, "if.else");
        let end = self.context.append_basic_block(f, "if.end");
        self.builder.build_conditional_branch(cv, tb, eb)?;
        self.builder.position_at_end(tb);
        let tf = self.generate_block(t)?;
        if tf.contains(Flow::NORMAL) {
            self.builder.build_unconditional_branch(end)?;
        }
        self.builder.position_at_end(eb);
        let ef = match e {
            Some(x) => self.generate_block(x)?,
            None => Flow::NORMAL,
        };
        if ef.contains(Flow::NORMAL) {
            self.builder.build_unconditional_branch(end)?;
        }
        let flow = tf.union(ef);
        self.builder.position_at_end(end);
        if !flow.contains(Flow::NORMAL) {
            self.builder.build_unreachable()?;
        }
        Ok(flow)
    }
    fn generate_while(&mut self, c: &TypedExpr, b: &TypedBlock) -> Result<Flow, CodegenError> {
        let f = self.current_function.unwrap();
        let cb = self.context.append_basic_block(f, "while.cond");
        let bb = self.context.append_basic_block(f, "while.body");
        let eb = self.context.append_basic_block(f, "while.end");
        self.builder.build_unconditional_branch(cb)?;
        self.builder.position_at_end(cb);
        let raw = self.generate_expression(c)?;
        let cv = self.as_condition(raw, c.span())?;
        let always = matches!(c, TypedExpr::Bool { value: true, .. });
        if always {
            self.builder.build_unconditional_branch(bb)?;
        } else {
            self.builder.build_conditional_branch(cv, bb, eb)?;
        }
        self.loop_targets.push(LoopTargets {
            continue_block: cb,
            break_block: eb,
        });
        self.builder.position_at_end(bb);
        let bf = self.generate_block(b)?;
        self.loop_targets.pop();
        if bf.contains(Flow::NORMAL) {
            self.builder.build_unconditional_branch(cb)?;
        }
        self.builder.position_at_end(eb);
        let exit = !always || bf.contains(Flow::BREAK);
        if always && !exit {
            self.builder.build_unreachable()?;
        }
        let mut flow = if bf.contains(Flow::RETURN) {
            Flow::RETURN
        } else {
            Flow(0)
        };
        if exit {
            flow = flow.union(Flow::NORMAL)
        }
        Ok(flow)
    }

    fn generate_expression(&mut self, e: &TypedExpr) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        match e {
            TypedExpr::Integer { value, ty, span } => {
                let BasicTypeEnum::IntType(t) = self.basic_type(ty.clone())? else {
                    return Err(CodegenError::at("integer has a non-integer type", *span));
                };
                Ok(
                    t.const_int_from_string(&value.to_string(), StringRadix::Decimal)
                        .ok_or_else(|| CodegenError::at("integer literal is out of range", *span))?
                        .into(),
                )
            }
            TypedExpr::Float { value, ty, span } => {
                let BasicTypeEnum::FloatType(t) = self.basic_type(ty.clone())? else {
                    return Err(CodegenError::at("float has a non-float type", *span));
                };
                Ok(t.const_float(*value).into())
            }
            TypedExpr::Bool { value, .. } => Ok(self
                .context
                .bool_type()
                .const_int(u64::from(*value), false)
                .into()),
            TypedExpr::Null { ty, span } => {
                let BasicTypeEnum::PointerType(pt) = self.basic_type(ty.clone())? else {
                    return Err(CodegenError::at("null requires a pointer type", *span));
                };
                Ok(pt.const_null().into())
            }
            TypedExpr::AddressOf { place, .. } => Ok(self.place_pointer(place, e.span())?.into()),
            TypedExpr::Dereference { place, ty, span } => {
                let p = self.place_pointer(place, *span)?;
                Ok(self
                    .builder
                    .build_load(self.basic_type(ty.clone())?, p, "deref.load")?)
            }
            TypedExpr::Layout {
                kind,
                target,
                field,
                span,
                ..
            } => {
                let value = self.layout_value(*kind, target, field.as_deref(), *span)?;
                Ok(self.usize_type().const_int(value, false).into())
            }
            TypedExpr::Load { id, span, .. } => {
                let l = self
                    .locals
                    .get(id)
                    .cloned()
                    .ok_or_else(|| CodegenError::at("unknown resolved local", *span))?;
                Ok(self.builder.build_load(l.llvm_type, l.pointer, "loadtmp")?)
            }
            TypedExpr::GlobalLoad { name, span, .. } => {
                let (g, ty, _) = self
                    .globals
                    .get(name)
                    .cloned()
                    .ok_or_else(|| CodegenError::at("unknown global", *span))?;
                Ok(self.builder.build_load(ty, g, "global.load")?)
            }
            TypedExpr::Field { place, .. } | TypedExpr::Index { place, .. } => {
                let p = self.place_pointer(place, e.span())?;
                let ty = self.basic_type(e.ty())?;
                Ok(self.builder.build_load(ty, p, "aggregate.load")?)
            }
            TypedExpr::StructLiteral { ty, fields, .. } => {
                let st = match ty {
                    ResolvedType::Struct(n) => self.structs[n],
                    _ => return Err(CodegenError::new("invalid struct literal type")),
                };
                let mut value = st.get_undef();
                for (i, x) in fields.iter().enumerate() {
                    let v = self.generate_expression(x)?;
                    value = self
                        .builder
                        .build_insert_value(value, v, i as u32, "struct.insert")?
                        .into_struct_value();
                }
                Ok(value.into())
            }
            TypedExpr::ArrayLiteral { ty, elements, .. } => {
                let BasicTypeEnum::ArrayType(at) = self.basic_type(ty.clone())? else {
                    return Err(CodegenError::new("invalid array literal type"));
                };
                let mut value = at.get_undef();
                for (i, x) in elements.iter().enumerate() {
                    let v = self.generate_expression(x)?;
                    value = self
                        .builder
                        .build_insert_value(value, v, i as u32, "array.insert")?
                        .into_array_value();
                }
                Ok(value.into())
            }
            TypedExpr::Unary {
                operator,
                operand,
                ty,
                span,
            } => {
                let v = self.generate_expression(operand)?;
                match (operator, v) {
                    (UnaryOp::Negate, BasicValueEnum::IntValue(x)) => {
                        Ok(self.builder.build_int_neg(x, "negtmp")?.into())
                    }
                    (UnaryOp::Negate, BasicValueEnum::FloatValue(x)) => {
                        Ok(self.builder.build_float_neg(x, "negtmp")?.into())
                    }
                    (UnaryOp::Not | UnaryOp::BitwiseNot, BasicValueEnum::IntValue(x)) => {
                        Ok(self.builder.build_not(x, "nottmp")?.into())
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
            } => self.generate_binary(left, *operator, right, operand_type.clone(), *span),
            TypedExpr::Call {
                function,
                name,
                arguments,
                ty,
                ..
            } => {
                if name == "make_slice" {
                    let pointer = self
                        .generate_expression(&arguments[0])?
                        .into_pointer_value();
                    let length = self.generate_expression(&arguments[1])?.into_int_value();
                    let not_null = self.builder.build_is_not_null(pointer, "slice.not_null")?;
                    let empty = self.builder.build_int_compare(
                        IntPredicate::EQ,
                        length,
                        length.get_type().const_zero(),
                        "slice.empty",
                    )?;
                    self.guard(
                        self.builder.build_or(not_null, empty, "slice.valid")?,
                        e.span(),
                    )?;
                    let st = self.slice_type();
                    let mut value = st.get_undef();
                    value = self
                        .builder
                        .build_insert_value(value, pointer, 0, "slice.ptr")?
                        .into_struct_value();
                    value = self
                        .builder
                        .build_insert_value(value, length, 1, "slice.len")?
                        .into_struct_value();
                    return Ok(value.into());
                }
                self.generate_call(*function, arguments, e.span())?
                    .ok_or_else(|| {
                        CodegenError::at(
                            format!("void function `{name}` has no value of type {ty:?}"),
                            e.span(),
                        )
                    })
            }
        }
    }
    fn generate_binary(
        &mut self,
        l: &TypedExpr,
        op: BinaryOp,
        r: &TypedExpr,
        ty: ResolvedType,
        span: Span,
    ) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        if matches!(op, BinaryOp::LogicalAnd | BinaryOp::LogicalOr) {
            return self.generate_logical(l, op, r, span);
        }
        let a = self.generate_expression(l)?;
        let b = self.generate_expression(r)?;
        if let (BasicValueEnum::PointerValue(a), BasicValueEnum::IntValue(b)) = (a, b) {
            let ResolvedType::Pointer(element) = l.ty() else {
                return Err(CodegenError::at(
                    "pointer arithmetic has a non-pointer lhs",
                    span,
                ));
            };
            let et = self.basic_type(*element)?;
            let index = if op == BinaryOp::Subtract {
                self.builder.build_int_neg(b, "ptr.neg")?
            } else {
                b
            };
            return Ok(unsafe { self.builder.build_gep(et, a, &[index], "ptr.add") }?.into());
        }
        if let (BasicValueEnum::PointerValue(a), BasicValueEnum::PointerValue(b)) = (a, b) {
            if op == BinaryOp::Subtract {
                let ResolvedType::Pointer(element) = l.ty() else {
                    return Err(CodegenError::at(
                        "pointer subtraction has a non-pointer lhs",
                        span,
                    ));
                };
                return Ok(self
                    .builder
                    .build_ptr_diff(self.basic_type(*element)?, a, b, "ptr.diff")?
                    .into());
            }
            let pred = match op {
                BinaryOp::Equal => IntPredicate::EQ,
                BinaryOp::NotEqual => IntPredicate::NE,
                BinaryOp::Less => IntPredicate::ULT,
                BinaryOp::LessEqual => IntPredicate::ULE,
                BinaryOp::Greater => IntPredicate::UGT,
                BinaryOp::GreaterEqual => IntPredicate::UGE,
                _ => return Err(CodegenError::at("invalid pointer operation", span)),
            };
            return Ok(self
                .builder
                .build_int_compare(
                    pred,
                    self.builder
                        .build_ptr_to_int(a, self.usize_type(), "ptr.a")?,
                    self.builder
                        .build_ptr_to_int(b, self.usize_type(), "ptr.b")?,
                    "ptr.cmp",
                )?
                .into());
        }
        match (a, b) {
            (BasicValueEnum::IntValue(a), BasicValueEnum::IntValue(b)) => {
                self.generate_integer(a, op, b, ty, span)
            }
            (BasicValueEnum::FloatValue(a), BasicValueEnum::FloatValue(b)) => {
                self.generate_float(a, op, b, span)
            }
            _ => Err(CodegenError::at(
                "binary operands have incompatible resolved types",
                span,
            )),
        }
    }
    fn generate_logical(
        &mut self,
        l: &TypedExpr,
        op: BinaryOp,
        r: &TypedExpr,
        _span: Span,
    ) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        let raw_left = self.generate_expression(l)?;
        let a = self.as_condition(raw_left, l.span())?;
        let f = self.current_function.unwrap();
        let rb = self.context.append_basic_block(f, "logical.right");
        let sb = self.context.append_basic_block(f, "logical.short");
        let end = self.context.append_basic_block(f, "logical.end");
        if op == BinaryOp::LogicalAnd {
            self.builder.build_conditional_branch(a, rb, sb)?;
        } else {
            self.builder.build_conditional_branch(a, sb, rb)?;
        }
        self.builder.position_at_end(rb);
        let raw_right = self.generate_expression(r)?;
        let b = self.as_condition(raw_right, r.span())?;
        self.builder.build_unconditional_branch(end)?;
        self.builder.position_at_end(sb);
        let short = self
            .context
            .bool_type()
            .const_int(u64::from(op == BinaryOp::LogicalOr), false);
        self.builder.build_unconditional_branch(end)?;
        self.builder.position_at_end(end);
        let phi = self
            .builder
            .build_phi(self.context.bool_type(), "logicaltmp")?;
        phi.add_incoming(&[(&b, rb), (&short, sb)]);
        Ok(phi.as_basic_value())
    }
    fn generate_integer(
        &mut self,
        a: IntValue<'ctx>,
        op: BinaryOp,
        b: IntValue<'ctx>,
        ty: ResolvedType,
        span: Span,
    ) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        if ty == ResolvedType::Bool {
            return Ok(match op {
                BinaryOp::Equal => self
                    .builder
                    .build_int_compare(IntPredicate::EQ, a, b, "eq")?
                    .into(),
                BinaryOp::NotEqual => self
                    .builder
                    .build_int_compare(IntPredicate::NE, a, b, "ne")?
                    .into(),
                _ => return Err(CodegenError::at("invalid boolean operator", span)),
            });
        }
        let ResolvedType::Integer { width, signed } = ty else {
            return Err(CodegenError::at(
                "integer operation has non-integer type",
                span,
            ));
        };
        let u = !signed;
        Ok(match op {
            BinaryOp::Add => self.builder.build_int_add(a, b, "add")?.into(),
            BinaryOp::Subtract => self.builder.build_int_sub(a, b, "sub")?.into(),
            BinaryOp::Multiply => self.builder.build_int_mul(a, b, "mul")?.into(),
            BinaryOp::Divide | BinaryOp::Modulo => {
                self.guard_div(a, b, signed, span)?;
                if op == BinaryOp::Divide {
                    if u {
                        self.builder.build_int_unsigned_div(a, b, "div")?.into()
                    } else {
                        self.builder.build_int_signed_div(a, b, "div")?.into()
                    }
                } else if u {
                    self.builder.build_int_unsigned_rem(a, b, "rem")?.into()
                } else {
                    self.builder.build_int_signed_rem(a, b, "rem")?.into()
                }
            }
            BinaryOp::Equal => self
                .builder
                .build_int_compare(IntPredicate::EQ, a, b, "eq")?
                .into(),
            BinaryOp::NotEqual => self
                .builder
                .build_int_compare(IntPredicate::NE, a, b, "ne")?
                .into(),
            BinaryOp::Less => self
                .builder
                .build_int_compare(
                    if u {
                        IntPredicate::ULT
                    } else {
                        IntPredicate::SLT
                    },
                    a,
                    b,
                    "lt",
                )?
                .into(),
            BinaryOp::LessEqual => self
                .builder
                .build_int_compare(
                    if u {
                        IntPredicate::ULE
                    } else {
                        IntPredicate::SLE
                    },
                    a,
                    b,
                    "le",
                )?
                .into(),
            BinaryOp::Greater => self
                .builder
                .build_int_compare(
                    if u {
                        IntPredicate::UGT
                    } else {
                        IntPredicate::SGT
                    },
                    a,
                    b,
                    "gt",
                )?
                .into(),
            BinaryOp::GreaterEqual => self
                .builder
                .build_int_compare(
                    if u {
                        IntPredicate::UGE
                    } else {
                        IntPredicate::SGE
                    },
                    a,
                    b,
                    "ge",
                )?
                .into(),
            BinaryOp::BitwiseAnd => self.builder.build_and(a, b, "and")?.into(),
            BinaryOp::BitwiseOr => self.builder.build_or(a, b, "or")?.into(),
            BinaryOp::BitwiseXor => self.builder.build_xor(a, b, "xor")?.into(),
            BinaryOp::ShiftLeft | BinaryOp::ShiftRight => {
                let bits = match width {
                    IntegerWidth::Bits(x) => x as u64,
                    IntegerWidth::Pointer => self.pointer_width as u64,
                };
                let valid = self.builder.build_int_compare(
                    IntPredicate::ULT,
                    b,
                    a.get_type().const_int(bits, false),
                    "shift.valid",
                )?;
                self.guard(valid, span)?;
                if op == BinaryOp::ShiftLeft {
                    self.builder.build_left_shift(a, b, "shl")?.into()
                } else {
                    self.builder.build_right_shift(a, b, signed, "shr")?.into()
                }
            }
            BinaryOp::LogicalAnd | BinaryOp::LogicalOr => {
                return Err(CodegenError::at("logical operators require bool", span));
            }
        })
    }
    fn guard_div(
        &mut self,
        a: IntValue<'ctx>,
        b: IntValue<'ctx>,
        signed: bool,
        s: Span,
    ) -> Result<(), CodegenError> {
        let nz = self.builder.build_int_compare(
            IntPredicate::NE,
            b,
            b.get_type().const_zero(),
            "div.nonzero",
        )?;
        self.guard(nz, s)?;
        if signed {
            let bits = a.get_type().get_bit_width();
            let min = self.builder.build_left_shift(
                a.get_type().const_int(1, false),
                a.get_type().const_int(u64::from(bits - 1), false),
                "min",
            )?;
            let ov = self.builder.build_and(
                self.builder
                    .build_int_compare(IntPredicate::EQ, a, min, "ismin")?,
                self.builder.build_int_compare(
                    IntPredicate::EQ,
                    b,
                    a.get_type().const_all_ones(),
                    "isneg",
                )?,
                "overflow",
            )?;
            self.guard(self.builder.build_not(ov, "safe")?, s)?
        }
        Ok(())
    }
    fn guard(&mut self, c: IntValue<'ctx>, _s: Span) -> Result<(), CodegenError> {
        let f = self.current_function.unwrap();
        let trap = self.context.append_basic_block(f, "trap");
        let cont = self.context.append_basic_block(f, "after.trap");
        self.builder.build_conditional_branch(c, cont, trap)?;
        self.builder.position_at_end(trap);
        let tf = self.module.get_function("llvm.trap").unwrap_or_else(|| {
            self.module.add_function(
                "llvm.trap",
                self.context.void_type().fn_type(&[], false),
                None,
            )
        });
        self.builder.build_call(tf, &[], "trap")?;
        self.builder.build_unreachable()?;
        self.builder.position_at_end(cont);
        Ok(())
    }
    fn generate_float(
        &self,
        a: inkwell::values::FloatValue<'ctx>,
        op: BinaryOp,
        b: inkwell::values::FloatValue<'ctx>,
        s: Span,
    ) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        Ok(match op {
            BinaryOp::Add => self.builder.build_float_add(a, b, "add")?.into(),
            BinaryOp::Subtract => self.builder.build_float_sub(a, b, "sub")?.into(),
            BinaryOp::Multiply => self.builder.build_float_mul(a, b, "mul")?.into(),
            BinaryOp::Divide => self.builder.build_float_div(a, b, "div")?.into(),
            BinaryOp::Modulo => self.builder.build_float_rem(a, b, "rem")?.into(),
            BinaryOp::Equal => self
                .builder
                .build_float_compare(FloatPredicate::OEQ, a, b, "eq")?
                .into(),
            BinaryOp::NotEqual => self
                .builder
                .build_float_compare(FloatPredicate::ONE, a, b, "ne")?
                .into(),
            BinaryOp::Less => self
                .builder
                .build_float_compare(FloatPredicate::OLT, a, b, "lt")?
                .into(),
            BinaryOp::LessEqual => self
                .builder
                .build_float_compare(FloatPredicate::OLE, a, b, "le")?
                .into(),
            BinaryOp::Greater => self
                .builder
                .build_float_compare(FloatPredicate::OGT, a, b, "gt")?
                .into(),
            BinaryOp::GreaterEqual => self
                .builder
                .build_float_compare(FloatPredicate::OGE, a, b, "ge")?
                .into(),
            _ => return Err(CodegenError::at("invalid floating point operator", s)),
        })
    }
    fn generate_call(
        &mut self,
        id: usize,
        args: &[TypedExpr],
        s: Span,
    ) -> Result<Option<BasicValueEnum<'ctx>>, CodegenError> {
        let f = self
            .functions
            .get(&id)
            .copied()
            .ok_or_else(|| CodegenError::at("unknown resolved call target", s))?;
        let vals = args
            .iter()
            .map(|a| {
                self.generate_expression(a)
                    .map(BasicMetadataValueEnum::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self
            .builder
            .build_call(f, &vals, "call")?
            .try_as_basic_value()
            .basic())
    }

    fn place_pointer(
        &mut self,
        p: &TypedPlace,
        s: Span,
    ) -> Result<PointerValue<'ctx>, CodegenError> {
        match p {
            TypedPlace::Local { id, .. } => Ok(self
                .locals
                .get(id)
                .ok_or_else(|| CodegenError::at("unknown local", s))?
                .pointer),
            TypedPlace::Global { name, .. } => Ok(self
                .globals
                .get(name)
                .ok_or_else(|| CodegenError::at("unknown global", s))?
                .0),
            TypedPlace::Temporary { value, ty } => {
                let lt = self.basic_type(ty.clone())?;
                let v = self.generate_expression(value)?;
                let ptr = self.builder.build_alloca(lt, "temporary.addr")?;
                self.builder.build_store(ptr, v)?;
                Ok(ptr)
            }
            TypedPlace::Field { base, index, .. } => {
                let bp = self.place_pointer(base, s)?;
                let ResolvedType::Struct(n) = base.ty() else {
                    return Err(CodegenError::at("field base is not a struct", s));
                };
                let st = self.structs[&n];
                Ok(self.builder.build_struct_gep(st, bp, *index, "field.ptr")?)
            }
            TypedPlace::Index {
                base,
                index,
                checked,
                ..
            } => {
                let bp = self.place_pointer(base, s)?;
                let iv = self.generate_expression(index)?.into_int_value();
                match base.ty() {
                    ResolvedType::Array { .. } => {
                        let at = self.basic_type(base.ty())?.into_array_type();
                        if *checked {
                            self.check_index(iv, p_length(p), s)?;
                        }
                        let zero = iv.get_type().const_zero();
                        Ok(unsafe {
                            self.builder
                                .build_in_bounds_gep(at, bp, &[zero, iv], "index.ptr")
                        }?)
                    }
                    ResolvedType::Slice(ref element) => {
                        let st = self.slice_type();
                        let slice = self
                            .builder
                            .build_load(st, bp, "slice.load")?
                            .into_struct_value();
                        let ptr = self
                            .builder
                            .build_extract_value(slice, 0, "slice.ptr")?
                            .into_pointer_value();
                        let len = self
                            .builder
                            .build_extract_value(slice, 1, "slice.len")?
                            .into_int_value();
                        if *checked {
                            let valid = self.builder.build_int_compare(
                                IntPredicate::ULT,
                                iv,
                                len,
                                "index.valid",
                            )?;
                            self.guard(valid, s)?;
                        }
                        let et = self.basic_type((**element).clone())?;
                        Ok(unsafe { self.builder.build_gep(et, ptr, &[iv], "slice.index.ptr") }?)
                    }
                    ResolvedType::Pointer(ref element) => {
                        let ptr = self
                            .builder
                            .build_load(
                                self.context.ptr_type(AddressSpace::default()),
                                bp,
                                "pointer.load",
                            )?
                            .into_pointer_value();
                        let et = self.basic_type((**element).clone())?;
                        Ok(unsafe { self.builder.build_gep(et, ptr, &[iv], "pointer.index.ptr") }?)
                    }
                    _ => Err(CodegenError::at("index base is not indexable", s)),
                }
            }
            TypedPlace::Dereference { pointer, .. } => {
                Ok(self.generate_expression(pointer)?.into_pointer_value())
            }
        }
    }
    fn check_index(
        &mut self,
        index: IntValue<'ctx>,
        length: Option<u64>,
        s: Span,
    ) -> Result<(), CodegenError> {
        if let Some(length) = length {
            let valid = self.builder.build_int_compare(
                IntPredicate::ULT,
                index,
                index.get_type().const_int(length, false),
                "index.valid",
            )?;
            self.guard(valid, s)?;
        }
        Ok(())
    }
    fn as_condition(
        &self,
        v: BasicValueEnum<'ctx>,
        s: Span,
    ) -> Result<IntValue<'ctx>, CodegenError> {
        match v {
            BasicValueEnum::IntValue(x) if x.get_type().get_bit_width() == 1 => Ok(x),
            _ => Err(CodegenError::at("condition must be a bool", s)),
        }
    }
    fn generate_constant(&self, e: &TypedExpr) -> Result<BasicValueEnum<'ctx>, CodegenError> {
        match e {
            TypedExpr::Integer { value, ty, span } => {
                let BasicTypeEnum::IntType(t) = self.basic_type(ty.clone())? else {
                    return Err(CodegenError::at("invalid constant integer", *span));
                };
                Ok(
                    t.const_int_from_string(&value.to_string(), StringRadix::Decimal)
                        .ok_or_else(|| CodegenError::at("invalid global integer", *span))?
                        .into(),
                )
            }
            TypedExpr::Float { value, ty, .. } => Ok(self
                .basic_type(ty.clone())?
                .into_float_type()
                .const_float(*value)
                .into()),
            TypedExpr::Bool { value, .. } => Ok(self
                .context
                .bool_type()
                .const_int(u64::from(*value), false)
                .into()),
            TypedExpr::Layout {
                kind,
                target,
                field,
                span,
                ..
            } => {
                let value = self.layout_value(*kind, target, field.as_deref(), *span)?;
                Ok(self.usize_type().const_int(value, false).into())
            }
            TypedExpr::StructLiteral { ty, fields, .. } => {
                let st = self.structs[match ty {
                    ResolvedType::Struct(n) => n,
                    _ => return Err(CodegenError::new("invalid struct constant")),
                }];
                let vals = fields
                    .iter()
                    .map(|x| self.generate_constant(x))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(st.const_named_struct(&vals).into())
            }
            TypedExpr::ArrayLiteral { ty, elements, .. } => {
                let BasicTypeEnum::ArrayType(at) = self.basic_type(ty.clone())? else {
                    return Err(CodegenError::new("invalid array constant"));
                };
                let vals = elements
                    .iter()
                    .map(|x| self.generate_constant(x))
                    .collect::<Result<Vec<_>, _>>()?;
                let av = unsafe { ArrayValue::new_const_array(&at.get_element_type(), &vals) };
                Ok(av.into())
            }
            TypedExpr::Unary {
                operator, operand, ..
            } => {
                let value = self.generate_constant(operand)?;
                match (operator, value) {
                    (UnaryOp::Negate, BasicValueEnum::IntValue(v)) => Ok(v.const_neg().into()),
                    (UnaryOp::Not | UnaryOp::BitwiseNot, BasicValueEnum::IntValue(v)) => {
                        Ok(v.const_not().into())
                    }
                    _ => Err(CodegenError::new("invalid constant unary expression")),
                }
            }
            _ => Err(CodegenError::new("global initializer is not a constant")),
        }
    }
    fn basic_type(&self, t: ResolvedType) -> Result<BasicTypeEnum<'ctx>, CodegenError> {
        Ok(match t {
            ResolvedType::Bool => self.context.bool_type().into(),
            ResolvedType::Integer { width, .. } => match width {
                IntegerWidth::Bits(8) => self.context.i8_type().into(),
                IntegerWidth::Bits(16) => self.context.i16_type().into(),
                IntegerWidth::Bits(32) => self.context.i32_type().into(),
                IntegerWidth::Bits(64) => self.context.i64_type().into(),
                IntegerWidth::Bits(128) => self.context.i128_type().into(),
                IntegerWidth::Pointer => self
                    .context
                    .custom_width_int_type(
                        NonZeroU32::new(self.pointer_width)
                            .ok_or_else(|| CodegenError::new("zero pointer width"))?,
                    )
                    .map_err(CodegenError::new)?
                    .into(),
                IntegerWidth::Bits(x) => {
                    return Err(CodegenError::new(format!("unsupported integer width {x}")));
                }
            },
            ResolvedType::Float { bits: 32 } => self.context.f32_type().into(),
            ResolvedType::Float { bits: 64 } => self.context.f64_type().into(),
            ResolvedType::Float { bits, .. } => {
                return Err(CodegenError::new(format!("unsupported float width {bits}")));
            }
            ResolvedType::Struct(n) => self
                .structs
                .get(&n)
                .copied()
                .ok_or_else(|| CodegenError::new(format!("unknown struct `{n}`")))?
                .into(),
            ResolvedType::Array { length, element } => {
                let length = u32::try_from(length).map_err(|_| {
                    CodegenError::new("fixed array length exceeds the LLVM aggregate limit")
                })?;
                self.basic_type(*element)?.array_type(length).into()
            }
            ResolvedType::Pointer(_) => self.context.ptr_type(AddressSpace::default()).into(),
            ResolvedType::Slice(_) => self.slice_type().into(),
            ResolvedType::Unit => return Err(CodegenError::new("unit is not a value type")),
        })
    }
    fn usize_type(&self) -> inkwell::types::IntType<'ctx> {
        self.context
            .custom_width_int_type(NonZeroU32::new(self.pointer_width).unwrap())
            .unwrap()
    }
    fn slice_type(&self) -> StructType<'ctx> {
        self.context.struct_type(
            &[
                self.context.ptr_type(AddressSpace::default()).into(),
                self.usize_type().into(),
            ],
            false,
        )
    }
    fn layout_value(
        &self,
        kind: LayoutKind,
        ty: &ResolvedType,
        field: Option<&str>,
        span: Span,
    ) -> Result<u64, CodegenError> {
        let llvm = self.basic_type(ty.clone())?;
        let value = match kind {
            LayoutKind::Size => self.target_data.get_abi_size(&llvm),
            LayoutKind::Align => u64::from(self.target_data.get_abi_alignment(&llvm)),
            LayoutKind::Offset => {
                let Some(field) = field else {
                    return Err(CodegenError::at("offset_of requires a field", span));
                };
                let ResolvedType::Struct(name) = ty else {
                    return Err(CodegenError::at("offset_of requires a struct", span));
                };
                let st = self
                    .structs
                    .get(name)
                    .ok_or_else(|| CodegenError::at("unknown struct", span))?;
                let index = self.struct_field_index(name, field, span)?;
                self.target_data
                    .offset_of_element(st, index)
                    .ok_or_else(|| CodegenError::at("could not compute field offset", span))?
            }
        };
        Ok(value)
    }
    fn struct_field_index(&self, name: &str, field: &str, span: Span) -> Result<u32, CodegenError> {
        let _st = self
            .structs
            .get(name)
            .ok_or_else(|| CodegenError::at("unknown struct", span))?;
        self.struct_fields
            .get(name)
            .and_then(|fields| fields.get(field))
            .copied()
            .ok_or_else(|| CodegenError::at(format!("unknown field `{field}`"), span))
    }
}
fn p_length(place: &TypedPlace) -> Option<u64> {
    match place {
        TypedPlace::Index { length, .. } => *length,
        _ => None,
    }
}

fn s_span(s: &TypedStmt) -> Span {
    match s {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::parse_source;
    use crate::semantic::analyze_typed;
    use inkwell::OptimizationLevel;
    use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine};

    #[test]
    fn native_layout_matches_llvm_for_structs_and_arrays() {
        Target::initialize_native(&InitializationConfig::default()).unwrap();
        let triple = TargetMachine::get_default_triple();
        let target = Target::from_triple(&triple).unwrap();
        let machine = target
            .create_target_machine(
                &triple,
                "generic",
                "",
                OptimizationLevel::None,
                RelocMode::Default,
                CodeModel::Default,
            )
            .unwrap();
        let data = machine.get_target_data();
        let source = "Pair :: struct { tag: i8; value: i32; } main :: () -> i32 { return 0; }";
        let program = parse_source(source).unwrap();
        let typed = analyze_typed(&program).unwrap();
        let context = Context::create();
        let module = CodeGenerator::with_pointer_width(
            &context,
            "layout_test",
            data.get_pointer_byte_size(None) * 8,
        )
        .generate_typed(&typed)
        .unwrap();
        let pair = module.get_struct_type("Pair").unwrap();
        assert_eq!(data.get_abi_size(&pair), 8);
        assert_eq!(data.get_abi_alignment(&pair), 4);
        assert_eq!(data.offset_of_element(&pair, 0), Some(0));
        assert_eq!(data.offset_of_element(&pair, 1), Some(4));

        let array = context.i32_type().array_type(3);
        assert_eq!(data.get_abi_size(&array), 12);
        assert_eq!(data.get_abi_alignment(&array), 4);
    }
}
