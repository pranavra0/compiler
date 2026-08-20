//! Shared control-flow MIR.
//!
//! Expressions remain typed expressions so low-level operations are not
//! erased. Structured statements are lowered once into blocks and explicit
//! terminators; execution backends only need to implement blocks and values.

use std::collections::HashMap;
use std::fmt;

use crate::lexer::Span;
use crate::typed::{
    FunctionId, ResolvedType, TypedBlock, TypedExpr, TypedFunction, TypedProgram, TypedStmt,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

#[derive(Debug, Clone, PartialEq)]
pub struct MirProgram {
    pub functions: Vec<MirFunction>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MirFunction {
    pub id: FunctionId,
    pub name: String,
    pub params: Vec<crate::typed::TypedParameter>,
    pub return_type: ResolvedType,
    pub blocks: Vec<MirBlock>,
    pub entry: BlockId,
    pub span: Span,
    pub is_extern: bool,
    pub abi: Option<String>,
    pub link_name: Option<String>,
    pub exported: bool,
    /// Cleanup actions active in each block. Expression-level propagation can
    /// leave a function without reaching an explicit return terminator.
    pub cleanup: HashMap<BlockId, Vec<MirCleanup>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MirBlock {
    pub id: BlockId,
    pub instructions: Vec<MirInstruction>,
    pub terminator: MirTerminator,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MirInstruction {
    RunCleanup {
        /// Cleanup actions carry captured values and are explicit MIR data.
        actions: Vec<MirCleanup>,
    },
    Declare {
        id: crate::typed::LocalId,
        name: String,
        ty: ResolvedType,
        mutable: bool,
        value: TypedExpr,
        span: Span,
    },
    Store {
        target: crate::typed::TypedPlace,
        value: TypedExpr,
        ty: ResolvedType,
        span: Span,
    },
    Expr {
        expression: TypedExpr,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct MirCleanup {
    pub function: FunctionId,
    pub name: String,
    pub arguments: Vec<TypedExpr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MirTerminator {
    Jump(BlockId),
    Branch {
        condition: TypedExpr,
        then_block: BlockId,
        else_block: BlockId,
        span: Span,
    },
    Return(Option<TypedExpr>),
    Unreachable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MirError {
    BreakOutsideLoop { span: Span },
    ContinueOutsideLoop { span: Span },
}

impl MirError {
    pub fn span(&self) -> Span {
        match self {
            Self::BreakOutsideLoop { span } | Self::ContinueOutsideLoop { span } => *span,
        }
    }
}

impl fmt::Display for MirError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BreakOutsideLoop { .. } => write!(f, "break outside loop in MIR lowering"),
            Self::ContinueOutsideLoop { .. } => write!(f, "continue outside loop in MIR lowering"),
        }
    }
}
impl std::error::Error for MirError {}

impl MirProgram {
    pub fn lower(program: &TypedProgram) -> Result<Self, MirError> {
        program
            .functions
            .iter()
            .map(MirFunction::lower)
            .collect::<Result<Vec<_>, _>>()
            .map(|functions| Self { functions })
    }

    pub fn function(&self, id: FunctionId) -> Option<&MirFunction> {
        self.functions.iter().find(|function| function.id == id)
    }
}

impl MirFunction {
    pub fn lower(function: &TypedFunction) -> Result<Self, MirError> {
        let mut lowerer = Lowerer {
            blocks: vec![MirBlock {
                id: BlockId(0),
                instructions: Vec::new(),
                terminator: MirTerminator::Unreachable,
            }],
            cleanup: HashMap::new(),
            loops: Vec::new(),
            scope_depth: 0,
            defers: Vec::new(),
            next_capture: u32::MAX,
        };
        let (end_block, falls_through) = lowerer.lower_block(&function.body, BlockId(0))?;
        if falls_through {
            lowerer.block_mut(end_block).terminator = MirTerminator::Return(None);
        }
        for block in &mut lowerer.blocks {
            if matches!(block.terminator, MirTerminator::Unreachable) {
                // Unconnected blocks and blocks terminated during lowering are
                // deliberately explicit. Backends must not infer fallthrough.
                block.terminator = MirTerminator::Unreachable;
            }
        }
        Ok(Self {
            id: function.id,
            name: function.name.clone(),
            params: function.params.clone(),
            return_type: function.return_type.clone(),
            blocks: lowerer.blocks,
            entry: BlockId(0),
            span: function.span,
            is_extern: function.is_extern,
            abi: function.abi.clone(),
            link_name: function.link_name.clone(),
            exported: function.exported,
            cleanup: lowerer.cleanup,
        })
    }
}

struct Lowerer {
    blocks: Vec<MirBlock>,
    cleanup: HashMap<BlockId, Vec<MirCleanup>>,
    loops: Vec<(BlockId, BlockId, usize)>,
    scope_depth: usize,
    defers: Vec<Vec<MirCleanup>>,
    next_capture: u32,
}

impl Lowerer {
    fn capture_id(&mut self) -> crate::typed::LocalId {
        let id = crate::typed::LocalId(self.next_capture);
        self.next_capture = self.next_capture.saturating_sub(1);
        id
    }

    fn cleanup_actions(&self, first_scope: usize) -> Vec<MirCleanup> {
        self.defers
            .iter()
            .skip(first_scope)
            .rev()
            .flat_map(|scope| scope.iter().rev().cloned())
            .collect()
    }

    fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(MirBlock {
            id,
            instructions: Vec::new(),
            terminator: MirTerminator::Unreachable,
        });
        id
    }

    fn block_mut(&mut self, id: BlockId) -> &mut MirBlock {
        &mut self.blocks[id.0 as usize]
    }

    /// Lower a block and return whether control can reach the end of it.
    fn lower_block(
        &mut self,
        block: &TypedBlock,
        mut current: BlockId,
    ) -> Result<(BlockId, bool), MirError> {
        self.scope_depth += 1;
        self.defers.push(Vec::new());
        self.cleanup.insert(current, self.cleanup_actions(0));
        let block_scope_depth = self.scope_depth;
        let mut falls_through = true;
        for statement in &block.statements {
            if !falls_through {
                break;
            }
            match statement {
                TypedStmt::If {
                    condition,
                    then_branch,
                    else_branch,
                    span,
                } => {
                    let then_block = self.new_block();
                    let else_block = self.new_block();
                    let join_block = self.new_block();
                    self.block_mut(current).terminator = MirTerminator::Branch {
                        condition: condition.clone(),
                        then_block,
                        else_block,
                        span: *span,
                    };
                    let (then_end, then_falls) = self.lower_block(then_branch, then_block)?;
                    if then_falls {
                        self.block_mut(then_end).terminator = MirTerminator::Jump(join_block);
                    }
                    let (else_end, else_falls) = if let Some(else_branch) = else_branch {
                        self.lower_block(else_branch, else_block)?
                    } else {
                        (else_block, true)
                    };
                    if else_falls {
                        self.block_mut(else_end).terminator = MirTerminator::Jump(join_block);
                    }
                    falls_through = then_falls || else_falls;
                    current = join_block;
                }
                TypedStmt::While {
                    condition,
                    body,
                    span,
                } => {
                    let condition_block = self.new_block();
                    let body_block = self.new_block();
                    let exit_block = self.new_block();
                    self.block_mut(current).terminator = MirTerminator::Jump(condition_block);
                    self.block_mut(condition_block).terminator = MirTerminator::Branch {
                        condition: condition.clone(),
                        then_block: body_block,
                        else_block: exit_block,
                        span: *span,
                    };
                    self.loops
                        .push((exit_block, condition_block, self.scope_depth));
                    let (body_end, body_falls) = self.lower_block(body, body_block)?;
                    self.loops.pop();
                    if body_falls {
                        self.block_mut(body_end).terminator = MirTerminator::Jump(condition_block);
                    }
                    current = exit_block;
                    falls_through = true;
                }
                TypedStmt::Break { span } => {
                    let Some((break_block, _, scope_depth)) = self.loops.last().copied() else {
                        return Err(MirError::BreakOutsideLoop { span: *span });
                    };
                    let actions = self.cleanup_actions(scope_depth);
                    self.block_mut(current)
                        .instructions
                        .push(MirInstruction::RunCleanup { actions });
                    self.block_mut(current).terminator = MirTerminator::Jump(break_block);
                    falls_through = false;
                }
                TypedStmt::Continue { span } => {
                    let Some((_, continue_block, scope_depth)) = self.loops.last().copied() else {
                        return Err(MirError::ContinueOutsideLoop { span: *span });
                    };
                    let actions = self.cleanup_actions(scope_depth);
                    self.block_mut(current)
                        .instructions
                        .push(MirInstruction::RunCleanup { actions });
                    self.block_mut(current).terminator = MirTerminator::Jump(continue_block);
                    falls_through = false;
                }
                TypedStmt::Return { value, .. } => {
                    let actions = self.cleanup_actions(0);
                    self.block_mut(current)
                        .instructions
                        .push(MirInstruction::RunCleanup { actions });
                    self.block_mut(current).terminator = MirTerminator::Return(value.clone());
                    falls_through = false;
                }
                TypedStmt::Defer {
                    function,
                    name,
                    arguments,
                    span,
                } => {
                    let mut captured = Vec::with_capacity(arguments.len());
                    for argument in arguments {
                        let id = self.capture_id();
                        let ty = argument.ty();
                        self.block_mut(current)
                            .instructions
                            .push(MirInstruction::Declare {
                                id,
                                name: format!("$defer{}", id.0),
                                ty: ty.clone(),
                                mutable: false,
                                value: argument.clone(),
                                span: *span,
                            });
                        captured.push(TypedExpr::Load {
                            id,
                            name: format!("$defer{}", id.0),
                            ty,
                            span: argument.span(),
                        });
                    }
                    self.defers
                        .last_mut()
                        .expect("MIR scope stack is established before statements")
                        .push(MirCleanup {
                            function: *function,
                            name: name.clone(),
                            arguments: captured,
                            span: *span,
                        });
                }
                TypedStmt::Declare {
                    id,
                    name,
                    ty,
                    mutable,
                    value,
                    span,
                } => self
                    .block_mut(current)
                    .instructions
                    .push(MirInstruction::Declare {
                        id: *id,
                        name: name.clone(),
                        ty: ty.clone(),
                        mutable: *mutable,
                        value: value.clone(),
                        span: *span,
                    }),
                TypedStmt::Store {
                    target,
                    value,
                    ty,
                    span,
                } => self
                    .block_mut(current)
                    .instructions
                    .push(MirInstruction::Store {
                        target: target.clone(),
                        value: value.clone(),
                        ty: ty.clone(),
                        span: *span,
                    }),
                TypedStmt::Expr { expression, span } => {
                    self.block_mut(current)
                        .instructions
                        .push(MirInstruction::Expr {
                            expression: expression.clone(),
                            span: *span,
                        })
                }
            }
        }
        if falls_through {
            let actions = self.cleanup_actions(self.scope_depth - 1);
            self.block_mut(current)
                .instructions
                .push(MirInstruction::RunCleanup { actions });
        }
        // Later expressions in this block must see defers registered above.
        self.cleanup.insert(current, self.cleanup_actions(0));
        debug_assert_eq!(self.scope_depth, block_scope_depth);
        self.defers.pop();
        self.scope_depth -= 1;
        Ok((current, falls_through))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline;

    #[test]
    fn lowers_structured_control_flow_to_explicit_blocks() {
        let typed = pipeline::analyze_source(
            "main :: () -> i32 { var x: i32 = 0; while x < 3 { if x == 1 { break; } x = x + 1; } return x; }",
        )
        .unwrap();
        let mir = MirProgram::lower(&typed).unwrap();
        let function = &mir.functions[0];
        assert!(
            function
                .blocks
                .iter()
                .any(|block| { matches!(block.terminator, MirTerminator::Branch { .. }) })
        );
        assert!(
            function
                .blocks
                .iter()
                .any(|block| { matches!(block.terminator, MirTerminator::Jump(_)) })
        );
        assert!(
            function
                .blocks
                .iter()
                .any(|block| { matches!(block.terminator, MirTerminator::Return(Some(_))) })
        );
    }
}
