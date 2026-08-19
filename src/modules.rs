//! Deterministic source-module loading and the small first-generation linker IR.
//!
//! The resolver deliberately flattens a module graph after resolving it.  The
//! language IR is still module independent, while private implementation names
//! cannot collide: declarations from `math.compy` become `math__name`.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::ast::*;
use crate::lexer::Span;
use crate::pipeline;

#[derive(Debug, Clone)]
pub struct Project {
    pub program: Program,
    pub root_source: String,
    pub root_path: PathBuf,
    /// Canonical source files in deterministic dependency order.
    pub dependencies: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct ProjectError {
    pub message: String,
    pub span: Span,
}
impl std::fmt::Display for ProjectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}
impl std::error::Error for ProjectError {}

struct Node {
    path: PathBuf,
    program: Program,
    prefix: String,
    aliases: HashMap<String, String>,
    visible_exports: HashMap<String, HashSet<String>>,
    /// Names explicitly exported by this module. Visibility is checked before
    /// flattening so imported access remains qualified and deterministic.
    exported: HashSet<String>,
}

pub fn resolve(root: impl AsRef<Path>, module_roots: &[PathBuf]) -> Result<Project, ProjectError> {
    let root = canonical(root.as_ref(), Span::new(0, 0))?;
    let root_source = fs::read_to_string(&root).map_err(|e| ProjectError {
        message: format!("could not read `{}`: {e}", root.display()),
        span: Span::new(0, 0),
    })?;
    let mut nodes = Vec::<Node>::new();
    let mut indexes = HashMap::<PathBuf, usize>::new();
    let mut visiting = Vec::<PathBuf>::new();
    visit(
        &root,
        true,
        module_roots,
        &mut nodes,
        &mut indexes,
        &mut visiting,
        Span::new(0, 0),
    )?;

    for node in &nodes {
        for declaration in &node.program.declarations {
            validate_visibility(declaration, node)?;
        }
    }
    let mut declarations = Vec::new();
    // Dependencies are emitted before users.  Function/type collection is
    // nevertheless two-pass, so this ordering is only for readable IR.
    for node in &nodes {
        for declaration in &node.program.declarations {
            declarations.push(rewrite_decl(declaration, node));
        }
    }
    Ok(Project {
        program: Program {
            imports: Vec::new(),
            declarations,
        },
        root_source,
        root_path: root,
        dependencies: nodes.iter().map(|node| node.path.clone()).collect(),
    })
}

fn visit(
    path: &Path,
    is_root: bool,
    roots: &[PathBuf],
    nodes: &mut Vec<Node>,
    indexes: &mut HashMap<PathBuf, usize>,
    visiting: &mut Vec<PathBuf>,
    edge_span: Span,
) -> Result<usize, ProjectError> {
    let path = canonical(path, edge_span)?;
    if let Some(index) = indexes.get(&path) {
        return Ok(*index);
    }
    if visiting.iter().any(|p| p == &path) {
        let chain = visiting
            .iter()
            .chain(std::iter::once(&path))
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(" -> ");
        return Err(ProjectError {
            message: format!("module import cycle: {chain}"),
            span: edge_span,
        });
    }
    visiting.push(path.clone());
    let source = fs::read_to_string(&path).map_err(|e| ProjectError {
        message: format!("could not read module `{}`: {e}", path.display()),
        span: Span::new(0, 0),
    })?;
    let program = pipeline::parse_source(&source).map_err(|e| ProjectError {
        message: format!("in module `{}`: {e}", path.display()),
        span: e.span(),
    })?;
    let prefix = if is_root {
        String::new()
    } else {
        path.file_stem()
            .and_then(|x| x.to_str())
            .unwrap_or("module")
            .replace('-', "_")
    };
    let exported = program
        .declarations
        .iter()
        .filter_map(|d| match d {
            Decl::Function(f) if f.exported => Some(f.name.clone()),
            Decl::Struct(s) if s.exported => Some(s.name.clone()),
            Decl::Variable(v) if v.exported => Some(v.name.clone()),
            _ => None,
        })
        .collect::<HashSet<_>>();
    let mut aliases = HashMap::new();
    let mut visible_exports = HashMap::new();
    for import in &program.imports {
        if aliases.contains_key(&import.alias) {
            return Err(ProjectError {
                message: format!("duplicate module alias `{}`", import.alias),
                span: import.span,
            });
        }
        let imported = find_module(&path, &import.path, roots, import.span)?;
        let imported_index = visit(
            &imported,
            false,
            roots,
            nodes,
            indexes,
            visiting,
            import.span,
        )?;
        let imported_prefix = nodes[imported_index].prefix.clone();
        let imported_exports = nodes[imported_index].exported.clone();
        aliases.insert(import.alias.clone(), imported_prefix);
        visible_exports.insert(import.alias.clone(), imported_exports);
    }
    visiting.pop();
    let index = nodes.len();
    indexes.insert(path.clone(), index);
    nodes.push(Node {
        path,
        program,
        prefix,
        aliases,
        visible_exports,
        exported,
    });
    Ok(index)
}

fn canonical(path: &Path, span: Span) -> Result<PathBuf, ProjectError> {
    fs::canonicalize(path).map_err(|e| ProjectError {
        message: format!("could not resolve module `{}`: {e}", path.display()),
        span,
    })
}

fn find_module(
    importer: &Path,
    requested: &str,
    roots: &[PathBuf],
    span: Span,
) -> Result<PathBuf, ProjectError> {
    let request = Path::new(requested);
    let mut candidates = Vec::new();
    let relative = importer.parent().unwrap_or(Path::new(".")).join(request);
    candidates.push(relative.clone());
    if request.extension().is_none() {
        candidates.push(relative.with_extension("compy"));
    }
    for root in roots {
        let p = root.join(request);
        candidates.push(p.clone());
        if request.extension().is_none() {
            candidates.push(p.with_extension("compy"));
        }
    }
    candidates.retain(|p| p.is_file());
    let mut canonical_candidates = candidates
        .into_iter()
        .filter_map(|path| fs::canonicalize(path).ok())
        .collect::<Vec<_>>();
    canonical_candidates.sort();
    canonical_candidates.dedup();
    match canonical_candidates.as_slice() {
        [only] => Ok(only.clone()),
        [] => Err(ProjectError {
            message: format!(
                "module `{requested}` imported by `{}` was not found",
                importer.display()
            ),
            span,
        }),
        many => Err(ProjectError {
            message: format!(
                "ambiguous module `{requested}`: {}",
                many.iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            span,
        }),
    }
}

fn validate_visibility(decl: &Decl, node: &Node) -> Result<(), ProjectError> {
    let check_type = |ty: &Type| validate_type_visibility(ty, node);
    match decl {
        Decl::Struct(s) => {
            for field in &s.fields {
                check_type(&field.ty)?;
            }
        }
        Decl::Variable(v) => {
            if let Some(ty) = &v.ty {
                check_type(ty)?;
            }
            validate_expr_visibility(&v.value, node)?;
        }
        Decl::Function(f) => {
            for p in &f.params {
                check_type(&p.ty)?;
            }
            check_type(&f.return_type)?;
            validate_block_visibility(&f.body, node)?;
        }
        Decl::Comptime { expression, .. } => validate_expr_visibility(expression, node)?,
    }
    Ok(())
}
fn validate_type_visibility(ty: &Type, node: &Node) -> Result<(), ProjectError> {
    match ty {
        Type::Named(name) if name.contains('.') => {
            let mut parts = name.splitn(2, '.');
            let alias = parts.next().unwrap();
            let item = parts.next().unwrap();
            if node.aliases.contains_key(alias)
                && !node
                    .visible_exports
                    .get(alias)
                    .is_some_and(|x| x.contains(item))
            {
                return Err(ProjectError {
                    message: format!("`{alias}.{item}` is not exported by module `{alias}`"),
                    span: Span::new(0, 0),
                });
            }
        }
        Type::Pointer(x) | Type::Slice(x) => validate_type_visibility(x, node)?,
        Type::Array { element, .. } => validate_type_visibility(element, node)?,
        Type::Result { success, error } => {
            validate_type_visibility(success, node)?;
            validate_type_visibility(error, node)?;
        }
        _ => {}
    }
    Ok(())
}
fn validate_block_visibility(block: &Block, node: &Node) -> Result<(), ProjectError> {
    for statement in &block.statements {
        validate_stmt_visibility(statement, node)?;
    }
    Ok(())
}
fn validate_stmt_visibility(stmt: &Stmt, node: &Node) -> Result<(), ProjectError> {
    match stmt {
        Stmt::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            validate_expr_visibility(condition, node)?;
            validate_block_visibility(then_branch, node)?;
            if let Some(b) = else_branch {
                validate_block_visibility(b, node)?;
            }
        }
        Stmt::While {
            condition, body, ..
        } => {
            validate_expr_visibility(condition, node)?;
            validate_block_visibility(body, node)?;
        }
        Stmt::Defer { call, .. }
        | Stmt::Expr {
            expression: call, ..
        } => validate_expr_visibility(call, node)?,
        Stmt::Return { value, .. } => {
            if let Some(x) = value {
                validate_expr_visibility(x, node)?;
            }
        }
        Stmt::Variable(v) => {
            if let Some(t) = &v.ty {
                validate_type_visibility(t, node)?;
            }
            validate_expr_visibility(&v.value, node)?;
        }
        Stmt::Assignment { target, value, .. } => {
            validate_expr_visibility(target, node)?;
            validate_expr_visibility(value, node)?;
        }
        Stmt::Break { .. } | Stmt::Continue { .. } => {}
    }
    Ok(())
}
fn validate_expr_visibility(expr: &Expr, node: &Node) -> Result<(), ProjectError> {
    match expr {
        Expr::Field { base, name, span } => {
            if let Expr::Identifier { name: alias, .. } = base.as_ref() {
                if node.aliases.contains_key(alias)
                    && !node
                        .visible_exports
                        .get(alias)
                        .is_some_and(|x| x.contains(name))
                {
                    return Err(ProjectError {
                        message: format!("`{alias}.{name}` is not exported by module `{alias}`"),
                        span: *span,
                    });
                }
            }
            validate_expr_visibility(base, node)?;
        }
        Expr::StructLiteral { name, fields, span } => {
            if name.contains('.') {
                let mut parts = name.splitn(2, '.');
                let alias = parts.next().unwrap();
                let item = parts.next().unwrap();
                if node.aliases.contains_key(alias)
                    && !node
                        .visible_exports
                        .get(alias)
                        .is_some_and(|x| x.contains(item))
                {
                    return Err(ProjectError {
                        message: format!("`{alias}.{item}` is not exported by module `{alias}`"),
                        span: *span,
                    });
                }
            }
            for f in fields {
                validate_expr_visibility(&f.value, node)?;
            }
        }
        Expr::Index { base, index, .. } | Expr::UncheckedIndex { base, index, .. } => {
            validate_expr_visibility(base, node)?;
            validate_expr_visibility(index, node)?;
        }
        Expr::Unary { operand, .. }
        | Expr::Propagate {
            expression: operand,
            ..
        }
        | Expr::Comptime {
            expression: operand,
            ..
        } => validate_expr_visibility(operand, node)?,
        Expr::Binary { left, right, .. } => {
            validate_expr_visibility(left, node)?;
            validate_expr_visibility(right, node)?;
        }
        Expr::Call {
            callee, arguments, ..
        } => {
            validate_expr_visibility(callee, node)?;
            for x in arguments {
                validate_expr_visibility(x, node)?;
            }
        }
        Expr::ArrayLiteral { ty, elements, .. } => {
            validate_type_visibility(ty, node)?;
            for x in elements {
                validate_expr_visibility(x, node)?;
            }
        }
        Expr::SizeOf { ty, .. } | Expr::AlignOf { ty, .. } => validate_type_visibility(ty, node)?,
        Expr::OffsetOf { ty, .. } => validate_type_visibility(ty, node)?,
        Expr::Integer { .. }
        | Expr::Float { .. }
        | Expr::Bool { .. }
        | Expr::Null { .. }
        | Expr::Identifier { .. } => {}
    }
    Ok(())
}

fn prefixed(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.into()
    } else {
        format!("{prefix}__{name}")
    }
}

fn local_names(node: &Node) -> HashSet<String> {
    node.program
        .declarations
        .iter()
        .filter_map(|d| match d {
            Decl::Function(x) => Some(x.name.clone()),
            Decl::Variable(x) => Some(x.name.clone()),
            Decl::Struct(x) => Some(x.name.clone()),
            Decl::Comptime { .. } => None,
        })
        .collect()
}
fn local_structs(node: &Node) -> HashSet<String> {
    node.program
        .declarations
        .iter()
        .filter_map(|d| match d {
            Decl::Struct(x) => Some(x.name.clone()),
            _ => None,
        })
        .collect()
}
fn map_name(node: &Node, name: &str, bound: &HashSet<String>) -> String {
    if bound.contains(name) {
        return name.into();
    }
    if local_names(node).contains(name) {
        prefixed(&node.prefix, name)
    } else {
        name.into()
    }
}
fn map_type(node: &Node, ty: &Type) -> Type {
    match ty {
        Type::Named(n) if local_structs(node).contains(n) => Type::Named(prefixed(&node.prefix, n)),
        Type::Named(n) if n.contains('.') => {
            let mut p = n.splitn(2, '.');
            let alias = p.next().unwrap();
            let name = p.next().unwrap();
            let visible = node
                .visible_exports
                .get(alias)
                .map(|names| names.contains(name))
                .unwrap_or(false);
            if !visible {
                Type::Named(format!("__private__{alias}__{name}"))
            } else {
                Type::Named(prefixed(
                    node.aliases.get(alias).map(String::as_str).unwrap_or(alias),
                    name,
                ))
            }
        }
        Type::Named(n) => Type::Named(n.clone()),
        Type::Unit => Type::Unit,
        Type::Pointer(x) => Type::Pointer(Box::new(map_type(node, x))),
        Type::Slice(x) => Type::Slice(Box::new(map_type(node, x))),
        Type::Array { length, element } => Type::Array {
            length: *length,
            element: Box::new(map_type(node, element)),
        },
        Type::Result { success, error } => Type::Result {
            success: Box::new(map_type(node, success)),
            error: Box::new(map_type(node, error)),
        },
    }
}

fn rewrite_decl(decl: &Decl, node: &Node) -> Decl {
    match decl {
        Decl::Struct(s) => Decl::Struct(StructDecl {
            name: prefixed(&node.prefix, &s.name),
            fields: s
                .fields
                .iter()
                .map(|f| StructField {
                    name: f.name.clone(),
                    ty: map_type(node, &f.ty),
                    span: f.span,
                })
                .collect(),
            span: s.span,
            exported: s.exported,
        }),
        Decl::Variable(v) => {
            let bound = HashSet::new();
            Decl::Variable(VariableDecl {
                name: prefixed(&node.prefix, &v.name),
                kind: v.kind,
                ty: v.ty.as_ref().map(|t| map_type(node, t)),
                value: rewrite_expr(&v.value, node, &bound),
                span: v.span,
                exported: v.exported,
            })
        }
        Decl::Function(f) => {
            let mut bound = f
                .params
                .iter()
                .map(|p| p.name.clone())
                .collect::<HashSet<_>>();
            let body = rewrite_block(&f.body, node, &mut bound);
            Decl::Function(FunctionDecl {
                name: prefixed(&node.prefix, &f.name),
                generic_params: f.generic_params.clone(),
                params: f
                    .params
                    .iter()
                    .map(|p| Parameter {
                        name: p.name.clone(),
                        ty: map_type(node, &p.ty),
                        span: p.span,
                    })
                    .collect(),
                return_type: map_type(node, &f.return_type),
                body,
                span: f.span,
                is_extern: f.is_extern,
                abi: f.abi.clone(),
                link_name: f.link_name.clone(),
                exported: f.exported,
            })
        }
        Decl::Comptime { expression, span } => Decl::Comptime {
            expression: rewrite_expr(expression, node, &HashSet::new()),
            span: *span,
        },
    }
}

fn rewrite_block(block: &Block, node: &Node, parent: &mut HashSet<String>) -> Block {
    let mut bound = parent.clone();
    let statements = block
        .statements
        .iter()
        .map(|s| rewrite_stmt(s, node, &mut bound))
        .collect();
    Block {
        statements,
        span: block.span,
    }
}
fn rewrite_stmt(stmt: &Stmt, node: &Node, bound: &mut HashSet<String>) -> Stmt {
    match stmt {
        Stmt::Variable(v) => {
            let value = rewrite_expr(&v.value, node, bound);
            bound.insert(v.name.clone());
            Stmt::Variable(VariableDecl {
                name: v.name.clone(),
                kind: v.kind,
                ty: v.ty.as_ref().map(|t| map_type(node, t)),
                value,
                span: v.span,
                exported: false,
            })
        }
        Stmt::If {
            condition,
            then_branch,
            else_branch,
            span,
        } => Stmt::If {
            condition: rewrite_expr(condition, node, bound),
            then_branch: rewrite_block(then_branch, node, bound),
            else_branch: else_branch.as_ref().map(|b| rewrite_block(b, node, bound)),
            span: *span,
        },
        Stmt::While {
            condition,
            body,
            span,
        } => Stmt::While {
            condition: rewrite_expr(condition, node, bound),
            body: rewrite_block(body, node, bound),
            span: *span,
        },
        Stmt::Break { span } => Stmt::Break { span: *span },
        Stmt::Continue { span } => Stmt::Continue { span: *span },
        Stmt::Defer { call, span } => Stmt::Defer {
            call: rewrite_expr(call, node, bound),
            span: *span,
        },
        Stmt::Return { value, span } => Stmt::Return {
            value: value.as_ref().map(|x| rewrite_expr(x, node, bound)),
            span: *span,
        },
        Stmt::Assignment {
            target,
            value,
            span,
        } => Stmt::Assignment {
            target: rewrite_expr(target, node, bound),
            value: rewrite_expr(value, node, bound),
            span: *span,
        },
        Stmt::Expr { expression, span } => Stmt::Expr {
            expression: rewrite_expr(expression, node, bound),
            span: *span,
        },
    }
}
fn rewrite_expr(expr: &Expr, node: &Node, bound: &HashSet<String>) -> Expr {
    let span = expr.span();
    match expr {
        Expr::Identifier { name, .. } => Expr::Identifier {
            name: map_name(node, name, bound),
            span,
        },
        Expr::StructLiteral { name, fields, .. } => Expr::StructLiteral {
            name: map_type_name(node, name),
            fields: fields
                .iter()
                .map(|f| StructInit {
                    name: f.name.clone(),
                    value: rewrite_expr(&f.value, node, bound),
                    span: f.span,
                })
                .collect(),
            span,
        },
        Expr::Field { base, name, .. } => {
            if let Expr::Identifier { name: alias, .. } = base.as_ref() {
                if let Some(prefix) = node.aliases.get(alias) {
                    let visible = node
                        .visible_exports
                        .get(alias)
                        .map(|names| names.contains(name))
                        .unwrap_or(false);
                    let name = if visible {
                        prefixed(prefix, name)
                    } else {
                        format!("__private__{alias}__{name}")
                    };
                    return Expr::Identifier { name, span };
                }
            }
            Expr::Field {
                base: Box::new(rewrite_expr(base, node, bound)),
                name: name.clone(),
                span,
            }
        }
        Expr::Index { base, index, .. } => Expr::Index {
            base: Box::new(rewrite_expr(base, node, bound)),
            index: Box::new(rewrite_expr(index, node, bound)),
            span,
        },
        Expr::UncheckedIndex { base, index, .. } => Expr::UncheckedIndex {
            base: Box::new(rewrite_expr(base, node, bound)),
            index: Box::new(rewrite_expr(index, node, bound)),
            span,
        },
        Expr::Unary {
            operator, operand, ..
        } => Expr::Unary {
            operator: *operator,
            operand: Box::new(rewrite_expr(operand, node, bound)),
            span,
        },
        Expr::Binary {
            left,
            operator,
            right,
            ..
        } => Expr::Binary {
            left: Box::new(rewrite_expr(left, node, bound)),
            operator: *operator,
            right: Box::new(rewrite_expr(right, node, bound)),
            span,
        },
        Expr::Call {
            callee, arguments, ..
        } => Expr::Call {
            callee: Box::new(rewrite_expr(callee, node, bound)),
            arguments: arguments
                .iter()
                .map(|x| rewrite_expr(x, node, bound))
                .collect(),
            span,
        },
        Expr::Propagate { expression, .. } => Expr::Propagate {
            expression: Box::new(rewrite_expr(expression, node, bound)),
            span,
        },
        Expr::Comptime { expression, .. } => Expr::Comptime {
            expression: Box::new(rewrite_expr(expression, node, bound)),
            span,
        },
        Expr::ArrayLiteral { ty, elements, .. } => Expr::ArrayLiteral {
            ty: map_type(node, ty),
            elements: elements
                .iter()
                .map(|x| rewrite_expr(x, node, bound))
                .collect(),
            span,
        },
        Expr::SizeOf { ty, .. } => Expr::SizeOf {
            ty: map_type(node, ty),
            span,
        },
        Expr::AlignOf { ty, .. } => Expr::AlignOf {
            ty: map_type(node, ty),
            span,
        },
        Expr::OffsetOf { ty, field, .. } => Expr::OffsetOf {
            ty: map_type(node, ty),
            field: field.clone(),
            span,
        },
        Expr::Integer { value, .. } => Expr::Integer {
            value: *value,
            span,
        },
        Expr::Float { value, .. } => Expr::Float {
            value: *value,
            span,
        },
        Expr::Bool { value, .. } => Expr::Bool {
            value: *value,
            span,
        },
        Expr::Null { .. } => Expr::Null { span },
    }
}
fn map_type_name(node: &Node, name: &str) -> String {
    if name.contains('.') {
        let mut p = name.splitn(2, '.');
        return prefixed(
            node.aliases
                .get(p.next().unwrap())
                .map(String::as_str)
                .unwrap_or(""),
            p.next().unwrap(),
        );
    }
    if local_structs(node).contains(name) {
        prefixed(&node.prefix, name)
    } else {
        name.into()
    }
}
