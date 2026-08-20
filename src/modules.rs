//! Deterministic source-module loading and module identity resolution.
//!
//! The graph is resolved by canonical path and every module/declaration gets a
//! compiler-owned identity.  The legacy `Program` returned by this module is a
//! compatibility view for the current frontend; its private names are derived
//! from those identities rather than from file-stem prefix rewriting.  The
//! graph and its declaration metadata are the authoritative representation.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::ast::*;
use crate::lexer::Span;
use crate::pipeline;
use crate::semantic;
use crate::typed::{DefId, DefinitionKind, ModuleId, TypedProgram};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleDefinition {
    pub id: DefId,
    pub module: ModuleId,
    /// Name as written in the declaring source file.
    pub source_name: String,
    /// Backend/linker spelling. This is metadata, not a name used for lookup.
    pub linker_name: String,
    pub kind: DefinitionKind,
    pub exported: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleImport {
    pub alias: String,
    pub target: ModuleId,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleInfo {
    pub id: ModuleId,
    pub path: PathBuf,
    pub imports: Vec<ModuleImport>,
    pub declarations: Vec<DefId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleGraph {
    pub root: ModuleId,
    pub modules: Vec<ModuleInfo>,
    pub definitions: Vec<ModuleDefinition>,
}

impl ModuleGraph {
    /// Verify that the typed frontend consumed the graph's declaration
    /// identities instead of rebuilding a name-based symbol order.
    pub fn validate_typed(&self, typed: &TypedProgram) -> Result<(), String> {
        if typed.symbols.definitions.len() < self.definitions.len() {
            return Err(format!(
                "resolved graph contains {} declarations but typed frontend produced {}",
                self.definitions.len(),
                typed.symbols.definitions.len()
            ));
        }
        for definition in &self.definitions {
            // Generic declarations are replaced by concrete declarations
            // before typed lowering, and imported declarations acquire a
            // backend spelling at the AST boundary. Match those two explicit
            // boundary cases by graph metadata; all later semantic references
            // still use the typed DefId.
            let symbol = typed
                .symbols
                .find(&definition.linker_name)
                .or_else(|| typed.symbols.find(&definition.source_name))
                .or_else(|| {
                    typed.symbols.definitions.iter().find(|symbol| {
                        symbol
                            .name
                            .starts_with(&format!("{}__", definition.source_name))
                            || symbol
                                .name
                                .ends_with(&format!("_{}", definition.source_name))
                    })
                })
                .ok_or_else(|| {
                    format!(
                        "typed frontend lost declaration {} (`{}` / `{}`)",
                        definition.id, definition.source_name, definition.linker_name
                    )
                })?;
            let expected = definition.kind.clone();
            if symbol.kind != expected {
                return Err(format!(
                    "typed declaration {} has the wrong kind",
                    definition.id
                ));
            }
        }
        Ok(())
    }

    pub fn module(&self, id: ModuleId) -> Option<&ModuleInfo> {
        self.modules
            .get(id.index())
            .filter(|module| module.id == id)
    }

    pub fn definition(&self, id: DefId) -> Option<&ModuleDefinition> {
        self.definitions
            .iter()
            .find(|definition| definition.id == id)
    }

    pub fn lookup(&self, module: ModuleId, name: &str) -> Option<&ModuleDefinition> {
        let module = self.module(module)?;
        module
            .declarations
            .iter()
            .filter_map(|id| self.definition(*id))
            .find(|definition| definition.source_name == name)
    }

    pub fn import(&self, module: ModuleId, alias: &str) -> Option<ModuleId> {
        self.module(module)?
            .imports
            .iter()
            .find(|import| import.alias == alias)
            .map(|import| import.target)
    }

    pub fn lookup_qualified(
        &self,
        module: ModuleId,
        alias: &str,
        name: &str,
    ) -> Option<&ModuleDefinition> {
        let target = self.import(module, alias)?;
        self.lookup(target, name)
    }

    /// Resolve a qualified source spelling without deriving a linker name.
    /// Backends should carry the returned ID and consult `definition` only
    /// when they need presentation or linkage metadata.
    pub fn resolve_qualified(&self, module: ModuleId, alias: &str, name: &str) -> Option<DefId> {
        self.lookup_qualified(module, alias, name)
            .map(|definition| definition.id)
    }

    pub fn resolve_local(&self, module: ModuleId, name: &str) -> Option<DefId> {
        self.lookup(module, name).map(|definition| definition.id)
    }
}

#[derive(Debug, Clone)]
pub struct Project {
    /// Compatibility AST consumed by the pre-MIR frontend. Resolution and
    /// identity information lives in `graph`; this field is not the module
    /// graph itself.
    pub program: Program,
    pub graph: ModuleGraph,
    pub root_source: String,
    pub root_path: PathBuf,
    /// Canonical source files in deterministic dependency order.
    pub dependencies: Vec<PathBuf>,
}

impl Project {
    /// Analyze the resolved project through its canonical graph entry point.
    /// The compatibility AST is an implementation boundary; callers no
    /// longer need to know how module source is assembled.
    pub fn analyze(&self, pointer_width: u32) -> Result<TypedProgram, pipeline::FrontendError> {
        pipeline::analyze_resolved_program(&self.program, &self.graph, pointer_width)
    }

    pub fn analyze_native(&self) -> Result<TypedProgram, pipeline::FrontendError> {
        self.analyze_native_with_pointer_width(usize::BITS)
    }

    pub fn analyze_native_with_pointer_width(
        &self,
        pointer_width: u32,
    ) -> Result<TypedProgram, pipeline::FrontendError> {
        let typed = self.analyze(pointer_width)?;
        semantic::validate_typed_entry_point(&typed).map_err(pipeline::FrontendError::Semantic)?;
        Ok(typed)
    }
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
    is_root: bool,
    /// Imports point at graph nodes, never at a derived string prefix.
    aliases: HashMap<String, usize>,
    visible_exports: HashMap<String, HashSet<String>>,
    /// Names explicitly exported by this module.
    exported: HashSet<String>,
    module_id: ModuleId,
    declaration_ids: HashMap<String, DefId>,
    /// Temporary compatibility spellings for the old AST frontend. All of
    /// these are derived from `module_id`/`DefId`; lookup uses graph identity.
    linker_names: HashMap<String, String>,
    alias_linker_names: HashMap<String, HashMap<String, String>>,
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
    let root_index = visit(
        &root,
        true,
        module_roots,
        &mut nodes,
        &mut indexes,
        &mut visiting,
        Span::new(0, 0),
    )?;

    // Assign identities only after traversal. Post-order traversal makes the
    // IDs deterministic and ensures dependencies precede their users.
    let mut next_def = 0u32;
    for (module_index, node) in nodes.iter_mut().enumerate() {
        node.module_id = ModuleId(module_index as u32);
        for declaration in &node.program.declarations {
            let Some(name) = declaration_name(declaration) else {
                continue;
            };
            let id = DefId(next_def);
            next_def += 1;
            node.declaration_ids.insert(name.to_string(), id);
            let linker_name = if node.is_root {
                name.to_string()
            } else {
                format!("__module_{}_{}", node.module_id.0, name)
            };
            node.linker_names.insert(name.to_string(), linker_name);
        }
    }
    for node_index in 0..nodes.len() {
        let aliases = nodes[node_index].aliases.clone();
        for (alias, target_index) in aliases {
            let target_names = nodes[target_index].linker_names.clone();
            nodes[node_index]
                .alias_linker_names
                .insert(alias, target_names);
        }
    }

    for node in &nodes {
        for declaration in &node.program.declarations {
            validate_visibility(declaration, node)?;
        }
    }
    let mut declarations = Vec::new();
    // The AST frontend still needs one declaration list. This is a boundary
    // adapter only: names are generated from graph identities and all source
    // names/linker names remain available in `graph`.
    for node in &nodes {
        for declaration in &node.program.declarations {
            declarations.push(rewrite_decl(declaration, node));
        }
    }
    let definitions = nodes
        .iter()
        .flat_map(|node| {
            node.program.declarations.iter().filter_map(|declaration| {
                let name = declaration_name(declaration)?;
                let id = node.declaration_ids[name];
                let (kind, exported, explicit_linker) = match declaration {
                    Decl::Function(f) => {
                        (DefinitionKind::Function, f.exported, f.link_name.clone())
                    }
                    Decl::Struct(s) => (DefinitionKind::Struct, s.exported, None),
                    Decl::Variable(v) => (
                        if matches!(v.kind, VariableKind::Immutable) {
                            DefinitionKind::Constant
                        } else {
                            DefinitionKind::Global
                        },
                        v.exported,
                        None,
                    ),
                    Decl::Comptime { .. } => return None,
                };
                Some(ModuleDefinition {
                    id,
                    module: node.module_id,
                    source_name: name.to_string(),
                    linker_name: explicit_linker.unwrap_or_else(|| node.linker_names[name].clone()),
                    kind,
                    exported,
                })
            })
        })
        .collect();
    let modules = nodes
        .iter()
        .map(|node| ModuleInfo {
            id: node.module_id,
            path: node.path.clone(),
            imports: node
                .program
                .imports
                .iter()
                .filter_map(|import| {
                    Some(ModuleImport {
                        alias: import.alias.clone(),
                        target: ModuleId(*node.aliases.get(&import.alias)? as u32),
                        span: import.span,
                    })
                })
                .collect(),
            declarations: node
                .program
                .declarations
                .iter()
                .filter_map(|declaration| declaration_name(declaration))
                .filter_map(|name| node.declaration_ids.get(name).copied())
                .collect(),
        })
        .collect();
    Ok(Project {
        program: Program {
            imports: Vec::new(),
            declarations,
        },
        graph: ModuleGraph {
            root: ModuleId(root_index as u32),
            modules,
            definitions,
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
        let imported_exports = nodes[imported_index].exported.clone();
        aliases.insert(import.alias.clone(), imported_index);
        visible_exports.insert(import.alias.clone(), imported_exports);
    }
    visiting.pop();
    let index = nodes.len();
    indexes.insert(path.clone(), index);
    nodes.push(Node {
        path,
        program,
        is_root,
        aliases,
        visible_exports,
        exported,
        module_id: ModuleId(u32::MAX),
        declaration_ids: HashMap::new(),
        linker_names: HashMap::new(),
        alias_linker_names: HashMap::new(),
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
            let (alias, item) = name.split_once('.').unwrap();

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
            if let Expr::Identifier { name: alias, .. } = base.as_ref()
                && node.aliases.contains_key(alias)
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
            validate_expr_visibility(base, node)?;
        }
        Expr::StructLiteral { name, fields, span } => {
            if name.contains('.') {
                let (alias, item) = name.split_once('.').unwrap();

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
        | Expr::String { .. }
        | Expr::Bool { .. }
        | Expr::Null { .. }
        | Expr::Identifier { .. } => {}
    }
    Ok(())
}

fn declaration_name(decl: &Decl) -> Option<&str> {
    match decl {
        Decl::Function(f) => Some(&f.name),
        Decl::Struct(s) => Some(&s.name),
        Decl::Variable(v) => Some(&v.name),
        Decl::Comptime { .. } => None,
    }
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
    node.linker_names
        .get(name)
        .cloned()
        .unwrap_or_else(|| name.into())
}
fn map_type(node: &Node, ty: &Type) -> Type {
    match ty {
        Type::Named(n) if local_structs(node).contains(n) => Type::Named(
            node.linker_names
                .get(n)
                .cloned()
                .unwrap_or_else(|| n.clone()),
        ),
        Type::Named(n) if n.contains('.') => {
            let (alias, name) = n.split_once('.').unwrap();

            let visible = node
                .visible_exports
                .get(alias)
                .map(|names| names.contains(name))
                .unwrap_or(false);
            if !visible {
                // Visibility has already been checked above. Keep the source
                // spelling here instead of manufacturing a sentinel name.
                Type::Named(format!("{alias}.{name}"))
            } else {
                Type::Named(
                    node.alias_linker_names
                        .get(alias)
                        .and_then(|names| names.get(name))
                        .cloned()
                        .unwrap_or_else(|| format!("{alias}.{name}")),
                )
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
            name: node.linker_names[&s.name].clone(),
            generic_params: s.generic_params.clone(),
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
                name: node.linker_names[&v.name].clone(),
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
                name: node.linker_names[&f.name].clone(),
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
            if let Expr::Identifier { name: alias, .. } = base.as_ref()
                && node.aliases.contains_key(alias)
            {
                let visible = node
                    .visible_exports
                    .get(alias)
                    .map(|names| names.contains(name))
                    .unwrap_or(false);
                let name = if visible {
                    node.alias_linker_names
                        .get(alias)
                        .and_then(|names| names.get(name))
                        .cloned()
                        .unwrap_or_else(|| format!("{alias}.{name}"))
                } else {
                    // `validate_visibility` normally rejects this path.
                    // Preserve the source identity if it is reached rather
                    // than manufacturing a private-name sentinel.
                    format!("{alias}.{name}")
                };
                return Expr::Identifier { name, span };
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
        Expr::String { value, .. } => Expr::String {
            value: value.clone(),
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
    if let Some((alias, item)) = name.split_once('.') {
        return node
            .alias_linker_names
            .get(alias)
            .and_then(|names| names.get(item))
            .cloned()
            .unwrap_or_else(|| name.into());
    }
    if local_structs(node).contains(name) {
        node.linker_names
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.into())
    } else {
        name.into()
    }
}
