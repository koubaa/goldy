//! Restricted GPU-dialect statement / expression IR.
//!
//! Frontends must lower through this IR rather than performing token or regex
//! substitution on source text.

/// Binary operators supported in the MVP dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

/// Unary operators supported in the MVP dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
}

/// Built-in calls recognized by the frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinFn {
    GlobalId,
    LocalId,
    WorkgroupId,
    WorkgroupSize,
    /// Selected math intrinsics (mapped to Slang free functions).
    Abs,
    Min,
    Max,
    Floor,
    Ceil,
    Sqrt,
}

/// Expression nodes.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    LitU32(u32),
    LitI32(i32),
    LitF32(f32),
    LitBool(bool),
    Var(String),
    Field {
        base: Box<Expr>,
        field: String,
    },
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
    },
    /// Buffer / slice `.len()` → Slang `.Length`.
    Len {
        base: Box<Expr>,
    },
    Binary {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    Call {
        func: BuiltinFn,
        args: Vec<Expr>,
    },
    /// `as` cast / numeric cast to a Slang type name.
    Cast {
        expr: Box<Expr>,
        ty: String,
    },
}

/// Statement nodes.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Let {
        name: String,
        mutable: bool,
        ty: Option<String>,
        init: Expr,
    },
    Assign {
        target: Expr,
        value: Expr,
    },
    If {
        cond: Expr,
        then_body: Vec<Stmt>,
        else_body: Option<Vec<Stmt>>,
    },
    While {
        cond: Expr,
        body: Vec<Stmt>,
    },
    /// `for i in start..end` (exclusive end).
    ForRange {
        var: String,
        start: Expr,
        end: Expr,
        body: Vec<Stmt>,
    },
    Return {
        value: Option<Expr>,
    },
    Expr(Expr),
}

/// A lowered compute kernel body plus signature metadata used for emission.
#[derive(Debug, Clone, PartialEq)]
pub struct ShaderKernel {
    pub name: String,
    pub workgroup_size: [u32; 3],
    pub params: Vec<crate::KernelParam>,
    pub builtins: crate::BuiltinMask,
    pub body: Vec<Stmt>,
    pub source_map: crate::SourceMap,
}
