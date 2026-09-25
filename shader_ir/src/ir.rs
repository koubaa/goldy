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
    Sin,
    Cos,
    Exp,
    Log,
    Pow,
    Length,
    Float2,
    Float3,
    Float4,
    Uint2,
    WorkgroupBarrier,
    /// The calling thread's lane in its subgroup, as `uint`.
    SubgroupLane,
    /// `(value, lane)`: `value` as subgroup lane `lane` holds it. Convergent: every
    /// lane of the subgroup must execute the call.
    SubgroupRead,
}

/// Rows, columns and summed extent of every [`Stmt::Matrix`] tile.
pub const MATRIX_TILE: u32 = 16;

/// Subgroup-scope matrix-unit operations on [`MATRIX_TILE`]-square tiles.
///
/// Operand tiles are row-major `half` workgroup arrays and accumulators hold `float`.
/// Every lane of the subgroup must execute each operation (convergent).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatrixOp {
    /// Declare the local accumulator `name`, zeroed.
    Accumulator { name: String },
    /// `acc += a · b`.
    MulAdd { acc: String, a: String, b: String },
    /// Store `acc` row-major into the `float` workgroup array `dest`.
    Store { acc: String, dest: String },
}

/// Tree-reduce operator for [`Stmt::WorkgroupReduce`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkgroupReduceOp {
    Sum,
    Max,
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
    /// Buffer / slice `.len()` → Slang `goldy_buf_len`, or a tensor's logical `numel`.
    Len {
        base: Box<Expr>,
    },
    /// Tensor `.dim(axis)` → checked logical extent.
    Dim {
        base: Box<Expr>,
        axis: Box<Expr>,
    },
    /// Tensor `.rank()` → packed layout rank (0..=4).
    Rank {
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
    /// `let mut scratch = gpu::workgroup_array::<T, N>()` → file-scope `groupshared`.
    WorkgroupArray {
        name: String,
        elem: String,
        len: u32,
    },
    /// Tree-reduce `val` across `n` lanes; every lane receives the result in `dest`.
    ///
    /// `n` must be a power of two and the workgroup `[n, 1, 1]`. Emitted Slang includes
    /// a trailing barrier, so `dest` is immediately readable. All workgroup threads must
    /// execute this statement (convergent).
    ///
    /// The association is the pairwise tree over adjacent local ids: `T(a, b) =
    /// T(a, m) ⊕ T(m, b)` with `m` the midpoint and the lower half on the left, on every
    /// target and for every lowering.
    ///
    /// [`SUBGROUP_WIDTH_DEFINE`](crate::SUBGROUP_WIDTH_DEFINE) selects the lowering
    /// with subgroup reads when the width `w` satisfies `w ≤ n ≤ w²`.
    WorkgroupReduce {
        op: WorkgroupReduceOp,
        n: u32,
        val: Expr,
        scratch: String,
        dest: Expr,
    },
    /// In-place softmax over `buf[base .. base+count)`. Trailing barrier.
    ///
    /// Unused lanes contribute identity (`-1e30` for max, `0` for sum). `count`
    /// must be greater than zero. All workgroup threads must execute this
    /// statement (convergent).
    WorkgroupSoftmax {
        n: u32,
        buf: String,
        base: Expr,
        count: Expr,
        scratch: String,
    },
    Matrix(MatrixOp),
    Expr(Expr),
}

/// Structured definition of a virtual compute entry.
///
/// Retained beside the canonical source so the entry can be lowered on its own or
/// composed with other definitions before physical entry-point generation.
#[derive(Debug, Clone, PartialEq)]
pub struct ShaderKernel {
    pub name: String,
    pub workgroup_size: [u32; 3],
    pub params: Vec<crate::KernelParam>,
    pub builtins: crate::BuiltinMask,
    pub body: Vec<Stmt>,
    pub source_map: crate::SourceMap,
    /// Slang declarations of `#[goldy::gpu]` types named by `params`, emitted ahead
    /// of the import. Resolved at prepare time, so empty in proc-macro output.
    pub type_decls: Vec<String>,
}
