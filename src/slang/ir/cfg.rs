//! Control-flow graph of one Slang IR function: blocks, edges, block-parameter incomings
//! (Slang's phis) and the dominator tree.
//!
//! Slang keeps control flow structured (`ifElse`, `loop`, `switch`) but every terminator
//! still names its successor blocks explicitly, so a conventional CFG falls out of the
//! terminators. Block parameters receive their values from the operands of the branches
//! that target the block (`unconditionalBranch` and `loop` carry arguments; the others do
//! not, their targets have no parameters).

use std::collections::HashMap;

use super::module::{op, Module};

pub(crate) struct Block {
    pub params: Vec<u32>,
    /// Non-parameter instructions in order; the last one is the terminator.
    pub body: Vec<u32>,
    pub preds: Vec<usize>,
    pub succs: Vec<usize>,
}

impl Block {
    pub fn terminator(&self) -> Option<u32> {
        self.body.last().copied()
    }
}

pub(crate) struct Cfg {
    /// Blocks in module order; block 0 is the entry block.
    pub blocks: Vec<Block>,
    /// Block instruction id -> index into `blocks`.
    pub block_of: HashMap<u32, usize>,
    /// Block index of each instruction defined in the function (params included).
    pub inst_block: HashMap<u32, usize>,
    /// Incoming `(pred block, value)` pairs for every non-entry block parameter.
    pub phis: HashMap<u32, Vec<(usize, u32)>>,
    /// Immediate dominator of each block (`None` for the entry and for unreachable blocks).
    pub idom: Vec<Option<usize>>,
    /// Reachable blocks in reverse post-order.
    pub rpo: Vec<usize>,
}

impl Cfg {
    pub fn build(m: &Module, func: u32) -> Cfg {
        let mut blocks: Vec<Block> = Vec::new();
        let mut block_of = HashMap::new();
        for b in m.body(func).filter(|&c| m.inst(c).op == op::BLOCK) {
            let mut params = Vec::new();
            let mut body = Vec::new();
            for c in m.body(b) {
                if m.inst(c).op == op::PARAM {
                    params.push(c);
                } else {
                    body.push(c);
                }
            }
            block_of.insert(b, blocks.len());
            blocks.push(Block {
                params,
                body,
                preds: Vec::new(),
                succs: Vec::new(),
            });
        }
        let mut inst_block = HashMap::new();
        for (bi, b) in blocks.iter().enumerate() {
            for &p in b.params.iter().chain(b.body.iter()) {
                inst_block.insert(p, bi);
            }
        }
        let mut phis: HashMap<u32, Vec<(usize, u32)>> = HashMap::new();
        let mut edges: Vec<(usize, usize, Vec<u32>)> = Vec::new();
        for (bi, b) in blocks.iter().enumerate() {
            let Some(term) = b.terminator() else { continue };
            let t = m.inst(term);
            let target = |k: usize| t.operand(k).and_then(|x| block_of.get(&x).copied());
            match t.op {
                op::UNCONDITIONAL_BRANCH => {
                    if let Some(s) = target(0) {
                        edges.push((bi, s, t.operands[1..].iter().flatten().copied().collect()));
                    }
                }
                op::LOOP => {
                    if let Some(s) = target(0) {
                        edges.push((
                            bi,
                            s,
                            t.operands[3.min(t.operands.len())..]
                                .iter()
                                .flatten()
                                .copied()
                                .collect(),
                        ));
                    }
                }
                op::CONDITIONAL_BRANCH | op::IF_ELSE => {
                    for k in 1..=2 {
                        if let Some(s) = target(k) {
                            edges.push((bi, s, Vec::new()));
                        }
                    }
                }
                op::SWITCH => {
                    if let Some(s) = target(2) {
                        edges.push((bi, s, Vec::new()));
                    }
                    let mut k = 4;
                    while k < t.operands.len() {
                        if let Some(s) = target(k) {
                            edges.push((bi, s, Vec::new()));
                        }
                        k += 2;
                    }
                }
                _ => {}
            }
        }
        for (from, to, args) in edges {
            if !blocks[from].succs.contains(&to) {
                blocks[from].succs.push(to);
            }
            blocks[to].preds.push(from);
            for (k, &p) in blocks[to].params.iter().enumerate() {
                if let Some(&a) = args.get(k) {
                    phis.entry(p).or_default().push((from, a));
                }
            }
        }
        let (idom, rpo) = dominators(&blocks);
        Cfg {
            blocks,
            block_of,
            inst_block,
            phis,
            idom,
            rpo,
        }
    }

    /// Parameters of the entry block, positionally: the function's parameters.
    pub fn params(&self) -> Vec<u32> {
        self.blocks.first().map(|b| b.params.clone()).unwrap_or_default()
    }
}

// ============================================================================
// Dominators (Cooper, Harvey & Kennedy)
// ============================================================================

fn dominators(blocks: &[Block]) -> (Vec<Option<usize>>, Vec<usize>) {
    let n = blocks.len();
    let mut idom: Vec<Option<usize>> = vec![None; n];
    if n == 0 {
        return (idom, Vec::new());
    }
    let mut post: Vec<usize> = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
    visited[0] = true;
    while let Some(top) = stack.last_mut() {
        let b = top.0;
        let succs = &blocks[b].succs;
        if top.1 < succs.len() {
            let s = succs[top.1];
            top.1 += 1;
            if !visited[s] {
                visited[s] = true;
                stack.push((s, 0));
            }
        } else {
            post.push(b);
            stack.pop();
        }
    }
    let rpo: Vec<usize> = post.iter().rev().copied().collect();
    let mut rpo_index = vec![usize::MAX; n];
    for (i, &b) in rpo.iter().enumerate() {
        rpo_index[b] = i;
    }
    idom[0] = Some(0);
    let mut changed = true;
    while changed {
        changed = false;
        for &b in rpo.iter().skip(1) {
            let mut new_idom: Option<usize> = None;
            for &p in &blocks[b].preds {
                if idom[p].is_none() {
                    continue;
                }
                new_idom = Some(match new_idom {
                    None => p,
                    Some(cur) => intersect(&idom, &rpo_index, p, cur),
                });
            }
            if new_idom.is_some() && idom[b] != new_idom {
                idom[b] = new_idom;
                changed = true;
            }
        }
    }
    idom[0] = None;
    (idom, rpo)
}

fn intersect(idom: &[Option<usize>], rpo_index: &[usize], mut a: usize, mut b: usize) -> usize {
    while a != b {
        while rpo_index[a] > rpo_index[b] {
            a = idom[a].unwrap_or(0);
        }
        while rpo_index[b] > rpo_index[a] {
            b = idom[b].unwrap_or(0);
        }
    }
    a
}
