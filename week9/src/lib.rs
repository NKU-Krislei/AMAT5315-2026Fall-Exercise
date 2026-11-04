//! Probabilistic inference from local factor tables.

mod bp;
mod checks;
mod diagram;
mod infer;
mod model;
mod order;
mod report;

pub use bp::{belief_propagation, BpResult, BpSettings, BpSweep};
pub use checks::{format_checks, matrix_and_ising, NumericChecks};
pub use diagram::ad_tuberculosis_diagram;
pub use infer::{contract, enumerate, Posterior};
pub use model::{read_evidence_file, read_log10_partition, read_marginals, Factor, Model};
pub use order::{
    conditioned_plan, execute_conditioned, execute_scalar, log10_partition, normalize_open,
    query_inputs, recount, recount_plan, scalar_plan, treesa_settings, Complexity, OrderChoice,
    PlanTree, ScalarPlan, ScaledContraction, TreeSaSettings,
};
pub use report::{compare, comparison_json, comparison_text, parse_evidence, Comparison};
