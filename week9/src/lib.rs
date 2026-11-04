//! Probabilistic inference from local factor tables.

mod checks;
mod diagram;
mod infer;
mod model;
mod order;
mod report;

pub use checks::{format_checks, matrix_and_ising, NumericChecks};
pub use diagram::ad_tuberculosis_diagram;
pub use infer::{contract, enumerate, Posterior};
pub use model::{Factor, Model};
pub use order::{
    execute_scalar, query_inputs, recount, recount_plan, scalar_plan, Complexity, OrderChoice,
    PlanTree, ScalarPlan,
};
pub use report::{compare, comparison_json, comparison_text, parse_evidence, Comparison};
