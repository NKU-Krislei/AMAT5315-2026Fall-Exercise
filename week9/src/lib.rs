//! Probabilistic inference from local factor tables.

mod checks;
mod diagram;
mod infer;
mod model;
mod report;

pub use checks::{format_checks, matrix_and_ising, NumericChecks};
pub use diagram::ad_tuberculosis_diagram;
pub use infer::{contract, enumerate, Posterior};
pub use model::{Factor, Model};
pub use report::{compare, comparison_json, comparison_text, parse_evidence, Comparison};
