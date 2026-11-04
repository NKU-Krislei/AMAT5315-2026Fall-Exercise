/// Column-major `AB`, `tr(AB)`, and the three-spin Ising partition function.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericChecks {
    pub ab_col_major: Vec<f64>,
    pub trace: f64,
    pub ising_contraction: f64,
    pub ising_enumeration: f64,
    pub ab_error: f64,
    pub trace_error: f64,
    pub ising_error: f64,
}

pub fn matrix_and_ising() -> NumericChecks {
    use tenferro_cpu::CpuEngine;
    use tenferro_einsum::TensorEinsumExt;
    use tenferro_tensor::Tensor;

    let mut engine = CpuEngine::serial();
    let a = Tensor::from_vec_row_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("A");
    let b = Tensor::from_vec_row_major(vec![2, 2], vec![2.0, 0.0, 1.0, 2.0]).expect("B");
    let product = [&a, &b].einsum("ij,jk->ik", &mut engine).expect("AB");
    let ab_col_major = product.as_slice::<f64>().expect("AB entries").to_vec();
    let expected_ab = [4.0, 10.0, 4.0, 8.0];
    let ab_error = ab_col_major
        .iter()
        .zip(expected_ab)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);

    let traced = [&a, &b].einsum("ij,ji->", &mut engine).expect("tr(AB)");
    let trace = traced.as_slice::<f64>().expect("trace")[0];
    let trace_error = (trace - 12.0).abs();

    let beta_j: f64 = 0.5;
    let spins = [-1.0, 1.0];
    let mut bond = Vec::with_capacity(4);
    for &left in &spins {
        for &right in &spins {
            bond.push((beta_j * left * right).exp());
        }
    }
    let k1 = Tensor::from_vec_row_major(vec![2, 2], bond.clone()).expect("bond");
    let k2 = Tensor::from_vec_row_major(vec![2, 2], bond.clone()).expect("bond");
    let k3 = Tensor::from_vec_row_major(vec![2, 2], bond).expect("bond");
    let contracted = [&k1, &k2, &k3]
        .einsum("ij,jk,ki->", &mut engine)
        .expect("Ising ring");
    let ising_contraction = contracted.as_slice::<f64>().expect("Z")[0];

    let mut ising_enumeration = 0.0;
    for &s1 in &spins {
        for &s2 in &spins {
            for &s3 in &spins {
                let energy = -(s1 * s2 + s2 * s3 + s3 * s1);
                ising_enumeration += (-beta_j * energy).exp();
            }
        }
    }
    let ising_error = (ising_contraction - ising_enumeration).abs();

    NumericChecks {
        ab_col_major,
        trace,
        ising_contraction,
        ising_enumeration,
        ab_error,
        trace_error,
        ising_error,
    }
}

pub fn format_checks(checks: &NumericChecks) -> String {
    format!(
        "AB column-major = {:?}\nabsolute error = {:.3e}\ntr(AB) = {:.17}\nabsolute error = {:.3e}\nIsing Z contraction = {:.17}\nIsing Z enumeration = {:.17}\nabsolute error = {:.3e}\n",
        checks.ab_col_major,
        checks.ab_error,
        checks.trace,
        checks.trace_error,
        checks.ising_contraction,
        checks.ising_enumeration,
        checks.ising_error
    )
}
