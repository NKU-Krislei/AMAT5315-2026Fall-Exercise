use crate::Model;

const NAMES: [&str; 8] = ["A", "T", "S", "L", "B", "E", "X", "D"];

/// AD tuberculosis query: observations `A=0`, `D=0`, and only `T` left open.
pub fn ad_tuberculosis_diagram(model: &Model) -> String {
    let mut factors = String::new();
    for factor in &model.factors {
        let axes = factor
            .scope
            .iter()
            .map(|var| var.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let (x, y, title) = place(&factor.scope);
        let axis_names = factor
            .scope
            .iter()
            .map(|&var| NAMES[var])
            .collect::<Vec<_>>()
            .join(", ");
        let tx = x + 12;
        let ty = y + 30;
        let ay = y + 52;
        factors.push_str(&format!(
            r##"<g data-factor="{title}" data-axes="{axes}"><rect x="{x}" y="{y}" width="168" height="72" rx="8" fill="#f7f4ea" stroke="#1f2933" stroke-width="1.5"/><text x="{tx}" y="{ty}" font-family="Georgia, serif" font-size="18" fill="#1f2933">{title}</text><text x="{tx}" y="{ay}" font-family="ui-monospace, monospace" font-size="13" fill="#52606d">axes {axis_names}</text></g>"##
        ));
    }

    format!(
        r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="720" viewBox="0 0 1080 720">
  <rect width="1080" height="720" fill="#fffdf8"/>
  <text x="40" y="42" font-family="Georgia, serif" font-size="22" fill="#1f2933">AD tuberculosis query</text>
  <text x="40" y="66" font-family="ui-monospace, monospace" font-size="13" fill="#52606d">fix A=0 and D=0, sum every other index, leave T open</text>
  <line data-wire="A" x1="208" y1="146" x2="292" y2="146" stroke="#1d4e89" stroke-width="3"/>
  <circle data-wire="S" cx="208" cy="322" r="6" fill="#2f7d4a"/>
  <circle data-wire="S" cx="250" cy="322" r="6" fill="#2f7d4a"/>
  <circle data-wire="S" cx="124" cy="466" r="6" fill="#2f7d4a"/>
  <path data-wire="S" d="M208 322 H250 M208 322 V466 H124" fill="none" stroke="#2f7d4a" stroke-width="3"/>
  <path data-wire="T" d="M376 110 V36 H626 V196" fill="none" stroke="#b42318" stroke-width="3"/>
  <path data-wire="L" d="M418 286 H480 V232 H542" fill="none" stroke="#7a3e9d" stroke-width="3"/>
  <path data-wire="B" d="M208 502 H542" fill="none" stroke="#b86e00" stroke-width="3"/>
  <path data-wire="E" d="M710 232 H748 V146 H760 M748 232 V502 H710" fill="none" stroke="#3d4c5c" stroke-width="3"/>
  <line data-wire="X" x1="928" y1="146" x2="968" y2="146" stroke="#52606d" stroke-width="3"/>
  <line data-wire="D" x1="626" y1="538" x2="626" y2="590" stroke="#0f6f8a" stroke-width="3"/>
  {factors}
  <g data-open="T"><circle cx="376" cy="36" r="7" fill="#b42318"/><text x="392" y="28" font-family="ui-monospace, monospace" font-size="14" fill="#b42318">open T</text></g>
  <g data-fixed="A=0"><rect x="214" y="124" width="70" height="22" rx="4" fill="#1d4e89"/><text x="222" y="140" font-family="ui-monospace, monospace" font-size="12" fill="#fffdf8">(1, 0) A</text></g>
  <g data-fixed="D=0"><rect x="592" y="594" width="78" height="22" rx="4" fill="#0f6f8a"/><text x="600" y="610" font-family="ui-monospace, monospace" font-size="12" fill="#fffdf8">(1, 0) D</text></g>
  <g data-sum="X"><rect x="972" y="134" width="78" height="22" rx="4" fill="#52606d"/><text x="980" y="150" font-family="ui-monospace, monospace" font-size="12" fill="#fffdf8">(1, 1) X</text></g>
</svg>
"##
    )
}

fn place(scope: &[usize]) -> (i32, i32, &'static str) {
    match scope {
        [0] => (40, 110, "p(A)"),
        [1, 0] => (292, 110, "p(T|A)"),
        [2] => (40, 286, "p(S)"),
        [3, 2] => (250, 286, "p(L|S)"),
        [4, 2] => (40, 466, "p(B|S)"),
        [5, 3, 1] => (542, 196, "p(E|T,L)"),
        [6, 5] => (760, 110, "p(X|E)"),
        [7, 5, 4] => (542, 466, "p(D|E,B)"),
        other => panic!("no diagram position for scope {other:?}"),
    }
}
