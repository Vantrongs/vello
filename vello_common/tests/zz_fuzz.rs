use std::time::Instant;
use vello_common::fearless_simd::Level;
use vello_common::flatten::{FlattenCtx, fill};
use vello_common::geometry::RectU16;
use vello_common::kurbo::{Affine, BezPath};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn coord(&mut self, huge: bool) -> f64 {
        let s = if self.next() < 0.5 { -1.0 } else { 1.0 };
        if huge {
            let e = self.next() * 300.0;
            let m = if self.next() < 0.3 { 1.0 } else { self.next() * 9.0 + 1.0 };
            s * m * 10f64.powf(e.floor())
        } else {
            -20.0 + self.next() * 140.0
        }
    }
}

#[test]
fn fuzz() {
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let level = Level::new();
    let mut worst = (0.0f64, 0usize, String::new());
    let mut most = (0usize, String::new());
    for i in 0..200_000 {
        let mut p = BezPath::new();
        let p0 = (rng.coord(false), rng.coord(false));
        let mut pts = Vec::new();
        for _ in 0..3 {
            let huge = rng.next() < 0.8;
            pts.push((rng.coord(huge), rng.coord(huge)));
        }
        // Symmetric controls sometimes, for cancellation at t = 0.5.
        if i % 3 == 0 {
            pts[1] = (-pts[0].0, -pts[0].1);
        }
        let p3 = if i % 2 == 0 { (rng.coord(false), rng.coord(false)) } else { pts[2] };
        p.move_to(p0);
        p.curve_to(pts[0], pts[1], p3);
        p.close_path();
        let mut buf = Vec::new();
        let t = Instant::now();
        fill(level, p.iter(), Affine::IDENTITY, &mut buf, &mut FlattenCtx::default(),
            RectU16 { x0: 0, y0: 0, x1: 100, y1: 100 });
        let dt = t.elapsed().as_secs_f64();
        if dt > worst.0 { worst = (dt, i, format!("{p0:?} {pts:?} {p3:?}")); }
        if buf.len() > most.0 { most = (buf.len(), format!("{p0:?} {pts:?} {p3:?}")); }
    }
    println!("slowest {:.3} ms (case {}): {}", worst.0 * 1e3, worst.1, worst.2);
    println!("most lines {}: {}", most.0, most.1);
}
