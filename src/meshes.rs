#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

#[derive(Clone, Copy)]
pub struct MeshRange {
    pub index_offset: u32,
    pub index_count: u32,
}

pub struct SceneMeshes {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
    pub unit_box: MeshRange,
    pub car_body: MeshRange,
    pub palm_trunk: MeshRange,
    pub palm_canopy: MeshRange,
    pub street_lamp: MeshRange,
}

fn push_quad(
    verts: &mut Vec<Vertex>,
    inds: &mut Vec<u16>,
    pts: [[f32; 3]; 4],
    normal: [f32; 3],
    uvs: [[f32; 2]; 4],
) {
    let base = verts.len() as u16;
    for i in 0..4 {
        verts.push(Vertex {
            pos: pts[i],
            normal,
            uv: uvs[i],
        });
    }
    inds.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn push_box(
    verts: &mut Vec<Vertex>,
    inds: &mut Vec<u16>,
    center: [f32; 3],
    size: [f32; 3],
    uv_rect: [f32; 4], // u0, v0, u1, v1
) {
    let hx = size[0] * 0.5;
    let hy = size[1] * 0.5;
    let hz = size[2] * 0.5;
    let [cx, cy, cz] = center;
    let [u0, v0, u1, v1] = uv_rect;
    let uvs = [[u0, v1], [u0, v0], [u1, v0], [u1, v1]];

    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        ([0.0, 0.0, 1.0], [[cx - hx, cy - hy, cz + hz], [cx - hx, cy + hy, cz + hz], [cx + hx, cy + hy, cz + hz], [cx + hx, cy - hy, cz + hz]]),
        ([0.0, 0.0, -1.0], [[cx + hx, cy - hy, cz - hz], [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz], [cx - hx, cy - hy, cz - hz]]),
        ([1.0, 0.0, 0.0], [[cx + hx, cy - hy, cz + hz], [cx + hx, cy + hy, cz + hz], [cx + hx, cy + hy, cz - hz], [cx + hx, cy - hy, cz - hz]]),
        ([-1.0, 0.0, 0.0], [[cx - hx, cy - hy, cz - hz], [cx - hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz + hz], [cx - hx, cy - hy, cz + hz]]),
        ([0.0, 1.0, 0.0], [[cx - hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz - hz], [cx + hx, cy + hy, cz - hz], [cx + hx, cy + hy, cz + hz]]),
        ([0.0, -1.0, 0.0], [[cx - hx, cy - hy, cz - hz], [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz - hz]]),
    ];
    for (n, pts) in faces {
        push_quad(verts, inds, pts, n, uvs);
    }
}

fn push_tapered_prism(
    verts: &mut Vec<Vertex>,
    inds: &mut Vec<u16>,
    z0: f32,
    z1: f32,
    y0_bot: f32,
    y0_top: f32,
    y1_bot: f32,
    y1_top: f32,
    x0: f32,
    x1: f32,
) {
    // 8 corners of a wedge / cabin section symmetric in X
    let uvs = [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
    // Top face
    push_quad(
        verts,
        inds,
        [[-x0, y0_top, z0], [-x1, y1_top, z1], [x1, y1_top, z1], [x0, y0_top, z0]],
        norm([0.0, z1 - z0, -(y1_top - y0_top)]),
        uvs,
    );
    // Bottom
    push_quad(
        verts,
        inds,
        [[-x1, y1_bot, z1], [-x0, y0_bot, z0], [x0, y0_bot, z0], [x1, y1_bot, z1]],
        [0.0, -1.0, 0.0],
        uvs,
    );
    // Front (+z1)
    push_quad(
        verts,
        inds,
        [[-x1, y1_bot, z1], [-x1, y1_top, z1], [x1, y1_top, z1], [x1, y1_bot, z1]],
        [0.0, 0.0, 1.0],
        [[0.0, 0.5], [0.0, 0.0], [0.5, 0.0], [0.5, 0.5]],
    );
    // Back (z0)
    push_quad(
        verts,
        inds,
        [[x0, y0_bot, z0], [x0, y0_top, z0], [-x0, y0_top, z0], [-x0, y0_bot, z0]],
        [0.0, 0.0, -1.0],
        [[0.5, 1.0], [0.5, 0.5], [1.0, 0.5], [1.0, 1.0]],
    );
    // Right (+X)
    push_quad(
        verts,
        inds,
        [[x0, y0_bot, z0], [x0, y0_top, z0], [x1, y1_top, z1], [x1, y1_bot, z1]],
        [1.0, 0.0, 0.0],
        uvs,
    );
    // Left (-X)
    push_quad(
        verts,
        inds,
        [[-x1, y1_bot, z1], [-x1, y1_top, z1], [-x0, y0_top, z0], [-x0, y0_bot, z0]],
        [-1.0, 0.0, 0.0],
        uvs,
    );
}

#[inline]
fn norm(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / l, v[1] / l, v[2] / l]
}

fn build_sports_car(verts: &mut Vec<Vertex>, inds: &mut Vec<u16>) {
    // Sleek 80s Vice-City Wedge Supercar (length ~4.4m along Z, width ~2.0m along X)
    // Main lower chassis (nose sloped forward +Z, tail at -Z)
    push_tapered_prism(verts, inds, -2.15, 0.55, 0.25, 0.85, 0.25, 0.82, 0.98, 0.96);
    // Sloped front hood (+Z)
    push_tapered_prism(verts, inds, 0.55, 2.15, 0.25, 0.82, 0.24, 0.56, 0.96, 0.88);
    // Aerodynamic cabin / windshield / rear fastback
    push_tapered_prism(verts, inds, -1.35, -0.55, 0.84, 1.28, 0.84, 1.32, 0.78, 0.76);
    push_tapered_prism(verts, inds, -0.55, 0.65, 0.84, 1.32, 0.82, 0.85, 0.76, 0.80);
    // Rear wing / spoiler
    push_box(verts, inds, [-0.68, 0.96, -1.95], [0.08, 0.26, 0.20], [0.0, 0.0, 0.2, 0.2]);
    push_box(verts, inds, [0.68, 0.96, -1.95], [0.08, 0.26, 0.20], [0.0, 0.0, 0.2, 0.2]);
    push_box(verts, inds, [0.0, 1.10, -1.98], [1.92, 0.07, 0.36], [0.0, 0.0, 1.0, 0.25]);
    // 4 wheels (dark lower quadrant of car texture)
    let w_uv = [0.75, 0.75, 0.98, 0.98];
    for &(wx, wz) in &[(-0.92f32, 1.32f32), (0.92, 1.32), (-0.94, -1.32), (0.94, -1.32)] {
        push_box(verts, inds, [wx, 0.32, wz], [0.28, 0.64, 0.64], w_uv);
    }
}

fn build_palm_trunk(verts: &mut Vec<Vertex>, inds: &mut Vec<u16>) {
    // 6-segment curved tapered trunk (~5.6m tall)
    let segs = 6usize;
    let sides = 8usize;
    let mut rings: Vec<Vec<[f32; 3]>> = Vec::with_capacity(segs + 1);
    for s in 0..=segs {
        let t = s as f32 / segs as f32;
        let y = t * 5.6;
        let lean_x = (t * t) * 0.85;
        let r = 0.28 * (1.0 - t * 0.52);
        let mut ring = Vec::with_capacity(sides);
        for k in 0..sides {
            let ang = (k as f32 / sides as f32) * std::f32::consts::TAU;
            ring.push([lean_x + ang.cos() * r, y, ang.sin() * r]);
        }
        rings.push(ring);
    }
    for s in 0..segs {
        let v0 = s as f32 / segs as f32;
        let v1 = (s + 1) as f32 / segs as f32;
        for k in 0..sides {
            let k2 = (k + 1) % sides;
            let u0 = k as f32 / sides as f32;
            let u1 = (k + 1) as f32 / sides as f32;
            let p00 = rings[s][k];
            let p01 = rings[s][k2];
            let p10 = rings[s + 1][k];
            let p11 = rings[s + 1][k2];
            let n = norm([
                (p00[0] + p01[0]) * 0.5 - (s as f32 / segs as f32).powi(2) * 0.85,
                0.15,
                (p00[2] + p01[2]) * 0.5,
            ]);
            push_quad(verts, inds, [p00, p10, p11, p01], n, [[u0, v0], [u0, v1], [u1, v1], [u1, v0]]);
        }
    }
}

fn build_palm_canopy(verts: &mut Vec<Vertex>, inds: &mut Vec<u16>) {
    // 7 arched 3D frond blades radiating from (0, 0, 0)
    let fronds = 7usize;
    let segs = 4usize;
    for f in 0..fronds {
        let ang = (f as f32 / fronds as f32) * std::f32::consts::TAU + (f as f32) * 0.17;
        let (sa, ca) = ang.sin_cos();
        let right = [-sa, 0.0, ca];
        let fwd = [ca, 0.0, sa];

        for s in 0..segs {
            let t0 = s as f32 / segs as f32;
            let t1 = (s + 1) as f32 / segs as f32;
            let r0 = t0 * 2.9;
            let r1 = t1 * 2.9;
            let y0 = (t0 * 1.4).sin() * 0.65 - t0 * t0 * 1.25;
            let y1 = (t1 * 1.4).sin() * 0.65 - t1 * t1 * 1.25;
            let w0 = (1.0 - t0 * 0.72) * 0.65;
            let w1 = (1.0 - t1 * 0.72) * 0.65;

            let p0l = [fwd[0] * r0 - right[0] * w0, y0 - 0.12 * t0, fwd[2] * r0 - right[2] * w0];
            let p0r = [fwd[0] * r0 + right[0] * w0, y0 - 0.12 * t0, fwd[2] * r0 + right[2] * w0];
            let p1l = [fwd[0] * r1 - right[0] * w1, y1 - 0.12 * t1, fwd[2] * r1 - right[2] * w1];
            let p1r = [fwd[0] * r1 + right[0] * w1, y1 - 0.12 * t1, fwd[2] * r1 + right[2] * w1];

            let uvs = [[0.0, t0], [0.0, t1], [1.0, t1], [1.0, t0]];
            // Top & bottom double-sided quads
            push_quad(verts, inds, [p0l, p1l, p1r, p0r], [0.0, 1.0, 0.0], uvs);
            push_quad(verts, inds, [p0r, p1r, p1l, p0l], [0.0, -1.0, 0.0], uvs);
        }
    }
}

fn build_street_lamp(verts: &mut Vec<Vertex>, inds: &mut Vec<u16>) {
    // Tall pole + curved arm extending +X + glowing lamp head
    push_box(verts, inds, [0.0, 3.4, 0.0], [0.18, 6.8, 0.18], [0.0, 0.0, 0.4, 1.0]);
    push_box(verts, inds, [0.95, 6.75, 0.0], [2.0, 0.14, 0.14], [0.0, 0.0, 0.4, 1.0]);
    push_box(verts, inds, [1.85, 6.58, 0.0], [0.45, 0.22, 0.35], [0.5, 0.0, 1.0, 0.5]);
}

pub fn build_all_meshes() -> SceneMeshes {
    let mut vertices = Vec::with_capacity(2048);
    let mut indices = Vec::with_capacity(4096);

    let r0 = indices.len() as u32;
    push_box(&mut vertices, &mut indices, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]);
    let unit_box = MeshRange { index_offset: r0, index_count: indices.len() as u32 - r0 };

    let r1 = indices.len() as u32;
    build_sports_car(&mut vertices, &mut indices);
    let car_body = MeshRange { index_offset: r1, index_count: indices.len() as u32 - r1 };

    let r2 = indices.len() as u32;
    build_palm_trunk(&mut vertices, &mut indices);
    let palm_trunk = MeshRange { index_offset: r2, index_count: indices.len() as u32 - r2 };

    let r3 = indices.len() as u32;
    build_palm_canopy(&mut vertices, &mut indices);
    let palm_canopy = MeshRange { index_offset: r3, index_count: indices.len() as u32 - r3 };

    let r4 = indices.len() as u32;
    build_street_lamp(&mut vertices, &mut indices);
    let street_lamp = MeshRange { index_offset: r4, index_count: indices.len() as u32 - r4 };

    SceneMeshes {
        vertices,
        indices,
        unit_box,
        car_body,
        palm_trunk,
        palm_canopy,
        street_lamp,
    }
}
