// resizes a texture into a render target

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;

struct VertexOutput
{
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f
}

@vertex
fn vsMain(@builtin(vertex_index) vertexIdx: u32) -> VertexOutput
{
    var positions = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
    let p = positions[vertexIdx];

    var out: VertexOutput;
    out.pos = vec4f(p, 0, 1);
    out.uv = vec2f(p.x * 0.5f + 0.5f, 0.5f - p.y * 0.5f);
    return out;
}

@fragment
fn fsMain(in: VertexOutput) -> @location(0) vec4f
{
    return textureSampleLevel(srcTex, srcSampler, in.uv, 0.f);
}
