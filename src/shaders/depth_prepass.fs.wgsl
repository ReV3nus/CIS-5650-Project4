// depth prepass: alpha test only

@group(${bindGroup_material}) @binding(0) var diffuseTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(1) var diffuseTexSampler: sampler;

@fragment
fn main(@location(2) uv: vec2f)
{
    if (textureSample(diffuseTex, diffuseTexSampler, uv).a < 0.5f) {
        discard;
    }
}
