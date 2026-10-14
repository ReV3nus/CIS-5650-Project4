// G-buffer: albedo, normal, world position

@group(${bindGroup_gBuffer}) @binding(0) var albedoTex: texture_2d<f32>;
@group(${bindGroup_gBuffer}) @binding(1) var normalTex: texture_2d<f32>;
@group(${bindGroup_gBuffer}) @binding(2) var positionTex: texture_2d<f32>;

struct GBufferSample {
    valid: bool,
    albedo: vec3f,
    nor: vec3f,
    pos: vec3f,
    viewDepth: f32
}

fn readGBuffer(coord: vec2i) -> GBufferSample {
    var g: GBufferSample;
    let albedo = textureLoad(albedoTex, coord, 0);
    g.valid = albedo.a != 0.f;
    g.albedo = albedo.rgb;
    g.nor = textureLoad(normalTex, coord, 0).xyz;
    g.pos = textureLoad(positionTex, coord, 0).xyz;
    g.viewDepth = -(cameraUniforms.viewMat * vec4f(g.pos, 1.f)).z;
    return g;
}
