// packed G-buffer: rg32uint + depth

@group(${bindGroup_gBuffer}) @binding(0) var gBufferTex: texture_2d<u32>;
@group(${bindGroup_gBuffer}) @binding(1) var depthTex: texture_depth_2d;

struct GBufferSample {
    valid: bool,
    albedo: vec3f,
    nor: vec3f,
    pos: vec3f,
    viewDepth: f32
}

fn readGBuffer(coord: vec2i) -> GBufferSample {
    var g: GBufferSample;
    let depth = textureLoad(depthTex, coord, 0);
    g.valid = depth < 1.f;
    let packed = textureLoad(gBufferTex, coord, 0).xy;
    g.albedo = unpack4x8unorm(packed.x).rgb;
    g.nor = decodeNormalOct(packed.y);
    g.pos = reconstructWorldPos(vec2f(coord) + 0.5f, depth, cameraUniforms.screenSize, cameraUniforms.invViewProjMat);
    g.viewDepth = linearizeDepth(depth, cameraUniforms.nearPlane, cameraUniforms.farPlane);
    return g;
}
