// compute version of the fullscreen shading pass

@group(${bindGroup_output}) @binding(0) var outputTex: texture_storage_2d<rgba8unorm, write>;

@compute
@workgroup_size(${screenWorkgroupSize}, ${screenWorkgroupSize})
fn main(@builtin(global_invocation_id) globalIdx: vec3u) {
    if (any(globalIdx.xy >= vec2u(cameraUniforms.screenSize))) {
        return;
    }
    let coord = vec2i(globalIdx.xy);

    let g = readGBuffer(coord);
    var color = vec3f(0);
    if (g.valid) {
        color = shadeClustered(g.albedo, g.pos, g.nor, vec2f(globalIdx.xy) + 0.5f, g.viewDepth);
    }
    textureStore(outputTex, coord, vec4f(color, 1));
}
