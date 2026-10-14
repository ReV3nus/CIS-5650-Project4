// toon shading: luminance bands + depth outlines

@group(${bindGroup_scene}) @binding(0) var<uniform> cameraUniforms: CameraUniforms;
@group(${bindGroup_scene}) @binding(1) var inputTex: texture_2d<f32>;
@group(${bindGroup_scene}) @binding(2) var depthTex: texture_depth_2d;
@group(${bindGroup_scene}) @binding(3) var outputTex: texture_storage_2d<rgba8unorm, write>;

const numBands = 5.f;
const edgeThreshold = 0.03f; // relative depth jump

fn viewDepthAt(coord: vec2i) -> f32 {
    let size = vec2i(cameraUniforms.screenSize);
    let depth = textureLoad(depthTex, clamp(coord, vec2i(0), size - 1), 0);
    return linearizeDepth(depth, cameraUniforms.nearPlane, cameraUniforms.farPlane);
}

@compute
@workgroup_size(${screenWorkgroupSize}, ${screenWorkgroupSize})
fn main(@builtin(global_invocation_id) globalIdx: vec3u) {
    if (any(globalIdx.xy >= vec2u(cameraUniforms.screenSize))) {
        return;
    }
    let coord = vec2i(globalIdx.xy);
    let color = textureLoad(inputTex, coord, 0).rgb;

    // bands in sqrt space so dark areas keep some detail
    let lum = dot(color, vec3f(0.2126f, 0.7152f, 0.0722f));
    let band = round(sqrt(lum) * numBands) / numBands;
    var toon = color * (band * band / max(lum, 1e-4f));

    let d = viewDepthAt(coord);
    let maxJump = max(max(abs(viewDepthAt(coord + vec2i(1, 0)) - d), abs(viewDepthAt(coord - vec2i(1, 0)) - d)),
                      max(abs(viewDepthAt(coord + vec2i(0, 1)) - d), abs(viewDepthAt(coord - vec2i(0, 1)) - d)));
    if (maxJump / d > edgeThreshold) {
        toon = vec3f(0);
    }

    textureStore(outputTex, coord, vec4f(toon, 1));
}
