// visibility buffer shading

@group(${bindGroup_output}) @binding(0) var visibilityTex: texture_2d<u32>;
@group(${bindGroup_output}) @binding(1) var depthTex: texture_depth_2d;
@group(${bindGroup_output}) @binding(2) var outputTex: texture_storage_2d<rgba8unorm, write>;

fn barycentric(p: vec3f, a: vec3f, b: vec3f, c: vec3f) -> vec3f {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let invDenom = 1.f / (d00 * d11 - d01 * d01);
    let v = (d11 * d20 - d01 * d21) * invDenom;
    let w = (d00 * d21 - d01 * d20) * invDenom;
    return vec3f(1.f - v - w, v, w);
}

@compute
@workgroup_size(${screenWorkgroupSize}, ${screenWorkgroupSize})
fn main(@builtin(global_invocation_id) globalIdx: vec3u) {
    if (any(globalIdx.xy >= vec2u(cameraUniforms.screenSize))) {
        return;
    }
    let coord = vec2i(globalIdx.xy);

    let visibilityId = textureLoad(visibilityTex, coord, 0).r;
    if (visibilityId == invalidVisibilityId) {
        textureStore(outputTex, coord, vec4f(0, 0, 0, 1));
        return;
    }
    let objectId = visibilityId >> ${visibilityTriangleBits}u;
    let triangleId = visibilityId & ((1u << ${visibilityTriangleBits}u) - 1u);

    let v0 = loadVertex(objectId, triangleId * 3u);
    let v1 = loadVertex(objectId, triangleId * 3u + 1u);
    let v2 = loadVertex(objectId, triangleId * 3u + 2u);
    let modelMat = objects[objectId].modelMat;
    let p0 = (modelMat * vec4f(v0.pos, 1.f)).xyz;
    let p1 = (modelMat * vec4f(v1.pos, 1.f)).xyz;
    let p2 = (modelMat * vec4f(v2.pos, 1.f)).xyz;

    let pixelCenter = vec2f(globalIdx.xy) + 0.5f;
    let depth = textureLoad(depthTex, coord, 0);
    let posWorld = reconstructWorldPos(pixelCenter, depth, cameraUniforms.screenSize, cameraUniforms.invViewProjMat);

    let bary = barycentric(posWorld, p0, p1, p2);
    let uv = bary.x * v0.uv + bary.y * v1.uv + bary.z * v2.uv;
    let nor = normalize(bary.x * v0.nor + bary.y * v1.nor + bary.z * v2.nor);
    // no derivatives in compute; the textures have no mips anyway
    let albedo = textureSampleLevel(diffuseTexArray, diffuseSampler, uv, objects[objectId].materialIdx, 0.f).rgb;

    let viewDepth = linearizeDepth(depth, cameraUniforms.nearPlane, cameraUniforms.farPlane);
    textureStore(outputTex, coord, vec4f(shadeClustered(albedo, posWorld, nor, pixelCenter, viewDepth), 1));
}
