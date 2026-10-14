// CHECKITOUT: code that you add here will be prepended to all shaders

struct Light {
    pos: vec3f,
    color: vec3f
}

struct LightSet {
    numLights: u32,
    lights: array<Light>
}

// TODO-2: you may want to create a ClusterSet struct similar to LightSet
// each cluster owns a range of the shared lightIndices pool
struct Cluster {
    offset: u32,
    numLights: u32
}

struct ClusterSet {
    clusters: array<Cluster, ${clusterX} * ${clusterY} * ${clusterZ}>,
    lightIndices: array<u32>
}

struct ActiveClusterList {
    count: u32,
    indices: array<u32>
}

struct CameraUniforms {
    // TODO-1.3: add an entry for the view proj mat (of type mat4x4f)
    viewProjMat: mat4x4f,
    viewMat: mat4x4f,
    invProjMat: mat4x4f,
    invViewProjMat: mat4x4f,
    screenSize: vec2f,
    nearPlane: f32,
    farPlane: f32,
    debugMode: u32
}

// exponential slices over [clusterNearPlane, clusterFarPlane]
fn clusterSliceDepth(slice: u32) -> f32 {
    let near = f32(${clusterNearPlane});
    let far = f32(${clusterFarPlane});
    return near * pow(far / near, f32(slice) / f32(${clusterZ}));
}

// inverse of clusterSliceDepth
fn clusterSliceCoord(viewDepth: f32) -> f32 {
    let near = f32(${clusterNearPlane});
    let far = f32(${clusterFarPlane});
    return log(viewDepth / near) / log(far / near) * f32(${clusterZ});
}

fn getClusterTile(fragCoord: vec2f, screenSize: vec2f) -> vec2u {
    return vec2u(clamp(fragCoord / screenSize * vec2f(${clusterX}, ${clusterY}),
                       vec2f(0), vec2f(${clusterX} - 1, ${clusterY} - 1)));
}

fn flattenClusterIndex(tile: vec2u, slice: u32) -> u32 {
    return tile.x + tile.y * ${clusterX} + slice * ${clusterX} * ${clusterY};
}

fn getClusterIndex(fragCoord: vec2f, viewDepth: f32, screenSize: vec2f) -> u32 {
    let slice = u32(clamp(floor(clusterSliceCoord(viewDepth)), 0.f, f32(${clusterZ} - 1)));
    return flattenClusterIndex(getClusterTile(fragCoord, screenSize), slice);
}

// depth buffer value -> view depth
fn linearizeDepth(depth: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - depth * (far - near));
}

fn reconstructWorldPos(fragCoord: vec2f, depth: f32, screenSize: vec2f, invViewProjMat: mat4x4f) -> vec3f {
    let ndc = vec2f(fragCoord.x / screenSize.x * 2.f - 1.f, 1.f - fragCoord.y / screenSize.y * 2.f);
    let p = invViewProjMat * vec4f(ndc, depth, 1.f);
    return p.xyz / p.w;
}

// octahedral normal encoding
fn octWrap(v: vec2f) -> vec2f {
    return (1.f - abs(v.yx)) * select(vec2f(-1.f), vec2f(1.f), v >= vec2f(0.f));
}

fn encodeNormalOct(n: vec3f) -> u32 {
    var p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.f) {
        p = octWrap(p);
    }
    return pack2x16snorm(p);
}

fn decodeNormalOct(e: u32) -> vec3f {
    let f = unpack2x16snorm(e);
    var n = vec3f(f, 1.f - abs(f.x) - abs(f.y));
    let t = max(-n.z, 0.f);
    n.x += select(t, -t, n.x >= 0.f);
    n.y += select(t, -t, n.y >= 0.f);
    return normalize(n);
}

// CHECKITOUT: this special attenuation function ensures lights don't affect geometry outside the maximum light radius
fn rangeAttenuation(distance: f32) -> f32 {
    return clamp(1.f - pow(distance / ${lightRadius}, 4.f), 0.f, 1.f) / (distance * distance);
}

fn calculateLightContrib(light: Light, posWorld: vec3f, nor: vec3f) -> vec3f {
    let vecToLight = light.pos - posWorld;
    let distToLight = length(vecToLight);

    let lambert = max(dot(nor, normalize(vecToLight)), 0.f);
    return light.color * lambert * rangeAttenuation(distToLight);
}
