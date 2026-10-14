// clustered lighting, shared by Forward+, Clustered Deferred and the visibility buffer

@group(${bindGroup_scene}) @binding(0) var<uniform> cameraUniforms: CameraUniforms;
@group(${bindGroup_scene}) @binding(1) var<storage, read> lightSet: LightSet;
@group(${bindGroup_scene}) @binding(2) var<storage, read> clusterSet: ClusterSet;

// blue -> cyan -> green -> yellow -> red
fn heatmap(t: f32) -> vec3f {
    let x = clamp(t, 0.f, 1.f) * 4.f;
    return clamp(vec3f(x - 2.f, select(x, 4.f - x, x > 2.f), 2.f - x), vec3f(0.f), vec3f(1.f));
}

fn hashColor(i: u32) -> vec3f {
    var h = i * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return vec3f(f32(h & 255u), f32((h >> 8u) & 255u), f32((h >> 16u) & 255u)) / 255.f;
}

fn shadeClustered(albedo: vec3f, posWorld: vec3f, nor: vec3f, fragCoord: vec2f, viewDepth: f32) -> vec3f {
    let clusterIdx = getClusterIndex(fragCoord, viewDepth, cameraUniforms.screenSize);
    let cluster = clusterSet.clusters[clusterIdx];
    let numLights = cluster.numLights;

    if (cameraUniforms.debugMode == 1u) { // lights per cluster
        return heatmap(f32(numLights) / 256.f);
    }
    if (cameraUniforms.debugMode == 2u) { // cluster id
        return hashColor(clusterIdx) * (0.3f + 0.7f * dot(albedo, vec3f(0.299f, 0.587f, 0.114f)));
    }

    var totalLightContrib = vec3f(0, 0, 0);
    for (var i = 0u; i < numLights; i++) {
        let light = lightSet.lights[clusterSet.lightIndices[cluster.offset + i]];
        totalLightContrib += calculateLightContrib(light, posWorld, nor);
    }
    return albedo * totalLightContrib;
}
