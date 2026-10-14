// TODO-2: implement the light clustering compute shader

// ------------------------------------
// Calculating cluster bounds:
// ------------------------------------
// For each cluster (X, Y, Z):
//     - Calculate the screen-space bounds for this cluster in 2D (XY).
//     - Calculate the depth bounds for this cluster in Z (near and far planes).
//     - Convert these screen and depth bounds into view-space coordinates.
//     - Store the computed bounding box (AABB) for the cluster.

// ------------------------------------
// Assigning lights to clusters:
// ------------------------------------
// For each cluster:
//     - Initialize a counter for the number of lights in this cluster.

//     For each light:
//         - Check if the light intersects with the cluster’s bounding box (AABB).
//         - If it does, add the light to the cluster's light list.
//         - Stop adding lights if the maximum number of lights is reached.

//     - Store the number of lights assigned to this cluster.

@group(${bindGroup_scene}) @binding(0) var<uniform> cameraUniforms: CameraUniforms;
@group(${bindGroup_scene}) @binding(1) var<storage, read> lightSet: LightSet;
@group(${bindGroup_scene}) @binding(2) var<storage, read_write> clusterSet: ClusterSet;
@group(${bindGroup_scene}) @binding(3) var<storage, read> activeClusterFlags: array<u32>;
@group(${bindGroup_scene}) @binding(4) var<storage, read> activeClusterList: ActiveClusterList;
// next free slot in clusterSet.lightIndices
@group(${bindGroup_scene}) @binding(5) var<storage, read_write> lightIndexCounter: atomic<u32>;

// unproject an NDC xy point onto the near plane (view space)
fn ndcToViewNear(ndc: vec2f) -> vec3f {
    let p = cameraUniforms.invProjMat * vec4f(ndc, 0.f, 1.f);
    return p.xyz / p.w;
}

fn sphereIntersectsAabb(center: vec3f, radius: f32, aabbMin: vec3f, aabbMax: vec3f) -> bool {
    let d = clamp(center, aabbMin, aabbMax) - center;
    return dot(d, d) <= radius * radius;
}

// one workgroup per cluster: gather hits in workgroup memory, then copy them to the pool
var<workgroup> wgNumLights: atomic<u32>;
var<workgroup> wgLightIndices: array<u32, ${maxLightsPerCluster}>;
var<workgroup> wgPoolOffset: u32;

fn clusterLights(clusterCoord: vec3u, localIdx: u32) {
    let clusterIdx = flattenClusterIndex(clusterCoord.xy, clusterCoord.z);

    // tile row 0 is the top of the screen
    let tileSize = vec2f(2.f) / vec2f(${clusterX}, ${clusterY});
    let ndcMin = vec2f(-1.f + f32(clusterCoord.x) * tileSize.x, 1.f - f32(clusterCoord.y + 1u) * tileSize.y);
    let ndcMax = ndcMin + tileSize;

    // the end slices extend to the camera planes
    let near = cameraUniforms.nearPlane;
    let sliceNear = select(clusterSliceDepth(clusterCoord.z), near, clusterCoord.z == 0u);
    let sliceFar = select(clusterSliceDepth(clusterCoord.z + 1u), cameraUniforms.farPlane,
                          clusterCoord.z == ${clusterZ}u - 1u);

    // tile corners on the near plane, scaled along the eye rays to the slice depths
    let c0 = ndcToViewNear(ndcMin);
    let c1 = ndcToViewNear(vec2f(ndcMax.x, ndcMin.y));
    let c2 = ndcToViewNear(vec2f(ndcMin.x, ndcMax.y));
    let c3 = ndcToViewNear(ndcMax);
    let cMin = min(min(c0, c1), min(c2, c3));
    let cMax = max(max(c0, c1), max(c2, c3));
    let scaleNear = sliceNear / near;
    let scaleFar = sliceFar / near;
    let aabbMin = min(cMin * scaleNear, cMin * scaleFar);
    let aabbMax = max(cMax * scaleNear, cMax * scaleFar);

    if (localIdx == 0u) {
        atomicStore(&wgNumLights, 0u);
    }
    workgroupBarrier();

    for (var lightIdx = localIdx; lightIdx < lightSet.numLights; lightIdx += ${clusteringWorkgroupSize}u) {
        let lightPosView = (cameraUniforms.viewMat * vec4f(lightSet.lights[lightIdx].pos, 1.f)).xyz;
        if (sphereIntersectsAabb(lightPosView, f32(${lightRadius}), aabbMin, aabbMax)) {
            let slot = atomicAdd(&wgNumLights, 1u);
            if (slot < ${maxLightsPerCluster}u) {
                wgLightIndices[slot] = lightIdx;
            }
        }
    }
    workgroupBarrier();

    let numLights = min(atomicLoad(&wgNumLights), ${maxLightsPerCluster}u);
    if (localIdx == 0u) {
        wgPoolOffset = atomicAdd(&lightIndexCounter, numLights);
    }
    workgroupBarrier();

    let offset = wgPoolOffset;
    let count = min(numLights, ${clusterLightPoolSize}u - min(offset, ${clusterLightPoolSize}u));
    for (var i = localIdx; i < count; i += ${clusteringWorkgroupSize}u) {
        clusterSet.lightIndices[offset + i] = wgLightIndices[i];
    }
    if (localIdx == 0u) {
        clusterSet.clusters[clusterIdx].offset = offset;
        clusterSet.clusters[clusterIdx].numLights = count;
    }
}

@compute
@workgroup_size(${clusteringWorkgroupSize})
fn main(@builtin(workgroup_id) clusterCoord: vec3u, @builtin(local_invocation_index) localIdx: u32) {
    clusterLights(clusterCoord, localIdx);
}

@compute
@workgroup_size(${clusteringWorkgroupSize})
fn mainSkipInactive(@builtin(workgroup_id) clusterCoord: vec3u, @builtin(local_invocation_index) localIdx: u32) {
    if (activeClusterFlags[flattenClusterIndex(clusterCoord.xy, clusterCoord.z)] == 0u) {
        return;
    }
    clusterLights(clusterCoord, localIdx);
}

// indirect dispatch, one workgroup per active cluster
@compute
@workgroup_size(${clusteringWorkgroupSize})
fn mainCompacted(@builtin(workgroup_id) workgroupIdx: vec3u, @builtin(local_invocation_index) localIdx: u32) {
    let clusterIdx = activeClusterList.indices[workgroupIdx.x];
    clusterLights(vec3u(clusterIdx % ${clusterX}, (clusterIdx / ${clusterX}) % ${clusterY},
                        clusterIdx / (${clusterX} * ${clusterY})), localIdx);
}
