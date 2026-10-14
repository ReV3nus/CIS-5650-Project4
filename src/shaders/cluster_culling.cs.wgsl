// depth-based cluster culling: markActive -> compact -> writeIndirectArgs

struct ActiveClusterListAtomic {
    count: atomic<u32>,
    indices: array<u32>
}

@group(${bindGroup_scene}) @binding(0) var<uniform> cameraUniforms: CameraUniforms;
@group(${bindGroup_scene}) @binding(1) var depthTex: texture_depth_2d;
@group(${bindGroup_scene}) @binding(2) var<storage, read_write> activeClusterFlags: array<u32>;
@group(${bindGroup_scene}) @binding(3) var<storage, read_write> activeClusterList: ActiveClusterListAtomic;
@group(${bindGroup_scene}) @binding(4) var<storage, read_write> indirectArgs: array<u32, 3>;

@compute
@workgroup_size(${screenWorkgroupSize}, ${screenWorkgroupSize})
fn markActive(@builtin(global_invocation_id) globalIdx: vec3u) {
    if (any(globalIdx.xy >= vec2u(cameraUniforms.screenSize))) {
        return;
    }
    let depth = textureLoad(depthTex, vec2i(globalIdx.xy), 0);
    if (depth >= 1.f) {
        return;
    }

    let near = cameraUniforms.nearPlane;
    let far = cameraUniforms.farPlane;
    let sliceCoord = clusterSliceCoord(linearizeDepth(depth, near, far));
    let slice = u32(clamp(floor(sliceCoord), 0.f, f32(${clusterZ} - 1)));
    let tile = getClusterTile(vec2f(globalIdx.xy) + 0.5f, cameraUniforms.screenSize);
    activeClusterFlags[flattenClusterIndex(tile, slice)] = 1u;

    // near a slice boundary, also mark the neighbor in case shading rounds the other way
    let f = fract(sliceCoord);
    if (f < 1e-3f && slice > 0u) {
        activeClusterFlags[flattenClusterIndex(tile, slice - 1u)] = 1u;
    }
    if (f > 1.f - 1e-3f && slice < ${clusterZ}u - 1u) {
        activeClusterFlags[flattenClusterIndex(tile, slice + 1u)] = 1u;
    }
}

@compute
@workgroup_size(${compactedClusteringWorkgroupSize})
fn compact(@builtin(global_invocation_id) globalIdx: vec3u) {
    let clusterIdx = globalIdx.x;
    if (clusterIdx >= ${clusterX}u * ${clusterY}u * ${clusterZ}u || activeClusterFlags[clusterIdx] == 0u) {
        return;
    }
    let slot = atomicAdd(&activeClusterList.count, 1u);
    activeClusterList.indices[slot] = clusterIdx;
}

@compute
@workgroup_size(1)
fn writeIndirectArgs() {
    indirectArgs[0] = atomicLoad(&activeClusterList.count);
    indirectArgs[1] = 1u;
    indirectArgs[2] = 1u;
}
