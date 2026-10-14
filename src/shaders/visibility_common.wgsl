// merged scene geometry for the visibility buffer

struct ObjectInfo {
    modelMat: mat4x4f,
    indexOffset: u32,
    vertexOffset: u32,
    materialIdx: u32, // layer in diffuseTexArray
    _pad: u32
}

@group(${bindGroup_visibility}) @binding(0) var<storage, read> objects: array<ObjectInfo>;
@group(${bindGroup_visibility}) @binding(1) var<storage, read> indices: array<u32>;
@group(${bindGroup_visibility}) @binding(2) var<storage, read> vertices: array<f32>; // pos, nor, uv
@group(${bindGroup_visibility}) @binding(3) var diffuseTexArray: texture_2d_array<f32>;
@group(${bindGroup_visibility}) @binding(4) var diffuseSampler: sampler;

const invalidVisibilityId = 0xffffffffu;

fn packVisibilityId(objectId: u32, triangleId: u32) -> u32 {
    return (objectId << ${visibilityTriangleBits}u) | triangleId;
}

struct VertexData {
    pos: vec3f,
    nor: vec3f,
    uv: vec2f
}

fn loadVertex(objectId: u32, corner: u32) -> VertexData {
    let base = (indices[objects[objectId].indexOffset + corner] + objects[objectId].vertexOffset) * 8u;
    var v: VertexData;
    v.pos = vec3f(vertices[base], vertices[base + 1u], vertices[base + 2u]);
    v.nor = vec3f(vertices[base + 3u], vertices[base + 4u], vertices[base + 5u]);
    v.uv = vec2f(vertices[base + 6u], vertices[base + 7u]);
    return v;
}
