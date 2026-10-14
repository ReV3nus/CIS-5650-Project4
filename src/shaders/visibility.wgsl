// visibility pass: writes (objectId, triangleId) per pixel

struct VertexOutput
{
    @builtin(position) fragPos: vec4f,
    @location(0) uv: vec2f,
    @location(1) @interpolate(flat) visibilityId: u32,
    @location(2) @interpolate(flat) materialIdx: u32
}

// non-indexed draw with vertex pulling; firstInstance is the object id
@vertex
fn vsMain(@builtin(vertex_index) vertexIdx: u32, @builtin(instance_index) objectId: u32) -> VertexOutput
{
    let v = loadVertex(objectId, vertexIdx);

    var out: VertexOutput;
    out.fragPos = cameraUniforms.viewProjMat * (objects[objectId].modelMat * vec4f(v.pos, 1.f));
    out.uv = v.uv;
    out.visibilityId = packVisibilityId(objectId, vertexIdx / 3u);
    out.materialIdx = objects[objectId].materialIdx;
    return out;
}

@fragment
fn fsMain(in: VertexOutput) -> @location(0) u32
{
    if (textureSample(diffuseTexArray, diffuseSampler, in.uv, in.materialIdx).a < 0.5f) {
        discard;
    }
    return in.visibilityId;
}
