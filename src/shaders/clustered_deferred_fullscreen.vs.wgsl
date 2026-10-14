// TODO-3: implement the Clustered Deferred fullscreen vertex shader

// This shader should be very simple as it does not need all of the information passed by the the naive vertex shader.

// fullscreen triangle
@vertex
fn main(@builtin(vertex_index) vertexIdx: u32) -> @builtin(position) vec4f
{
    var positions = array<vec2f, 3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
    return vec4f(positions[vertexIdx], 0, 1);
}
