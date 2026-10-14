// TODO-3: implement the Clustered Deferred fullscreen fragment shader

// Similar to the Forward+ fragment shader, but with vertex information coming from the G-buffer instead.

// readGBuffer() is in gbuffer_packed.wgsl / gbuffer_unpacked.wgsl

@fragment
fn main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f
{
    let g = readGBuffer(vec2i(fragCoord.xy));
    if (!g.valid) {
        return vec4f(0, 0, 0, 1);
    }

    return vec4f(shadeClustered(g.albedo, g.pos, g.nor, fragCoord.xy, g.viewDepth), 1);
}
