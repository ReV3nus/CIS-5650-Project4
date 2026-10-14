// TODO-2: implement the Forward+ fragment shader

// See naive.fs.wgsl for basic fragment shader setup; this shader should use light clusters instead of looping over all lights

// ------------------------------------
// Shading process:
// ------------------------------------
// Determine which cluster contains the current fragment.
// Retrieve the number of lights that affect the current fragment from the cluster’s data.
// Initialize a variable to accumulate the total light contribution for the fragment.
// For each light in the cluster:
//     Access the light's properties using its index.
//     Calculate the contribution of the light based on its position, the fragment’s position, and the surface normal.
//     Add the calculated contribution to the total light accumulation.
// Multiply the fragment’s diffuse color by the accumulated light contribution.
// Return the final color, ensuring that the alpha component is set appropriately (typically to 1).

// flipped when using the depth prepass
override alphaTest: bool = true;
override clusterFromDepth: bool = false;

@group(${bindGroup_material}) @binding(0) var diffuseTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(1) var diffuseTexSampler: sampler;

struct FragmentInput
{
    @builtin(position) fragCoord: vec4f,
    @location(0) pos: vec3f,
    @location(1) nor: vec3f,
    @location(2) uv: vec2f
}

@fragment
fn main(in: FragmentInput) -> @location(0) vec4f
{
    let diffuseColor = textureSample(diffuseTex, diffuseTexSampler, in.uv);
    if (alphaTest && diffuseColor.a < 0.5f) {
        discard;
    }

    var viewDepth: f32;
    if (clusterFromDepth) {
        viewDepth = linearizeDepth(in.fragCoord.z, cameraUniforms.nearPlane, cameraUniforms.farPlane);
    } else {
        viewDepth = -(cameraUniforms.viewMat * vec4f(in.pos, 1.f)).z;
    }

    let finalColor = shadeClustered(diffuseColor.rgb, in.pos, normalize(in.nor), in.fragCoord.xy, viewDepth);
    return vec4(finalColor, 1);
}
