// CHECKITOUT: this file loads all the shaders and preprocesses them with some common code

import { Camera } from '../stage/camera';

import commonRaw from './common.wgsl?raw';

import naiveVertRaw from './naive.vs.wgsl?raw';
import naiveFragRaw from './naive.fs.wgsl?raw';

import forwardPlusFragRaw from './forward_plus.fs.wgsl?raw';

import clusteredDeferredFragRaw from './clustered_deferred.fs.wgsl?raw';
import clusteredDeferredFullscreenVertRaw from './clustered_deferred_fullscreen.vs.wgsl?raw';
import clusteredDeferredFullscreenFragRaw from './clustered_deferred_fullscreen.fs.wgsl?raw';

import moveLightsComputeRaw from './move_lights.cs.wgsl?raw';
import clusteringComputeRaw from './clustering.cs.wgsl?raw';

import clusteredShadingRaw from './clustered_shading.wgsl?raw';
import gBufferUnpackedRaw from './gbuffer_unpacked.wgsl?raw';
import gBufferPackedRaw from './gbuffer_packed.wgsl?raw';
import clusteredDeferredPackedFragRaw from './clustered_deferred_packed.fs.wgsl?raw';
import clusteredDeferredShadingComputeRaw from './clustered_deferred_shading.cs.wgsl?raw';
import depthPrepassFragRaw from './depth_prepass.fs.wgsl?raw';
import clusterCullingComputeRaw from './cluster_culling.cs.wgsl?raw';
import toonComputeRaw from './toon.cs.wgsl?raw';
import visibilityCommonRaw from './visibility_common.wgsl?raw';
import visibilityRaw from './visibility.wgsl?raw';
import visibilityShadingComputeRaw from './visibility_shading.cs.wgsl?raw';
import textureBlitRaw from './texture_blit.wgsl?raw';

// CONSTANTS (for use in shaders)
// =================================

// CHECKITOUT: feel free to add more constants here and to refer to them in your shader code

// Note that these are declared in a somewhat roundabout way because otherwise minification will drop variables
// that are unused in host side code.
export const constants = {
    bindGroup_scene: 0,
    bindGroup_model: 1,
    bindGroup_material: 2,

    bindGroup_gBuffer: 1,
    bindGroup_visibility: 1,
    bindGroup_output: 2,

    moveLightsWorkgroupSize: 128,

    lightRadius: 2,

    clusterX: 32,
    clusterY: 18,
    clusterZ: 48,
    // depth range of the Z slices, independent of the camera planes
    clusterNearPlane: 1,
    clusterFarPlane: 40,
    maxLightsPerCluster: 2048,
    clusterLightPoolSize: 8388608, // light indices shared by all clusters
    clusteringWorkgroupSize: 64,
    compactedClusteringWorkgroupSize: 64,

    screenWorkgroupSize: 8,

    visibilityTriangleBits: 16 // id = (objectId << 16) | triangleId
};

export const numClusters = constants.clusterX * constants.clusterY * constants.clusterZ;

// =================================

// no eval: the bundler would tree-shake constants that only shaders use
function evalShaderRaw(raw: string) {
    return raw.replace(/\$\{(\w+)\}/g, (_, name: string) => {
        const value = (constants as { [key: string]: number })[name];
        if (value === undefined) {
            throw new Error(`unknown shader constant: ${name}`);
        }
        return String(value);
    });
}

const commonSrc: string = evalShaderRaw(commonRaw);

function processShaderRaw(raw: string) {
    return commonSrc + evalShaderRaw(raw);
}

export const naiveVertSrc: string = processShaderRaw(naiveVertRaw);
export const naiveFragSrc: string = processShaderRaw(naiveFragRaw);

function processClusteredShaderRaw(...raws: string[]) {
    return processShaderRaw(clusteredShadingRaw) + raws.map(evalShaderRaw).join('\n');
}

export const forwardPlusFragSrc: string = processClusteredShaderRaw(forwardPlusFragRaw);
export const depthPrepassFragSrc: string = processShaderRaw(depthPrepassFragRaw);

export const clusteredDeferredFragSrc: string = processShaderRaw(clusteredDeferredFragRaw);
export const clusteredDeferredPackedFragSrc: string = processShaderRaw(clusteredDeferredPackedFragRaw);
export const clusteredDeferredFullscreenVertSrc: string = processShaderRaw(clusteredDeferredFullscreenVertRaw);

export function clusteredDeferredShadingSrc(packed: boolean, compute: boolean): string {
    return processClusteredShaderRaw(packed ? gBufferPackedRaw : gBufferUnpackedRaw,
        compute ? clusteredDeferredShadingComputeRaw : clusteredDeferredFullscreenFragRaw);
}

export const moveLightsComputeSrc: string = processShaderRaw(moveLightsComputeRaw);
export const clusteringComputeSrc: string = processShaderRaw(clusteringComputeRaw);
export const clusterCullingComputeSrc: string = processShaderRaw(clusterCullingComputeRaw);

export const toonComputeSrc: string = processShaderRaw(toonComputeRaw);

export const visibilitySrc: string = processClusteredShaderRaw(visibilityCommonRaw, visibilityRaw);
export const visibilityShadingComputeSrc: string = processClusteredShaderRaw(visibilityCommonRaw, visibilityShadingComputeRaw);
export const textureBlitSrc: string = processShaderRaw(textureBlitRaw);
