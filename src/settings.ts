// GUI options; most are read when a renderer is created
export const debugViews = { none: 0, clusterLightCount: 1, clusterIndex: 2 };
export const clusterCullingModes = { off: 'off', skipInactive: 'skip inactive', compact: 'stream compaction' };
export const postProcessModes = { none: 'none', toon: 'toon' };

export const settings = {
    animateLights: true,
    debugView: 'none' as keyof typeof debugViews,
    postProcess: postProcessModes.none,
    renderBundles: false,

    clusterCulling: clusterCullingModes.off,

    // Forward+ only
    depthPrepass: false,

    // Clustered Deferred only
    packedGBuffer: false,
    computeShading: false
};
