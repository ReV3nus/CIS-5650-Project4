import { vec3 } from "wgpu-matrix";
import { device } from "../renderer";

import * as shaders from '../shaders/shaders';
import { Camera } from "./camera";
import { settings, clusterCullingModes } from "../settings";

// h in [0, 1]
function hueToRgb(h: number) {
    let f = (n: number, k = (n + h * 6) % 6) => 1 - Math.max(Math.min(k, 4 - k, 1), 0);
    return vec3.lerp(vec3.create(1, 1, 1), vec3.create(f(5), f(3), f(1)), 0.8);
}

export class Lights {
    private camera: Camera;

    numLights = 500;
    static readonly maxNumLights = 5000;
    static readonly numFloatsPerLight = 8; // vec3f is aligned at 16 byte boundaries

    static readonly lightIntensity = 0.1;

    lightsArray = new Float32Array(Lights.maxNumLights * Lights.numFloatsPerLight);
    lightSetStorageBuffer: GPUBuffer;

    timeUniformBuffer: GPUBuffer;

    moveLightsComputeBindGroupLayout: GPUBindGroupLayout;
    moveLightsComputeBindGroup: GPUBindGroup;
    moveLightsComputePipeline: GPUComputePipeline;

    // TODO-2: add layouts, pipelines, textures, etc. needed for light clustering here
    // {offset, numLights} per cluster, then the light index pool
    static readonly clusterSetByteSize = shaders.numClusters * 8 + shaders.constants.clusterLightPoolSize * 4;

    clusterSetStorageBuffer: GPUBuffer;
    lightIndexCounterBuffer: GPUBuffer;

    // cluster culling
    activeClusterFlagsBuffer: GPUBuffer;
    activeClusterListBuffer: GPUBuffer;
    clusteringIndirectBuffer: GPUBuffer;

    clusteringComputeBindGroupLayout: GPUBindGroupLayout;
    clusteringComputeBindGroup: GPUBindGroup;
    clusteringComputePipeline: GPUComputePipeline;
    clusteringSkipInactivePipeline: GPUComputePipeline;
    clusteringCompactedPipeline: GPUComputePipeline;

    constructor(camera: Camera) {
        this.camera = camera;

        this.lightSetStorageBuffer = device.createBuffer({
            label: "lights",
            size: 16 + this.lightsArray.byteLength, // 16 for numLights + padding
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.populateLightsBuffer();
        this.updateLightSetUniformNumLights();

        this.timeUniformBuffer = device.createBuffer({
            label: "time uniform",
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.moveLightsComputeBindGroupLayout = device.createBindGroupLayout({
            label: "move lights compute bind group layout",
            entries: [
                { // lightSet
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" }
                },
                { // time
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" }
                }
            ]
        });

        this.moveLightsComputeBindGroup = device.createBindGroup({
            label: "move lights compute bind group",
            layout: this.moveLightsComputeBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.lightSetStorageBuffer }
                },
                {
                    binding: 1,
                    resource: { buffer: this.timeUniformBuffer }
                }
            ]
        });

        this.moveLightsComputePipeline = device.createComputePipeline({
            label: "move lights compute pipeline",
            layout: device.createPipelineLayout({
                label: "move lights compute pipeline layout",
                bindGroupLayouts: [ this.moveLightsComputeBindGroupLayout ]
            }),
            compute: {
                module: device.createShaderModule({
                    label: "move lights compute shader",
                    code: shaders.moveLightsComputeSrc
                }),
                entryPoint: "main"
            }
        });

        // TODO-2: initialize layouts, pipelines, textures, etc. needed for light clustering here
        this.clusterSetStorageBuffer = device.createBuffer({
            label: "cluster set",
            size: Lights.clusterSetByteSize,
            usage: GPUBufferUsage.STORAGE
        });
        this.lightIndexCounterBuffer = device.createBuffer({
            label: "cluster light index counter",
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        this.activeClusterFlagsBuffer = device.createBuffer({
            label: "active cluster flags",
            size: shaders.numClusters * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.activeClusterListBuffer = device.createBuffer({
            label: "active cluster list",
            size: 4 + shaders.numClusters * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.clusteringIndirectBuffer = device.createBuffer({
            label: "clustering indirect dispatch args",
            size: 3 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT
        });

        this.clusteringComputeBindGroupLayout = device.createBindGroupLayout({
            label: "clustering compute bind group layout",
            entries: [
                { // cameraUniforms
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" }
                },
                { // lightSet
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" }
                },
                { // clusterSet
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" }
                },
                { // activeClusterFlags
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" }
                },
                { // activeClusterList
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" }
                },
                { // lightIndexCounter
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" }
                }
            ]
        });

        this.clusteringComputeBindGroup = device.createBindGroup({
            label: "clustering compute bind group",
            layout: this.clusteringComputeBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.camera.uniformsBuffer }
                },
                {
                    binding: 1,
                    resource: { buffer: this.lightSetStorageBuffer }
                },
                {
                    binding: 2,
                    resource: { buffer: this.clusterSetStorageBuffer }
                },
                {
                    binding: 3,
                    resource: { buffer: this.activeClusterFlagsBuffer }
                },
                {
                    binding: 4,
                    resource: { buffer: this.activeClusterListBuffer }
                },
                {
                    binding: 5,
                    resource: { buffer: this.lightIndexCounterBuffer }
                }
            ]
        });

        const clusteringModule = device.createShaderModule({
            label: "clustering compute shader",
            code: shaders.clusteringComputeSrc
        });
        const clusteringPipelineLayout = device.createPipelineLayout({
            label: "clustering compute pipeline layout",
            bindGroupLayouts: [ this.clusteringComputeBindGroupLayout ]
        });
        const createClusteringPipeline = (entryPoint: string) => device.createComputePipeline({
            label: `clustering compute pipeline (${entryPoint})`,
            layout: clusteringPipelineLayout,
            compute: {
                module: clusteringModule,
                entryPoint: entryPoint
            }
        });
        this.clusteringComputePipeline = createClusteringPipeline("main");
        this.clusteringSkipInactivePipeline = createClusteringPipeline("mainSkipInactive");
        this.clusteringCompactedPipeline = createClusteringPipeline("mainCompacted");
    }

    private populateLightsBuffer() {
        for (let lightIdx = 0; lightIdx < Lights.maxNumLights; ++lightIdx) {
            // light pos is set by compute shader so no need to set it here
            const lightColor = vec3.scale(hueToRgb(Math.random()), Lights.lightIntensity);
            this.lightsArray.set(lightColor, (lightIdx * Lights.numFloatsPerLight) + 4);
        }

        device.queue.writeBuffer(this.lightSetStorageBuffer, 16, this.lightsArray);
    }

    updateLightSetUniformNumLights() {
        device.queue.writeBuffer(this.lightSetStorageBuffer, 0, new Uint32Array([this.numLights]));
    }

    doLightClustering(encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites,
                      cullingMode: string = clusterCullingModes.off) {
        // TODO-2: run the light clustering compute pass(es) here
        // implementing clustering here allows for reusing the code in both Forward+ and Clustered Deferred
        encoder.clearBuffer(this.lightIndexCounterBuffer);

        const computePass = encoder.beginComputePass({
            label: "light clustering compute pass",
            timestampWrites: timestampWrites
        });
        computePass.setBindGroup(shaders.constants.bindGroup_scene, this.clusteringComputeBindGroup);

        if (cullingMode === clusterCullingModes.compact) {
            computePass.setPipeline(this.clusteringCompactedPipeline);
            computePass.dispatchWorkgroupsIndirect(this.clusteringIndirectBuffer, 0);
        } else {
            computePass.setPipeline(cullingMode === clusterCullingModes.skipInactive
                ? this.clusteringSkipInactivePipeline : this.clusteringComputePipeline);

            computePass.dispatchWorkgroups(shaders.constants.clusterX, shaders.constants.clusterY, shaders.constants.clusterZ);
        }

        computePass.end();
    }

    setTime(time: number) {
        device.queue.writeBuffer(this.timeUniformBuffer, 0, new Float32Array([time]));
    }

    // CHECKITOUT: this is where the light movement compute shader is dispatched from the host
    onFrame(time: number) {
        if (settings.animateLights) {
            this.setTime(time);
        }

        // not using same encoder as render pass so this doesn't interfere with measuring actual rendering performance
        const encoder = device.createCommandEncoder();

        const computePass = encoder.beginComputePass();
        computePass.setPipeline(this.moveLightsComputePipeline);

        computePass.setBindGroup(0, this.moveLightsComputeBindGroup);

        const workgroupCount = Math.ceil(this.numLights / shaders.constants.moveLightsWorkgroupSize);
        computePass.dispatchWorkgroups(workgroupCount);

        computePass.end();

        device.queue.submit([encoder.finish()]);
    }
}
