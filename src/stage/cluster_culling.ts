import { device, canvas } from "../renderer";
import * as shaders from '../shaders/shaders';
import { Camera } from "./camera";
import { Lights } from "./lights";

// marks clusters that contain visible geometry, optionally compacted for an indirect dispatch
export class ClusterCulling {
    private lights: Lights;

    private bindGroup: GPUBindGroup;
    private markActivePipeline: GPUComputePipeline;
    private compactPipeline: GPUComputePipeline;
    private writeIndirectArgsPipeline: GPUComputePipeline;

    constructor(lights: Lights, camera: Camera, depthTextureView: GPUTextureView) {
        this.lights = lights;

        const bindGroupLayout = device.createBindGroupLayout({
            label: "cluster culling bind group layout",
            entries: [
                { // cameraUniforms
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" }
                },
                { // depthTex
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: "depth" }
                },
                { // activeClusterFlags
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" }
                },
                { // activeClusterList
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" }
                },
                { // indirectArgs
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" }
                }
            ]
        });

        this.bindGroup = device.createBindGroup({
            label: "cluster culling bind group",
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: camera.uniformsBuffer } },
                { binding: 1, resource: depthTextureView },
                { binding: 2, resource: { buffer: lights.activeClusterFlagsBuffer } },
                { binding: 3, resource: { buffer: lights.activeClusterListBuffer } },
                { binding: 4, resource: { buffer: lights.clusteringIndirectBuffer } }
            ]
        });

        const module = device.createShaderModule({
            label: "cluster culling compute shader",
            code: shaders.clusterCullingComputeSrc
        });
        const layout = device.createPipelineLayout({
            label: "cluster culling pipeline layout",
            bindGroupLayouts: [ bindGroupLayout ]
        });
        const createPipeline = (entryPoint: string) => device.createComputePipeline({
            label: `cluster culling pipeline (${entryPoint})`,
            layout: layout,
            compute: { module: module, entryPoint: entryPoint }
        });
        this.markActivePipeline = createPipeline("markActive");
        this.compactPipeline = createPipeline("compact");
        this.writeIndirectArgsPipeline = createPipeline("writeIndirectArgs");
    }

    run(encoder: GPUCommandEncoder, compact: boolean,
        markTimestampWrites?: GPUComputePassTimestampWrites, compactTimestampWrites?: GPUComputePassTimestampWrites) {
        encoder.clearBuffer(this.lights.activeClusterFlagsBuffer);
        encoder.clearBuffer(this.lights.activeClusterListBuffer, 0, 4); // count

        const markPass = encoder.beginComputePass({ label: "mark active clusters pass", timestampWrites: markTimestampWrites });
        markPass.setPipeline(this.markActivePipeline);
        markPass.setBindGroup(shaders.constants.bindGroup_scene, this.bindGroup);
        markPass.dispatchWorkgroups(
            Math.ceil(canvas.width / shaders.constants.screenWorkgroupSize),
            Math.ceil(canvas.height / shaders.constants.screenWorkgroupSize)
        );
        markPass.end();

        if (!compact) {
            return;
        }

        const compactPass = encoder.beginComputePass({ label: "compact active clusters pass", timestampWrites: compactTimestampWrites });
        compactPass.setBindGroup(shaders.constants.bindGroup_scene, this.bindGroup);
        compactPass.setPipeline(this.compactPipeline);
        compactPass.dispatchWorkgroups(Math.ceil(shaders.numClusters / shaders.constants.compactedClusteringWorkgroupSize));
        compactPass.setPipeline(this.writeIndirectArgsPipeline);
        compactPass.dispatchWorkgroups(1);
        compactPass.end();
    }
}
