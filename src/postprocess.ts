import { device, canvas } from './renderer';
import * as shaders from './shaders/shaders';
import { Camera } from './stage/camera';

export class ToonPostProcess {
    readonly inputTexture: GPUTexture;
    readonly inputTextureView: GPUTextureView;

    private camera: Camera;
    private depthTextureView: GPUTextureView;

    private bindGroupLayout: GPUBindGroupLayout;
    private pipeline: GPUComputePipeline;

    constructor(camera: Camera, depthTextureView: GPUTextureView) {
        this.camera = camera;
        this.depthTextureView = depthTextureView;

        this.inputTexture = device.createTexture({
            label: "post-process input",
            size: [canvas.width, canvas.height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING
        });
        this.inputTextureView = this.inputTexture.createView();

        this.bindGroupLayout = device.createBindGroupLayout({
            label: "toon bind group layout",
            entries: [
                { // cameraUniforms
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" }
                },
                { // inputTex
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: {}
                },
                { // depthTex
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: "depth" }
                },
                { // outputTex
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: { access: "write-only", format: "rgba8unorm" }
                }
            ]
        });

        this.pipeline = device.createComputePipeline({
            label: "toon compute pipeline",
            layout: device.createPipelineLayout({
                label: "toon pipeline layout",
                bindGroupLayouts: [ this.bindGroupLayout ]
            }),
            compute: {
                module: device.createShaderModule({
                    label: "toon compute shader",
                    code: shaders.toonComputeSrc
                }),
                entryPoint: "main"
            }
        });
    }

    run(encoder: GPUCommandEncoder, outputView: GPUTextureView, timestampWrites?: GPUComputePassTimestampWrites) {
        const bindGroup = device.createBindGroup({
            label: "toon bind group",
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer } },
                { binding: 1, resource: this.inputTextureView },
                { binding: 2, resource: this.depthTextureView },
                { binding: 3, resource: outputView }
            ]
        });

        const computePass = encoder.beginComputePass({ label: "toon post-process pass", timestampWrites: timestampWrites });
        computePass.setPipeline(this.pipeline);
        computePass.setBindGroup(shaders.constants.bindGroup_scene, bindGroup);
        computePass.dispatchWorkgroups(
            Math.ceil(canvas.width / shaders.constants.screenWorkgroupSize),
            Math.ceil(canvas.height / shaders.constants.screenWorkgroupSize)
        );
        computePass.end();
    }
}
