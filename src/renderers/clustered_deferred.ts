import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';
import { settings } from '../settings';

export class ClusteredDeferredRenderer extends renderer.Renderer {
    // TODO-3: add layouts, pipelines, textures, etc. needed for Forward+ here
    // you may need extra uniforms such as the camera view matrix and the canvas resolution
    static readonly albedoFormat: GPUTextureFormat = "rgba8unorm";
    static readonly normalFormat: GPUTextureFormat = "rgba16float";
    static readonly positionFormat: GPUTextureFormat = "rgba32float";
    // x = albedo, y = octahedral normal; position comes from depth
    static readonly packedFormat: GPUTextureFormat = "rg32uint";

    packedGBuffer: boolean;
    computeShading: boolean;

    sceneUniformsBindGroupLayout: GPUBindGroupLayout;
    sceneUniformsBindGroup: GPUBindGroup;

    gBufferBindGroupLayout: GPUBindGroupLayout;
    gBufferBindGroup: GPUBindGroup;
    outputBindGroupLayout: GPUBindGroupLayout | undefined;

    gBufferTextures: GPUTexture[] = [];
    gBufferTextureViews: GPUTextureView[] = [];

    depthFormat: GPUTextureFormat;
    depthTexture: GPUTexture;
    depthTextureView: GPUTextureView;

    gBufferPipeline: GPURenderPipeline;
    gBufferBundle: GPURenderBundle | undefined;
    fullscreenPipeline: GPURenderPipeline | undefined;
    shadingComputePipeline: GPUComputePipeline | undefined;

    constructor(stage: Stage) {
        super(stage);

        // TODO-3: initialize layouts, pipelines, textures, etc. needed for Forward+ here
        // you'll need two pipelines: one for the G-buffer pass and one for the fullscreen pass
        this.packedGBuffer = settings.packedGBuffer;
        this.computeShading = settings.computeShading;

        this.sceneUniformsBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "clustered deferred scene uniforms bind group layout",
            entries: [
                { // cameraUniforms
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" }
                },
                { // lightSet
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" }
                },
                { // clusterSet
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" }
                }
            ]
        });

        this.sceneUniformsBindGroup = renderer.device.createBindGroup({
            label: "clustered deferred scene uniforms bind group",
            layout: this.sceneUniformsBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.camera.uniformsBuffer }
                },
                {
                    binding: 1,
                    resource: { buffer: this.lights.lightSetStorageBuffer }
                },
                {
                    binding: 2,
                    resource: { buffer: this.lights.clusterSetStorageBuffer }
                }
            ]
        });

        const gBufferFormats = this.packedGBuffer
            ? [ ClusteredDeferredRenderer.packedFormat ]
            : [ ClusteredDeferredRenderer.albedoFormat, ClusteredDeferredRenderer.normalFormat, ClusteredDeferredRenderer.positionFormat ];
        for (const format of gBufferFormats) {
            const texture = renderer.device.createTexture({
                label: `g-buffer ${format}`,
                size: [renderer.canvas.width, renderer.canvas.height],
                format: format,
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
            });
            this.gBufferTextures.push(texture);
            this.gBufferTextureViews.push(texture.createView());
        }

        // packed: position is rebuilt from depth
        this.depthFormat = this.packedGBuffer ? "depth32float" : "depth24plus";
        this.depthTexture = renderer.device.createTexture({
            size: [renderer.canvas.width, renderer.canvas.height],
            format: this.depthFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.depthTextureView = this.depthTexture.createView();

        const shadingStage = this.computeShading ? GPUShaderStage.COMPUTE : GPUShaderStage.FRAGMENT;
        this.gBufferBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "g-buffer bind group layout",
            entries: this.packedGBuffer ? [
                { // gBufferTex
                    binding: 0,
                    visibility: shadingStage,
                    texture: { sampleType: "uint" }
                },
                { // depthTex
                    binding: 1,
                    visibility: shadingStage,
                    texture: { sampleType: "depth" }
                }
            ] : [
                { // albedo
                    binding: 0,
                    visibility: shadingStage,
                    texture: {}
                },
                { // normal
                    binding: 1,
                    visibility: shadingStage,
                    texture: {}
                },
                { // position
                    binding: 2,
                    visibility: shadingStage,
                    texture: { sampleType: "unfilterable-float" }
                }
            ]
        });

        this.gBufferBindGroup = renderer.device.createBindGroup({
            label: "g-buffer bind group",
            layout: this.gBufferBindGroupLayout,
            entries: (this.packedGBuffer
                ? [ this.gBufferTextureViews[0], this.depthTextureView ]
                : this.gBufferTextureViews
            ).map((view, i) => ({ binding: i, resource: view }))
        });

        this.gBufferPipeline = renderer.device.createRenderPipeline({
            label: "g-buffer pipeline",
            layout: renderer.device.createPipelineLayout({
                label: "g-buffer pipeline layout",
                bindGroupLayouts: [
                    this.sceneUniformsBindGroupLayout,
                    renderer.modelBindGroupLayout,
                    renderer.materialBindGroupLayout
                ]
            }),
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: "less",
                format: this.depthFormat
            },
            vertex: {
                module: renderer.device.createShaderModule({
                    label: "naive vert shader (g-buffer)",
                    code: shaders.naiveVertSrc
                }),
                buffers: [ renderer.vertexBufferLayout ]
            },
            fragment: {
                module: renderer.device.createShaderModule({
                    label: "g-buffer frag shader",
                    code: this.packedGBuffer ? shaders.clusteredDeferredPackedFragSrc : shaders.clusteredDeferredFragSrc
                }),
                targets: gBufferFormats.map(format => ({ format: format }))
            }
        });
        this.gBufferBundle = this.createSceneBundle(this.gBufferPipeline, this.sceneUniformsBindGroup,
            gBufferFormats, this.depthFormat);

        const shadingModule = renderer.device.createShaderModule({
            label: "clustered deferred shading shader",
            code: shaders.clusteredDeferredShadingSrc(this.packedGBuffer, this.computeShading)
        });

        if (this.computeShading) {
            this.outputBindGroupLayout = renderer.device.createBindGroupLayout({
                label: "deferred compute output bind group layout",
                entries: [
                    { // outputTex
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        storageTexture: { access: "write-only", format: "rgba8unorm" }
                    }
                ]
            });

            this.shadingComputePipeline = renderer.device.createComputePipeline({
                label: "clustered deferred compute shading pipeline",
                layout: renderer.device.createPipelineLayout({
                    label: "clustered deferred compute shading pipeline layout",
                    bindGroupLayouts: [
                        this.sceneUniformsBindGroupLayout,
                        this.gBufferBindGroupLayout,
                        this.outputBindGroupLayout
                    ]
                }),
                compute: {
                    module: shadingModule,
                    entryPoint: "main"
                }
            });
        } else {
            this.fullscreenPipeline = renderer.device.createRenderPipeline({
                label: "clustered deferred fullscreen pipeline",
                layout: renderer.device.createPipelineLayout({
                    label: "clustered deferred fullscreen pipeline layout",
                    bindGroupLayouts: [
                        this.sceneUniformsBindGroupLayout,
                        this.gBufferBindGroupLayout
                    ]
                }),
                vertex: {
                    module: renderer.device.createShaderModule({
                        label: "clustered deferred fullscreen vert shader",
                        code: shaders.clusteredDeferredFullscreenVertSrc
                    })
                },
                fragment: {
                    module: shadingModule,
                    targets: [
                        {
                            format: renderer.canvasFormat
                        }
                    ]
                }
            });
        }

        this.initClusterCulling(this.depthTextureView);
        this.initPostProcess(this.depthTextureView);
    }

    override draw() {
        // TODO-3: run the Forward+ rendering pass:
        // - run the clustering compute shader
        // - run the G-buffer pass, outputting position, albedo, and normals
        // - run the fullscreen pass, which reads from the G-buffer and performs lighting calculations
        const encoder = renderer.device.createCommandEncoder();
        const canvasTextureView = this.getColorTargetView();

        const gBufferPass = encoder.beginRenderPass({
            label: "g-buffer render pass",
            colorAttachments: this.gBufferTextureViews.map(view => ({
                view: view,
                clearValue: [0, 0, 0, 0],
                loadOp: "clear",
                storeOp: "store"
            } as GPURenderPassColorAttachment)),
            depthStencilAttachment: {
                view: this.depthTextureView,
                depthClearValue: 1.0,
                depthLoadOp: "clear",
                depthStoreOp: "store"
            },
            timestampWrites: this.profiler.timestampWrites("gBuffer")
        });
        this.drawScene(gBufferPass, this.gBufferPipeline, this.sceneUniformsBindGroup, this.gBufferBundle);
        gBufferPass.end();

        this.cullAndClusterLights(encoder);

        if (this.shadingComputePipeline) {
            const outputBindGroup = renderer.device.createBindGroup({
                label: "deferred compute output bind group",
                layout: this.outputBindGroupLayout!,
                entries: [ { binding: 0, resource: canvasTextureView } ]
            });

            const computePass = encoder.beginComputePass({
                label: "clustered deferred compute shading pass",
                timestampWrites: this.profiler.timestampWrites("shading")
            });
            computePass.setPipeline(this.shadingComputePipeline);
            computePass.setBindGroup(shaders.constants.bindGroup_scene, this.sceneUniformsBindGroup);
            computePass.setBindGroup(shaders.constants.bindGroup_gBuffer, this.gBufferBindGroup);
            computePass.setBindGroup(shaders.constants.bindGroup_output, outputBindGroup);
            computePass.dispatchWorkgroups(
                Math.ceil(renderer.canvas.width / shaders.constants.screenWorkgroupSize),
                Math.ceil(renderer.canvas.height / shaders.constants.screenWorkgroupSize)
            );
            computePass.end();
        } else {
            const fullscreenPass = encoder.beginRenderPass({
                label: "clustered deferred fullscreen pass",
                colorAttachments: [
                    {
                        view: canvasTextureView,
                        clearValue: [0, 0, 0, 0],
                        loadOp: "clear",
                        storeOp: "store"
                    }
                ],
                timestampWrites: this.profiler.timestampWrites("shading")
            });
            fullscreenPass.setPipeline(this.fullscreenPipeline!);

            fullscreenPass.setBindGroup(shaders.constants.bindGroup_scene, this.sceneUniformsBindGroup);
            fullscreenPass.setBindGroup(shaders.constants.bindGroup_gBuffer, this.gBufferBindGroup);

            fullscreenPass.draw(3);

            fullscreenPass.end();
        }

        this.runPostProcess(encoder);
        this.profiler.resolve(encoder);
        renderer.device.queue.submit([encoder.finish()]);
    }
}
