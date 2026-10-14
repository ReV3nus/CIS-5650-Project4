import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';
import { settings, clusterCullingModes } from '../settings';

export class ForwardPlusRenderer extends renderer.Renderer {
    // TODO-2: add layouts, pipelines, textures, etc. needed for Forward+ here
    // you may need extra uniforms such as the camera view matrix and the canvas resolution
    sceneUniformsBindGroupLayout: GPUBindGroupLayout;
    sceneUniformsBindGroup: GPUBindGroup;

    depthFormat: GPUTextureFormat;
    depthTexture: GPUTexture;
    depthTextureView: GPUTextureView;

    pipeline: GPURenderPipeline;
    sceneBundle: GPURenderBundle | undefined;

    depthPrepass: boolean;
    prepassPipeline: GPURenderPipeline | undefined;
    prepassBundle: GPURenderBundle | undefined;

    constructor(stage: Stage) {
        super(stage);

        // TODO-2: initialize layouts, pipelines, textures, etc. needed for Forward+ here
        this.depthPrepass = settings.depthPrepass;
        // culling needs the prepass depth
        const cullingMode = this.depthPrepass ? settings.clusterCulling : clusterCullingModes.off;
        const culling = cullingMode !== clusterCullingModes.off;

        this.sceneUniformsBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "forward+ scene uniforms bind group layout",
            entries: [
                { // cameraUniforms
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: "uniform" }
                },
                { // lightSet
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" }
                },
                { // clusterSet
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" }
                }
            ]
        });

        this.sceneUniformsBindGroup = renderer.device.createBindGroup({
            label: "forward+ scene uniforms bind group",
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

        this.depthFormat = this.depthPrepass ? "depth32float" : "depth24plus";
        this.depthTexture = renderer.device.createTexture({
            size: [renderer.canvas.width, renderer.canvas.height],
            format: this.depthFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.depthTextureView = this.depthTexture.createView();

        const pipelineLayout = renderer.device.createPipelineLayout({
            label: "forward+ pipeline layout",
            bindGroupLayouts: [
                this.sceneUniformsBindGroupLayout,
                renderer.modelBindGroupLayout,
                renderer.materialBindGroupLayout
            ]
        });
        const vertexState: GPUVertexState = {
            module: renderer.device.createShaderModule({
                label: "naive vert shader (forward+)",
                code: shaders.naiveVertSrc
            }),
            buffers: [ renderer.vertexBufferLayout ]
        };

        this.pipeline = renderer.device.createRenderPipeline({
            label: "forward+ pipeline",
            layout: pipelineLayout,
            depthStencil: this.depthPrepass ? {
                depthWriteEnabled: false,
                depthCompare: "equal",
                format: this.depthFormat
            } : {
                depthWriteEnabled: true,
                depthCompare: "less",
                format: this.depthFormat
            },
            vertex: vertexState,
            fragment: {
                module: renderer.device.createShaderModule({
                    label: "forward+ frag shader",
                    code: shaders.forwardPlusFragSrc
                }),
                constants: {
                    alphaTest: this.depthPrepass ? 0 : 1,
                    clusterFromDepth: culling ? 1 : 0
                },
                targets: [
                    {
                        format: renderer.canvasFormat
                    }
                ]
            }
        });
        this.sceneBundle = this.createSceneBundle(this.pipeline, this.sceneUniformsBindGroup,
            [renderer.canvasFormat], this.depthFormat);

        if (this.depthPrepass) {
            this.prepassPipeline = renderer.device.createRenderPipeline({
                label: "forward+ depth prepass pipeline",
                layout: pipelineLayout,
                depthStencil: {
                    depthWriteEnabled: true,
                    depthCompare: "less",
                    format: this.depthFormat
                },
                vertex: vertexState,
                fragment: {
                    module: renderer.device.createShaderModule({
                        label: "depth prepass frag shader",
                        code: shaders.depthPrepassFragSrc
                    }),
                    targets: []
                }
            });
            this.prepassBundle = this.createSceneBundle(this.prepassPipeline, this.sceneUniformsBindGroup,
                [], this.depthFormat);
        }

        this.initClusterCulling(this.depthTextureView, cullingMode);
        this.initPostProcess(this.depthTextureView);
    }

    override draw() {
        // TODO-2: run the Forward+ rendering pass:
        // - run the clustering compute shader
        // - run the main rendering pass, using the computed clusters for efficient lighting
        const encoder = renderer.device.createCommandEncoder();
        const canvasTextureView = this.getColorTargetView();

        if (this.prepassPipeline) {
            const prepass = encoder.beginRenderPass({
                label: "forward+ depth prepass",
                colorAttachments: [],
                depthStencilAttachment: {
                    view: this.depthTextureView,
                    depthClearValue: 1.0,
                    depthLoadOp: "clear",
                    depthStoreOp: "store"
                },
                timestampWrites: this.profiler.timestampWrites("prepass")
            });
            this.drawScene(prepass, this.prepassPipeline, this.sceneUniformsBindGroup, this.prepassBundle);
            prepass.end();
        }

        this.cullAndClusterLights(encoder);

        const renderPass = encoder.beginRenderPass({
            label: "forward+ render pass",
            colorAttachments: [
                {
                    view: canvasTextureView,
                    clearValue: [0, 0, 0, 0],
                    loadOp: "clear",
                    storeOp: "store"
                }
            ],
            depthStencilAttachment: {
                view: this.depthTextureView,
                depthClearValue: 1.0,
                depthLoadOp: this.depthPrepass ? "load" : "clear",
                depthStoreOp: "store"
            },
            timestampWrites: this.profiler.timestampWrites("shading")
        });
        this.drawScene(renderPass, this.pipeline, this.sceneUniformsBindGroup, this.sceneBundle);
        renderPass.end();

        this.runPostProcess(encoder);
        this.profiler.resolve(encoder);
        renderer.device.queue.submit([encoder.finish()]);
    }
}
