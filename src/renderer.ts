import { Scene } from './stage/scene';
import { Lights } from './stage/lights';
import { Camera } from './stage/camera';
import { Stage } from './stage/stage';
import { GpuProfiler } from './profiler';
import { ToonPostProcess } from './postprocess';
import * as shaders from './shaders/shaders';
import { settings, postProcessModes, clusterCullingModes } from './settings';
import { ClusterCulling } from './stage/cluster_culling';

export var canvas: HTMLCanvasElement;
export var canvasFormat: GPUTextureFormat;
export var context: GPUCanvasContext;
export var device: GPUDevice;
export var canvasTextureView: GPUTextureView;

export var aspectRatio: number;
export const fovYDegrees = 45;

export var modelBindGroupLayout: GPUBindGroupLayout;
export var materialBindGroupLayout: GPUBindGroupLayout;

// CHECKITOUT: this function initializes WebGPU and also creates some bind group layouts shared by all the renderers
export async function initWebGPU() {
    canvas = document.getElementById("mainCanvas") as HTMLCanvasElement;

    const devicePixelRatio = window.devicePixelRatio;
    canvas.width = canvas.clientWidth * devicePixelRatio;
    canvas.height = canvas.clientHeight * devicePixelRatio;

    // e.g. ?resolution=1920x1080 for benchmarks
    const resolution = new URLSearchParams(window.location.search).get('resolution')?.match(/^(\d+)x(\d+)$/);
    if (resolution) {
        canvas.width = parseInt(resolution[1]);
        canvas.height = parseInt(resolution[2]);
    }

    aspectRatio = canvas.width / canvas.height;

    if (!navigator.gpu)
    {
        let errorMessageElement = document.createElement("h1");
        errorMessageElement.textContent = "This browser doesn't support WebGPU! Try using Google Chrome.";
        errorMessageElement.style.paddingLeft = '0.4em';
        document.body.innerHTML = '';
        document.body.appendChild(errorMessageElement);
        throw new Error("WebGPU not supported on this browser");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter)
    {
        throw new Error("no appropriate GPUAdapter found");
    }

    device = await adapter.requestDevice({
        requiredFeatures: adapter.features.has('timestamp-query') ? ['timestamp-query'] : []
    });
    device.lost.then(info => console.error(`WebGPU device lost (${info.reason}): ${info.message}`));
    device.addEventListener('uncapturederror', event => console.error('WebGPU error:', event.error.message));

    context = canvas.getContext("webgpu")!;
    // rgba8unorm so compute passes can write to the canvas
    canvasFormat = "rgba8unorm";
    context.configure({
        device: device,
        format: canvasFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING
    });

    console.log("WebGPU init successsful");

    modelBindGroupLayout = device.createBindGroupLayout({
        label: "model bind group layout",
        entries: [
            { // modelMat
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                buffer: { type: "uniform" }
            }
        ]
    });

    materialBindGroupLayout = device.createBindGroupLayout({
        label: "material bind group layout",
        entries: [
            { // diffuseTex
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                texture: {}
            },
            { // diffuseTexSampler
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                sampler: {}
            }
        ]
    });
}

export const vertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 32,
    attributes: [
        { // pos
            format: "float32x3",
            offset: 0,
            shaderLocation: 0
        },
        { // nor
            format: "float32x3",
            offset: 12,
            shaderLocation: 1
        },
        { // uv
            format: "float32x2",
            offset: 24,
            shaderLocation: 2
        }
    ]
};

export abstract class Renderer {
    protected scene: Scene;
    protected lights: Lights;
    protected camera: Camera;

    protected stats: Stats;

    readonly profiler = new GpuProfiler();

    private postProcess: ToonPostProcess | undefined;

    protected clusterCullingMode: string = clusterCullingModes.off;
    private clusterCulling: ClusterCulling | undefined;

    private prevTime: number = 0;
    private frameRequestId: number;

    constructor(stage: Stage) {
        this.scene = stage.scene;
        this.lights = stage.lights;
        this.camera = stage.camera;
        this.stats = stage.stats;

        this.frameRequestId = requestAnimationFrame((t) => this.onFrame(t));
    }

    stop(): void {
        cancelAnimationFrame(this.frameRequestId);
    }

    protected abstract draw(): void;

    protected initPostProcess(depthTextureView: GPUTextureView) {
        if (settings.postProcess === postProcessModes.toon) {
            this.postProcess = new ToonPostProcess(this.camera, depthTextureView);
        }
    }

    protected initClusterCulling(depthTextureView: GPUTextureView, mode: string = settings.clusterCulling) {
        this.clusterCullingMode = mode;
        if (mode !== clusterCullingModes.off) {
            this.clusterCulling = new ClusterCulling(this.lights, this.camera, depthTextureView);
        }
    }

    // with culling on, call after the depth buffer is filled
    protected cullAndClusterLights(encoder: GPUCommandEncoder) {
        if (this.clusterCulling) {
            const compact = this.clusterCullingMode === clusterCullingModes.compact;
            this.clusterCulling.run(encoder, compact,
                this.profiler.timestampWrites("cullMark"),
                compact ? this.profiler.timestampWrites("cullCompact") : undefined);
        }
        this.lights.doLightClustering(encoder, this.profiler.timestampWrites("clustering"), this.clusterCullingMode);
    }

    // offscreen texture when post-processing
    protected getColorTargetView(): GPUTextureView {
        return this.postProcess?.inputTextureView ?? context.getCurrentTexture().createView();
    }

    protected runPostProcess(encoder: GPUCommandEncoder) {
        this.postProcess?.run(encoder, context.getCurrentTexture().createView(), this.profiler.timestampWrites("post"));
    }

    private encodeSceneDraws(pass: GPURenderPassEncoder | GPURenderBundleEncoder,
                             pipeline: GPURenderPipeline, sceneBindGroup: GPUBindGroup) {
        pass.setPipeline(pipeline);
        pass.setBindGroup(shaders.constants.bindGroup_scene, sceneBindGroup);

        this.scene.iterate(node => {
            pass.setBindGroup(shaders.constants.bindGroup_model, node.modelBindGroup);
        }, material => {
            pass.setBindGroup(shaders.constants.bindGroup_material, material.materialBindGroup);
        }, primitive => {
            pass.setVertexBuffer(0, primitive.vertexBuffer);
            pass.setIndexBuffer(primitive.indexBuffer, 'uint32');
            pass.drawIndexed(primitive.numIndices);
        });
    }

    protected createSceneBundle(pipeline: GPURenderPipeline, sceneBindGroup: GPUBindGroup,
                                colorFormats: GPUTextureFormat[], depthStencilFormat: GPUTextureFormat) {
        if (!settings.renderBundles) {
            return undefined;
        }
        const bundleEncoder = device.createRenderBundleEncoder({
            label: "scene render bundle",
            colorFormats: colorFormats,
            depthStencilFormat: depthStencilFormat
        });
        this.encodeSceneDraws(bundleEncoder, pipeline, sceneBindGroup);
        return bundleEncoder.finish();
    }

    protected drawScene(pass: GPURenderPassEncoder, pipeline: GPURenderPipeline, sceneBindGroup: GPUBindGroup,
                        bundle: GPURenderBundle | undefined) {
        if (bundle) {
            pass.executeBundles([bundle]);
        } else {
            this.encodeSceneDraws(pass, pipeline, sceneBindGroup);
        }
    }

    // CHECKITOUT: this is the main rendering loop
    private onFrame(time: number) {
        if (this.prevTime == 0) {
            this.prevTime = time;
        }

        let deltaTime = time - this.prevTime;
        this.camera.onFrame(deltaTime);
        this.lights.onFrame(time);

        this.stats.begin();

        this.profiler.beginFrame();
        const cpuStart = performance.now();
        this.draw();
        this.profiler.record("cpu", performance.now() - cpuStart);
        this.profiler.readback();

        this.stats.end();

        this.prevTime = time;
        this.frameRequestId = requestAnimationFrame((t) => this.onFrame(t));
    }
}
