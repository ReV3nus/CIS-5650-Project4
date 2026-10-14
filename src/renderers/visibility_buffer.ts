import { Mat4, mat4 } from 'wgpu-matrix';
import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';
import { Material, Primitive } from '../stage/scene';
import { settings } from '../settings';

export class VisibilityBufferRenderer extends renderer.Renderer {
    static readonly textureArraySize = 1024; // Sponza's diffuse textures are 1024^2
    static readonly floatsPerObject = 20; // ObjectInfo: mat4 + 4 u32

    sceneUniformsBindGroupLayout: GPUBindGroupLayout;
    sceneUniformsBindGroup: GPUBindGroup;
    geometryBindGroupLayout: GPUBindGroupLayout;
    geometryBindGroup: GPUBindGroup;
    outputBindGroupLayout: GPUBindGroupLayout;

    objectsBuffer: GPUBuffer;
    indicesBuffer: GPUBuffer;
    verticesBuffer: GPUBuffer;
    diffuseTextureArray: GPUTexture;
    objectNumIndices: number[] = [];

    visibilityTexture: GPUTexture;
    visibilityTextureView: GPUTextureView;
    depthTexture: GPUTexture;
    depthTextureView: GPUTextureView;

    rasterPipeline: GPURenderPipeline;
    rasterBundle: GPURenderBundle | undefined;
    shadingPipeline: GPUComputePipeline;

    constructor(stage: Stage) {
        super(stage);

        // one object per primitive
        const objects: { modelMat: Mat4, primitive: Primitive, material: Material }[] = [];
        const materialLayers = new Map<Material, number>();
        let currentModelMat: Mat4 = mat4.identity();
        let currentMaterial: Material | undefined;
        this.scene.iterate(node => {
            currentModelMat = node.transform;
        }, material => {
            currentMaterial = material;
            if (!materialLayers.has(material)) {
                materialLayers.set(material, materialLayers.size);
            }
        }, primitive => {
            objects.push({ modelMat: currentModelMat, primitive: primitive, material: currentMaterial! });
        });

        const maxTriangles = Math.max(...objects.map(o => o.primitive.numIndices / 3));
        if (objects.length >= (1 << (32 - shaders.constants.visibilityTriangleBits))
            || maxTriangles >= (1 << shaders.constants.visibilityTriangleBits)) {
            throw new Error("scene too large for the visibility id encoding");
        }

        const objectData = new Float32Array(objects.length * VisibilityBufferRenderer.floatsPerObject);
        const objectDataU32 = new Uint32Array(objectData.buffer);
        let totalIndices = 0;
        let totalVertices = 0;
        objects.forEach((object, i) => {
            const base = i * VisibilityBufferRenderer.floatsPerObject;
            objectData.set(object.modelMat, base);
            objectDataU32[base + 16] = totalIndices;
            objectDataU32[base + 17] = totalVertices;
            objectDataU32[base + 18] = materialLayers.get(object.material)!;
            this.objectNumIndices.push(object.primitive.numIndices);
            totalIndices += object.primitive.numIndices;
            totalVertices += object.primitive.vertexBuffer.size / 32;
        });

        this.objectsBuffer = renderer.device.createBuffer({
            label: "visibility objects",
            size: objectData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        renderer.device.queue.writeBuffer(this.objectsBuffer, 0, objectData);
        this.indicesBuffer = renderer.device.createBuffer({
            label: "visibility indices",
            size: totalIndices * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.verticesBuffer = renderer.device.createBuffer({
            label: "visibility vertices",
            size: totalVertices * 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        const setupEncoder = renderer.device.createCommandEncoder({ label: "visibility buffer setup" });
        let indexByteOffset = 0;
        let vertexByteOffset = 0;
        for (const object of objects) {
            const indexBytes = object.primitive.numIndices * 4;
            setupEncoder.copyBufferToBuffer(object.primitive.indexBuffer, 0, this.indicesBuffer, indexByteOffset, indexBytes);
            setupEncoder.copyBufferToBuffer(object.primitive.vertexBuffer, 0, this.verticesBuffer, vertexByteOffset, object.primitive.vertexBuffer.size);
            indexByteOffset += indexBytes;
            vertexByteOffset += object.primitive.vertexBuffer.size;
        }

        this.diffuseTextureArray = this.buildTextureArray(setupEncoder, [...materialLayers.keys()]);
        renderer.device.queue.submit([setupEncoder.finish()]);

        this.visibilityTexture = renderer.device.createTexture({
            label: "visibility buffer",
            size: [renderer.canvas.width, renderer.canvas.height],
            format: "r32uint",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.visibilityTextureView = this.visibilityTexture.createView();
        this.depthTexture = renderer.device.createTexture({
            label: "visibility depth",
            size: [renderer.canvas.width, renderer.canvas.height],
            format: "depth32float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.depthTextureView = this.depthTexture.createView();

        const allStages = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE;
        this.sceneUniformsBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "visibility scene uniforms bind group layout",
            entries: [
                { binding: 0, visibility: allStages, buffer: { type: "uniform" } }, // cameraUniforms
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // lightSet
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } } // clusterSet
            ]
        });
        this.sceneUniformsBindGroup = renderer.device.createBindGroup({
            label: "visibility scene uniforms bind group",
            layout: this.sceneUniformsBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.camera.uniformsBuffer } },
                { binding: 1, resource: { buffer: this.lights.lightSetStorageBuffer } },
                { binding: 2, resource: { buffer: this.lights.clusterSetStorageBuffer } }
            ]
        });

        this.geometryBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "visibility geometry bind group layout",
            entries: [
                { binding: 0, visibility: allStages, buffer: { type: "read-only-storage" } }, // objects
                { binding: 1, visibility: allStages, buffer: { type: "read-only-storage" } }, // indices
                { binding: 2, visibility: allStages, buffer: { type: "read-only-storage" } }, // vertices
                { binding: 3, visibility: allStages, texture: { viewDimension: "2d-array" } }, // diffuseTexArray
                { binding: 4, visibility: allStages, sampler: {} } // diffuseSampler
            ]
        });
        this.geometryBindGroup = renderer.device.createBindGroup({
            label: "visibility geometry bind group",
            layout: this.geometryBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.objectsBuffer } },
                { binding: 1, resource: { buffer: this.indicesBuffer } },
                { binding: 2, resource: { buffer: this.verticesBuffer } },
                { binding: 3, resource: this.diffuseTextureArray.createView({ dimension: "2d-array" }) },
                {
                    binding: 4, resource: renderer.device.createSampler({
                        magFilter: "linear", minFilter: "linear", addressModeU: "repeat", addressModeV: "repeat"
                    })
                }
            ]
        });

        this.outputBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "visibility output bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint" } }, // visibilityTex
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "depth" } }, // depthTex
                { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba8unorm" } } // outputTex
            ]
        });

        const rasterModule = renderer.device.createShaderModule({
            label: "visibility raster shader",
            code: shaders.visibilitySrc
        });
        this.rasterPipeline = renderer.device.createRenderPipeline({
            label: "visibility raster pipeline",
            layout: renderer.device.createPipelineLayout({
                label: "visibility raster pipeline layout",
                bindGroupLayouts: [ this.sceneUniformsBindGroupLayout, this.geometryBindGroupLayout ]
            }),
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: "less",
                format: "depth32float"
            },
            vertex: { module: rasterModule, entryPoint: "vsMain" },
            fragment: { module: rasterModule, entryPoint: "fsMain", targets: [ { format: "r32uint" } ] }
        });

        if (settings.renderBundles) {
            const bundleEncoder = renderer.device.createRenderBundleEncoder({
                label: "visibility raster bundle",
                colorFormats: [ "r32uint" ],
                depthStencilFormat: "depth32float"
            });
            this.encodeRasterDraws(bundleEncoder);
            this.rasterBundle = bundleEncoder.finish();
        }

        this.shadingPipeline = renderer.device.createComputePipeline({
            label: "visibility shading pipeline",
            layout: renderer.device.createPipelineLayout({
                label: "visibility shading pipeline layout",
                bindGroupLayouts: [ this.sceneUniformsBindGroupLayout, this.geometryBindGroupLayout, this.outputBindGroupLayout ]
            }),
            compute: {
                module: renderer.device.createShaderModule({
                    label: "visibility shading shader",
                    code: shaders.visibilityShadingComputeSrc
                }),
                entryPoint: "main"
            }
        });

        this.initClusterCulling(this.depthTextureView);
        this.initPostProcess(this.depthTextureView);
    }

    private buildTextureArray(encoder: GPUCommandEncoder, materials: Material[]): GPUTexture {
        const size = VisibilityBufferRenderer.textureArraySize;
        const textureArray = renderer.device.createTexture({
            label: "diffuse texture array",
            size: [size, size, materials.length],
            format: "rgba8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
        });

        const blitModule = renderer.device.createShaderModule({ label: "texture blit shader", code: shaders.textureBlitSrc });
        const blitPipeline = renderer.device.createRenderPipeline({
            label: "texture blit pipeline",
            layout: "auto",
            vertex: { module: blitModule, entryPoint: "vsMain" },
            fragment: { module: blitModule, entryPoint: "fsMain", targets: [ { format: "rgba8unorm" } ] }
        });
        const sampler = renderer.device.createSampler({ magFilter: "linear", minFilter: "linear" });

        materials.forEach((material, layer) => {
            const pass = encoder.beginRenderPass({
                label: "texture array blit pass",
                colorAttachments: [{
                    view: textureArray.createView({ dimension: "2d", baseArrayLayer: layer, arrayLayerCount: 1 }),
                    loadOp: "clear",
                    storeOp: "store"
                }]
            });
            pass.setPipeline(blitPipeline);
            pass.setBindGroup(0, renderer.device.createBindGroup({
                layout: blitPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: material.diffuseTexture.createView() },
                    { binding: 1, resource: sampler }
                ]
            }));
            pass.draw(3);
            pass.end();
        });

        return textureArray;
    }

    // firstInstance carries the object id
    private encodeRasterDraws(pass: GPURenderPassEncoder | GPURenderBundleEncoder) {
        pass.setPipeline(this.rasterPipeline);
        pass.setBindGroup(shaders.constants.bindGroup_scene, this.sceneUniformsBindGroup);
        pass.setBindGroup(shaders.constants.bindGroup_visibility, this.geometryBindGroup);
        this.objectNumIndices.forEach((numIndices, objectId) => pass.draw(numIndices, 1, 0, objectId));
    }

    override draw() {
        const encoder = renderer.device.createCommandEncoder();
        const outputView = this.getColorTargetView();

        const rasterPass = encoder.beginRenderPass({
            label: "visibility raster pass",
            colorAttachments: [{
                view: this.visibilityTextureView,
                clearValue: [0xffffffff, 0, 0, 0], // invalidVisibilityId
                loadOp: "clear",
                storeOp: "store"
            }],
            depthStencilAttachment: {
                view: this.depthTextureView,
                depthClearValue: 1.0,
                depthLoadOp: "clear",
                depthStoreOp: "store"
            },
            timestampWrites: this.profiler.timestampWrites("visibility")
        });
        if (this.rasterBundle) {
            rasterPass.executeBundles([this.rasterBundle]);
        } else {
            this.encodeRasterDraws(rasterPass);
        }
        rasterPass.end();

        this.cullAndClusterLights(encoder);

        const outputBindGroup = renderer.device.createBindGroup({
            label: "visibility output bind group",
            layout: this.outputBindGroupLayout,
            entries: [
                { binding: 0, resource: this.visibilityTextureView },
                { binding: 1, resource: this.depthTextureView },
                { binding: 2, resource: outputView }
            ]
        });
        const computePass = encoder.beginComputePass({
            label: "visibility shading pass",
            timestampWrites: this.profiler.timestampWrites("shading")
        });
        computePass.setPipeline(this.shadingPipeline);
        computePass.setBindGroup(shaders.constants.bindGroup_scene, this.sceneUniformsBindGroup);
        computePass.setBindGroup(shaders.constants.bindGroup_visibility, this.geometryBindGroup);
        computePass.setBindGroup(shaders.constants.bindGroup_output, outputBindGroup);
        computePass.dispatchWorkgroups(
            Math.ceil(renderer.canvas.width / shaders.constants.screenWorkgroupSize),
            Math.ceil(renderer.canvas.height / shaders.constants.screenWorkgroupSize)
        );
        computePass.end();

        this.runPostProcess(encoder);
        this.profiler.resolve(encoder);
        renderer.device.queue.submit([encoder.finish()]);
    }
}
