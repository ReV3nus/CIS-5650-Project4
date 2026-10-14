import Stats from 'stats.js';
import { GUI } from 'dat.gui';

import { initWebGPU, Renderer } from './renderer';
import { NaiveRenderer } from './renderers/naive';
import { ForwardPlusRenderer } from './renderers/forward_plus';
import { ClusteredDeferredRenderer } from './renderers/clustered_deferred';
import { VisibilityBufferRenderer } from './renderers/visibility_buffer';
import { settings, debugViews, clusterCullingModes, postProcessModes } from './settings';

import { setupLoaders, Scene } from './stage/scene';
import { Lights } from './stage/lights';
import { Camera } from './stage/camera';
import { Stage } from './stage/stage';

await initWebGPU();
setupLoaders();

let scene = new Scene();
await scene.loadGltf('./scenes/sponza/Sponza.gltf');

const camera = new Camera();
const lights = new Lights(camera);

const stats = new Stats();
stats.showPanel(0);
document.body.appendChild(stats.dom);

const gui = new GUI();
const numLightsController = gui.add(lights, 'numLights').min(1).max(Lights.maxNumLights).step(1).onChange(() => {
    lights.updateLightSetUniformNumLights();
});

const stage = new Stage(scene, lights, camera, stats);

var renderer: Renderer | undefined;

function setRenderer(mode: string) {
    renderer?.stop();

    switch (mode) {
        case renderModes.naive:
            renderer = new NaiveRenderer(stage);
            break;
        case renderModes.forwardPlus:
            renderer = new ForwardPlusRenderer(stage);
            break;
        case renderModes.clusteredDeferred:
            renderer = new ClusteredDeferredRenderer(stage);
            break;
        case renderModes.visibilityBuffer:
            renderer = new VisibilityBufferRenderer(stage);
            break;
    }
}

const renderModes = {
    naive: 'naive', forwardPlus: 'forward+', clusteredDeferred: 'clustered deferred', visibilityBuffer: 'visibility buffer'
};
let renderModeController = gui.add({ mode: renderModes.naive }, 'mode', renderModes);
renderModeController.onChange(setRenderer);

const rebuildRenderer = () => setRenderer(renderModeController.getValue());
gui.add(settings, 'animateLights').name('animate lights');
gui.add(settings, 'debugView', Object.keys(debugViews)).name('debug view');
gui.add(settings, 'postProcess', postProcessModes).name('post process').onChange(rebuildRenderer);
gui.add(settings, 'renderBundles').name('render bundles').onChange(rebuildRenderer);
gui.add(settings, 'clusterCulling', clusterCullingModes).name('cluster culling').onChange(rebuildRenderer);
const forwardPlusFolder = gui.addFolder('Forward+');
forwardPlusFolder.add(settings, 'depthPrepass').name('depth prepass').onChange(rebuildRenderer);
const deferredFolder = gui.addFolder('Clustered Deferred');
deferredFolder.add(settings, 'packedGBuffer').name('packed G-buffer').onChange(rebuildRenderer);
deferredFolder.add(settings, 'computeShading').name('compute shading').onChange(rebuildRenderer);

setRenderer(renderModeController.getValue());

const gpuTiming = { gpuFrameMs: 'n/a' };
gui.add(gpuTiming, 'gpuFrameMs').name('GPU frame (ms)').listen();
setInterval(() => {
    const frameMs = renderer?.profiler.medians()['frame'];
    gpuTiming.gpuFrameMs = frameMs === undefined ? 'n/a' : frameMs.toFixed(2);
}, 250);

// BENCHMARKING
// =================================

function waitFrames(numFrames: number) {
    return new Promise<void>(resolve => {
        const step = () => (--numFrames <= 0) ? resolve() : requestAnimationFrame(step);
        requestAnimationFrame(step);
    });
}

type BenchmarkConfig = { name: string, mode: string, settings?: Partial<typeof settings> };

const defaultSettings = { ...settings };
const defaultBenchmarkConfigs: BenchmarkConfig[] = [
    { name: 'naive', mode: renderModes.naive },
    { name: 'forward+', mode: renderModes.forwardPlus },
    { name: 'forward+ prepass', mode: renderModes.forwardPlus, settings: { depthPrepass: true } },
    { name: 'forward+ prepass + skip inactive', mode: renderModes.forwardPlus,
      settings: { depthPrepass: true, clusterCulling: clusterCullingModes.skipInactive } },
    { name: 'forward+ prepass + compaction', mode: renderModes.forwardPlus,
      settings: { depthPrepass: true, clusterCulling: clusterCullingModes.compact } },
    { name: 'deferred', mode: renderModes.clusteredDeferred },
    { name: 'deferred packed', mode: renderModes.clusteredDeferred, settings: { packedGBuffer: true } },
    { name: 'deferred compute', mode: renderModes.clusteredDeferred, settings: { computeShading: true } },
    { name: 'deferred packed + compute', mode: renderModes.clusteredDeferred,
      settings: { packedGBuffer: true, computeShading: true } },
    { name: 'visibility buffer', mode: renderModes.visibilityBuffer },
    { name: 'deferred + compaction', mode: renderModes.clusteredDeferred,
      settings: { clusterCulling: clusterCullingModes.compact } },
    { name: 'deferred packed + compute + compaction', mode: renderModes.clusteredDeferred,
      settings: { packedGBuffer: true, computeShading: true, clusterCulling: clusterCullingModes.compact } },
    { name: 'visibility buffer + compaction', mode: renderModes.visibilityBuffer,
      settings: { clusterCulling: clusterCullingModes.compact } }
];

function applyConfig(config: BenchmarkConfig) {
    Object.assign(settings, defaultSettings, config.settings ?? {});
    gui.updateDisplay();
    renderModeController.setValue(config.mode); // rebuilds the renderer
}

// median ms per pass for each config and light count; lights are frozen so every config sees the same layout
async function runBenchmark(configs = defaultBenchmarkConfigs, lightCounts = [500, 1000, 2000, 3000, 4000, 5000],
                            warmupFrames = 30, sampleFrames = 120, lightTime = 10000) {
    const savedSettings = { ...settings };
    const savedMode = renderModeController.getValue();
    const savedNumLights = lights.numLights;

    const results: { [key: string]: string | number }[] = [];
    for (const config of configs) {
        applyConfig(config);
        settings.animateLights = false;
        lights.setTime(lightTime);
        for (const numLights of lightCounts) {
            numLightsController.setValue(numLights);
            await waitFrames(warmupFrames);
            renderer!.profiler.clear();
            await waitFrames(sampleFrames);

            const medians = renderer!.profiler.medians();
            const row: { [key: string]: string | number } = { config: config.name, numLights: numLights };
            for (const name in medians) {
                row[name] = +medians[name].toFixed(3);
            }
            results.push(row);
        }
    }

    Object.assign(settings, savedSettings);
    gui.updateDisplay();
    renderModeController.setValue(savedMode);
    numLightsController.setValue(savedNumLights);

    console.table(results);
    return results;
}

// CPU time of draw(), averaged over many calls since performance.now() is too coarse for one
async function runCpuBenchmark(configs = defaultBenchmarkConfigs, numLights = 1000, iterations = 200, repeats = 5) {
    const results: { [key: string]: string | number }[] = [];
    for (const config of configs) {
        applyConfig(config);
        numLightsController.setValue(numLights);
        await waitFrames(30);

        const runs: number[] = [];
        for (let r = 0; r < repeats; r++) {
            const start = performance.now();
            for (let i = 0; i < iterations; i++) {
                renderer!.profiler.beginFrame();
                (renderer as any).draw();
            }
            runs.push((performance.now() - start) / iterations);
            await waitFrames(10);
        }
        runs.sort((a, b) => a - b);
        results.push({ config: config.name, cpuMs: +runs[Math.floor(repeats / 2)].toFixed(4) });
    }
    console.table(results);
    return results;
}

gui.add({ runBenchmark: () => runBenchmark() }, 'runBenchmark').name('run benchmark');

function freezeLights(time: number) {
    settings.animateLights = false;
    gui.updateDisplay();
    lights.setTime(time);
}

Object.assign(window, { runBenchmark, runCpuBenchmark, freezeLights });
