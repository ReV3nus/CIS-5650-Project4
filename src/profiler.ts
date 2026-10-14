import { device } from './renderer';

// per frame: beginFrame() -> timestampWrites() per pass -> resolve() -> submit -> readback()
export class GpuProfiler {
    static readonly maxPasses = 8;
    static readonly maxSamples = 240;

    readonly enabled: boolean;

    private querySet: GPUQuerySet | undefined;
    private resolveBuffer: GPUBuffer | undefined;
    private freeReadBuffers: GPUBuffer[] = [];
    private pending: { buffer: GPUBuffer, names: string[] } | undefined;
    private passNames: string[] = [];

    // ms per pass, plus "frame" and "cpu"
    private samples = new Map<string, number[]>();

    constructor() {
        this.enabled = device.features.has('timestamp-query');
        if (!this.enabled) {
            return;
        }

        this.querySet = device.createQuerySet({
            label: "profiler query set",
            type: "timestamp",
            count: GpuProfiler.maxPasses * 2
        });
        this.resolveBuffer = device.createBuffer({
            label: "profiler resolve buffer",
            size: GpuProfiler.maxPasses * 2 * 8,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
        });
    }

    beginFrame() {
        this.passNames = [];
    }

    timestampWrites(name: string) {
        if (!this.enabled || this.passNames.length >= GpuProfiler.maxPasses) {
            return undefined;
        }

        const idx = this.passNames.length;
        this.passNames.push(name);
        return {
            querySet: this.querySet!,
            beginningOfPassWriteIndex: idx * 2,
            endOfPassWriteIndex: idx * 2 + 1
        };
    }

    resolve(encoder: GPUCommandEncoder) {
        if (!this.enabled || this.passNames.length === 0) {
            return;
        }

        const numQueries = this.passNames.length * 2;
        const buffer = this.freeReadBuffers.pop() ?? device.createBuffer({
            label: "profiler readback buffer",
            size: GpuProfiler.maxPasses * 2 * 8,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        encoder.resolveQuerySet(this.querySet!, 0, numQueries, this.resolveBuffer!, 0);
        encoder.copyBufferToBuffer(this.resolveBuffer!, 0, buffer, 0, numQueries * 8);
        this.pending = { buffer: buffer, names: this.passNames };
    }

    readback() {
        const pending = this.pending;
        this.pending = undefined;
        if (!pending) {
            return;
        }

        pending.buffer.mapAsync(GPUMapMode.READ).then(() => {
            const times = new BigUint64Array(pending.buffer.getMappedRange());
            pending.names.forEach((name, i) => {
                this.record(name, Number(times[i * 2 + 1] - times[i * 2]) / 1e6);
            });
            const last = pending.names.length - 1;
            this.record("frame", Number(times[last * 2 + 1] - times[0]) / 1e6);

            pending.buffer.unmap();
            this.freeReadBuffers.push(pending.buffer);
        });
    }

    record(name: string, ms: number) {
        let arr = this.samples.get(name);
        if (!arr) {
            arr = [];
            this.samples.set(name, arr);
        }
        arr.push(ms);
        if (arr.length > GpuProfiler.maxSamples) {
            arr.shift();
        }
    }

    clear() {
        this.samples.clear();
    }

    medians(): { [name: string]: number } {
        const result: { [name: string]: number } = {};
        for (const [name, arr] of this.samples) {
            const sorted = [...arr].sort((a, b) => a - b);
            result[name] = sorted[Math.floor(sorted.length / 2)];
        }
        return result;
    }
}
