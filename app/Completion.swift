import Charts
import SwiftUI

struct CompletionPrompt: Identifiable {
    let id = UUID()
    var content: AttributedString
    var selection = AttributedTextSelection()
    var completion: TextCompletion?

    init(text: String = "Once upon a time,") { content = AttributedString(text) }
    var text: String { String(content.characters) }

    // Generated text stands out; typing after it does not.
    mutating func append(_ text: String) {
        var generated = AttributedString(text)
        generated.foregroundColor = .accentColor
        generated.backgroundColor = Color.accentColor.opacity(0.12)
        content += generated
        var typing = AttributeContainer()
        typing.foregroundColor = Color.primary
        typing.backgroundColor = Color.clear
        selection = AttributedTextSelection(insertionPoint: content.endIndex, typingAttributes: typing)
    }
}

// One prompt's output tokens so far, at a moment of a run.
struct TokenSample: Identifiable {
    let id = UUID()
    let prompt: Int
    let seconds: Double
    var tokens: Int
}

// The Text Completion page's prompts, options and latest results; they outlive visits to other pages.
@MainActor @Observable final class CompletionPlayground {
    let demo: Demo
    var prompts = [CompletionPrompt()]
    var maxNewTokens = 32
    var sampling = true
    var temperature = 0.8
    var topK = 40
    var topP = 0.95
    var seed: Int64?
    var batchSize = 0  // automatic
    var results: [TextCompletion] { prompts.compactMap(\.completion) }
    var elapsedMs: Double?
    var timeline: [TokenSample] = []  // of the latest run, in time order
    var counts: [TokenCount] = []
    var error: String?
    var previewError: String?
    private var resultsModel: String?  // the completion model the results came from

    init(demo: Demo) { self.demo = demo }

    var context: Int { demo.current?.context ?? 1024 }
    var batchSizes: [Int] { demo.loaded?.batchSizes ?? [1] }

    // Everything a token preview depends on; it refreshes when any of it changes.
    struct PreviewInput: Hashable {
        let prompts: [String]
        let model: String
        let ready: Bool
        let busy: Bool
    }

    var previewInput: PreviewInput {
        PreviewInput(prompts: prompts.map(\.text), model: demo.model,
                     ready: demo.ready, busy: demo.busy)
    }

    var canRun: Bool {
        !demo.busy && !demo.loading && demo.compatibilityIssue == nil && demo.current?.kind == .completion
            && previewError == nil
    }

    func modelChanged() {
        guard demo.current?.kind == .completion, demo.model != resultsModel else { return }
        for index in prompts.indices { prompts[index].completion = nil }
        elapsedMs = nil
        timeline = []
        counts = []
        error = nil
        previewError = nil
        batchSize = 0
    }

    func preview() async {
        let input = previewInput
        guard input.ready, !input.busy, demo.current?.kind == .completion else {
            counts = []
            previewError = nil
            return
        }
        do {
            let counts = try await demo.runtime.count(CompletionRequest(prompt: input.prompts, maxTokens: 0))
            guard input == previewInput else { return }
            withAnimation { self.counts = counts }
            previewError = counts.firstIndex { $0.tokens >= context }
                .map { "Prompt \($0 + 1) fills the \(context.formatted())-token context." }
        } catch {
            if input == previewInput { previewError = error.localizedDescription }
        }
    }

    func run() async {
        demo.busy = true
        defer { demo.busy = false }
        error = nil
        // A maximum, so each run generates at most what still fits after the longest prompt.
        let room = counts.count == prompts.count ? counts.map { context - $0.tokens }.min() : nil
        let request = CompletionRequest(prompt: prompts.map(\.text), maxTokens: min(maxNewTokens, room ?? .max),
            temperature: sampling ? temperature : 0, topK: sampling ? topK : 0, topP: sampling ? topP : 1,
            seed: sampling ? seed : nil, batchSize: batchSize == 0 ? nil : batchSize)
        let ids = prompts.map(\.id)
        let original = prompts.map(\.content)
        for index in prompts.indices { prompts[index].completion = nil }
        elapsedMs = nil
        // Prompts batched later stay at zero until their batch runs.
        timeline = ids.indices.map { TokenSample(prompt: $0, seconds: 0, tokens: 0) }
        let started = ContinuousClock.now
        do {
            let reply = try await demo.runtime.complete(request) { deltas in
                await self.receive(deltas, ids: ids, seconds: (ContinuousClock.now - started) / .seconds(1))
            }
            // The complete text replaces what streamed in.
            for (row, result) in reply.choices.enumerated() {
                if let index = prompts.firstIndex(where: { $0.id == ids[row] }) {
                    prompts[index].content = original[row]
                    prompts[index].append(result.text)
                    prompts[index].completion = result
                }
                // Streamed pieces can undercount; the reply's count is exact.
                if let last = timeline.lastIndex(where: { $0.prompt == row }) {
                    timeline[last].tokens = result.metrics.outputTokens
                }
            }
            elapsedMs = reply.metrics.elapsedMs
            resultsModel = demo.model
        } catch { self.error = error.localizedDescription }
    }

    private func receive(_ deltas: [CompletionDelta], ids: [UUID], seconds: Double) {
        for delta in deltas {
            if let index = prompts.firstIndex(where: { $0.id == ids[delta.index] }) { prompts[index].append(delta.text) }
            let tokens = timeline.last { $0.prompt == delta.index }?.tokens ?? 0
            timeline.append(TokenSample(prompt: delta.index, seconds: seconds, tokens: tokens + delta.tokens))
        }
    }
}

struct CompletionView: View {
    @Bindable var playground: CompletionPlayground
    let showModels: () -> Void
    @AppStorage("showingInspector") private var showingInspector = false
    private var demo: Demo { playground.demo }

    var body: some View {
        TaskPage {
            if demo.compatibilityIssue != nil {
                TaskSection { ModelAvailability(demo: demo, showModels: showModels) }
            }
            TaskSection("Prompts") {
                ForEach($playground.prompts) { $prompt in
                    let index = playground.prompts.firstIndex { $0.id == prompt.id } ?? 0
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Prompt \(index + 1)").font(.headline)
                            if playground.counts.indices.contains(index) {
                                Text("\(playground.counts[index].tokens) / \(playground.context) tokens")
                                    .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                                    .contentTransition(.numericText())
                            }
                            Spacer()
                            if playground.prompts.count > 1 {
                                Button("Remove prompt", systemImage: "minus.circle") {
                                    playground.prompts.removeAll { $0.id == prompt.id }
                                }.labelStyle(.iconOnly)
                            }
                        }
                        TaskTextEditor(text: $prompt.content, selection: $prompt.selection, demo: demo)
                            .accessibilityLabel("Prompt \(index + 1)")
                        if let result = prompt.completion {
                            Text("\(result.metrics.inputTokens) in · \(result.metrics.outputTokens) out · batch \(result.metrics.batchSize) · \(result.finishReason == "stop" ? "End of text" : "Token limit")")
                                .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                                .frame(maxWidth: .infinity, alignment: .trailing)
                        }
                    }.padding(.vertical, 4)
                }
                Button("Add Prompt", systemImage: "plus") {
                    playground.prompts.append(CompletionPrompt(text: ""))
                }
            }.disabled(demo.busy)

            TaskSection {
                HStack {
                    Button(playground.results.isEmpty ? "Generate" : "Continue", systemImage: "text.append") {
                        Task { await playground.run() }
                    }
                    .disabled(!playground.canRun)
                    .help(playground.results.isEmpty ? "" : "Generate more from the full text of each prompt.")
                    if demo.busy {
                        ProgressView().controlSize(.small)
                        Text("Generating…").foregroundStyle(.secondary)
                    } else if let elapsed = playground.elapsedMs {
                        let tokens = playground.results.reduce(0) { $0 + $1.metrics.outputTokens }
                        Text("\(tokens) output tokens · \((elapsed / 1000).formatted(.number.precision(.fractionLength(2)))) s · \((Double(tokens) * 1000 / max(elapsed, 1)).formatted(.number.precision(.fractionLength(1)))) tokens/s")
                            .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                    }
                }
                if let message = playground.error ?? playground.previewError ?? demo.error {
                    Text(message).foregroundStyle(.red).textSelection(.enabled)
                }
            }
            if !playground.timeline.isEmpty {
                TaskSection("Output tokens") { TokenTimeline(samples: playground.timeline) }
            }
        } settings: {
            TaskSection("Generation") {
                LabeledContent("Maximum new tokens") {
                    HStack {
                        TextField("Maximum new tokens", value: $playground.maxNewTokens, format: .number)
                            .labelsHidden().frame(width: 70)
                        Stepper("Maximum new tokens", value: $playground.maxNewTokens,
                                in: 1...max(1, playground.context - 1)).labelsHidden()
                    }
                }
                Picker("Batch size", selection: $playground.batchSize) {
                    Text("Automatic").tag(0)
                    ForEach(playground.batchSizes, id: \.self) { Text("Up to \($0)").tag($0) }
                }
                Picker("Decoding", selection: $playground.sampling) {
                    Text("Sampling").tag(true)
                    Text("Greedy").tag(false)
                }
                if playground.sampling {
                    LabeledContent("Temperature") {
                        TextField("Temperature", value: $playground.temperature, format: .number)
                            .labelsHidden().frame(width: 90)
                    }
                    LabeledContent("Top K") {
                        TextField("Top K", value: $playground.topK, format: .number)
                            .labelsHidden().frame(width: 90)
                    }
                    .help("0 samples from the whole vocabulary.")
                    LabeledContent("Top P") {
                        TextField("Top P", value: $playground.topP, format: .number)
                            .labelsHidden().frame(width: 90)
                    }
                    LabeledContent("Seed") {
                        TextField("Seed", value: $playground.seed, format: .number.grouping(.never), prompt: Text("Random"))
                            .labelsHidden().frame(maxWidth: 180)
                    }
                    .help("Replays the same text for each prompt.")
                }
            }.disabled(demo.busy)
        }
        .navigationTitle(ModelTask.completion.title)
        .toolbar { ModelToolbar(demo: demo, task: .completion, showingInspector: $showingInspector, showModels: showModels) }
        .inspector(isPresented: $showingInspector) { ModelInspector(demo: demo) }
        .focusedSceneValue(\.taskAction, TaskAction(title: playground.results.isEmpty ? "Generate" : "Continue",
                                                    enabled: playground.canRun) { Task { await playground.run() } })
        .task(id: playground.previewInput) {
            try? await Task.sleep(for: .milliseconds(250))  // wait for typing to pause
            if !Task.isCancelled { await playground.preview() }
        }
        .onChange(of: playground.context) {
            playground.maxNewTokens = min(playground.maxNewTokens, max(1, playground.context - 1))
        }
    }
}

// Each prompt's output tokens over a run; the slope is its speed, and batches show as lines rising together.
// Past the palette's colors, prompts fold into one total.
struct TokenTimeline: View {
    let samples: [TokenSample]
    @State private var selected: Double?

    private var series: [(name: String, samples: [TokenSample])] {
        let prompts = Set(samples.map(\.prompt)).sorted()
        if prompts.count <= Color.series.count {
            return prompts.map { prompt in ("Prompt \(prompt + 1)", samples.filter { $0.prompt == prompt }) }
        }
        var latest: [Int: Int] = [:]
        return [("All prompts", samples.map { sample in
            latest[sample.prompt] = sample.tokens
            return TokenSample(prompt: 0, seconds: sample.seconds, tokens: latest.values.reduce(0, +))
        })]
    }

    var body: some View {
        let series = series
        Chart {
            ForEach(series, id: \.name) { line in
                ForEach(line.samples) { sample in
                    LineMark(x: .value("Seconds", sample.seconds), y: .value("Tokens", sample.tokens))
                        .foregroundStyle(by: .value("Prompt", line.name))
                        .lineStyle(StrokeStyle(lineWidth: 2))
                }
            }
            if let selected {
                RuleMark(x: .value("Seconds", selected))
                    .foregroundStyle(.secondary.opacity(0.5))
                    .annotation(position: .top, overflowResolution: .init(x: .fit(to: .chart), y: .disabled)) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("\(selected.formatted(.number.precision(.fractionLength(2)))) s").bold()
                            ForEach(series, id: \.name) { line in
                                let tokens = line.samples.last { $0.seconds <= selected }?.tokens ?? 0
                                Text("\(line.name): \(tokens) tokens")
                            }
                        }
                        .font(.caption.monospacedDigit())
                        .padding(6)
                        .background(.background, in: .rect(cornerRadius: 6))
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
                    }
            }
        }
        .chartForegroundStyleScale(domain: series.map(\.name), range: Color.series.prefix(series.count))
        .chartLegend(series.count > 1 ? .visible : .hidden)
        .chartXAxisLabel("Seconds")
        .chartXSelection(value: $selected)
        .frame(height: 180)
    }
}
