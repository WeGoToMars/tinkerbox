import AppKit
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

// One editable question and its latest answer.
struct QuestionDraft: Identifiable {
    let id = UUID()
    var type = QuestionType.noul
    var instructions: String
    var choices = "Refund request\nTechnical support\nOther"
    var levels = "Not urgent\nSomewhat urgent\nVery urgent"
    var answer: Answer?
    var elapsedMs: Double?  // its own pass; nil when it shared one

    init(instructions: String = "") { self.instructions = instructions }

    var criteria: [String] {
        guard type != .noul else { return [] }
        return (type == .choice ? choices : levels).components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }

    var validation: String? {
        if type == .choice && (criteria.count < 2 || Set(criteria).count != criteria.count) {
            return "Add at least two unique options."
        }
        if type == .score && !(2...10).contains(criteria.count) {
            return "Add 2–10 levels, ordered from lowest to highest."
        }
        return nil
    }

    var question: Question { Question(type: type, instructions: instructions, criteria: criteria) }
}

enum RunMode: String, CaseIterable {
    case parallel = "Parallel", sequential = "Sequential"
}

// A whole run: every question, in one shared pass or one pass each.
struct RunTotals {
    var elapsedMs = 0.0
    var tokens = 0
    var flops: Double? = 0  // nil once a pass did not report its compute
    var neuralEngineMs: Double? = 0

    mutating func add(_ result: ItemResult, elapsedMs: Double) {
        self.elapsedMs += elapsedMs
        tokens += result.tokens
        flops = flops.flatMap { total in result.flops.map { total + $0 } }
        neuralEngineMs = neuralEngineMs.flatMap { total in result.neuralEngineMs.map { total + $0 } }
    }
}

// The catalog and the one model the worker has loaded, shared by every task page.
@MainActor @Observable final class Demo {
    var catalog = Catalog()
    let storage = Storage()
    let downloads = Downloads()
    var model = Runtime.selectedModel(for: .decision)
    var ready: Bool { runtime.state.reply?.modelId == model }
    // The selected model's download, then the runtime loading it.
    var loading: Bool { downloads.contains(model) || runtime.state.loading }
    var status: String { downloads.contains(model) ? "Downloading model…" : runtime.state.status }
    var progress: Double? { downloads.contains(model) ? downloads.progress[model] : runtime.state.progress }
    var busy = false  // a request from the app is running
    var unloading = false
    var error: String?
    var setupMs: Double?
    var modelName: String?
    var cpuModel: String?
    var architecture: String?
    var questionTrees = false
    var settings: [ModelSetting] = []
    var info: [ModelInfo] = []
    var speechParameters: [SpeechParameter] = []
    var speechCapabilities: SpeechCapabilities?
    // Only values that differ from the defaults are kept.
    var settingValues: [String: Int] = [:] {
        didSet { Runtime.store(settings: settingValues, for: model) }
    }
    let runtime = Runtime.shared
    private var generation = 0

    var current: CatalogModel? { catalog.models.first { $0.id == model } }
    var compatibilityIssue: String? {
        current.flatMap { Catalog.availabilityIssue(for: $0, asset:
            ProcessInfo.processInfo.environment["TINKERBOX_ASSET"].map { URL(fileURLWithPath: $0) }) }
    }

    func isDownloaded(_ id: String) -> Bool { catalog.downloaded.contains(id) && !downloads.contains(id) }

    // The models a task can switch to without a download; the Models library lists the rest.
    func models(for task: ModelTask) -> [CatalogModel] {
        catalog.models.filter { model in
            model.kind == task && model.isSupported && (isDownloaded(model.id) || model.id == self.model)
        }
    }

    // Downloads run beside the runtime, so they never hold up requests or model switches.
    private func fetch(_ id: String) async throws {
        if let model = catalog.models.first(where: { $0.id == id }),
           let issue = Catalog.availabilityIssue(for: model, asset:
                ProcessInfo.processInfo.environment["TINKERBOX_ASSET"].map { URL(fileURLWithPath: $0) }) {
            throw DecisionError.invalid(issue)
        }
        // The worker loads a local bundle given at launch instead.
        if UserDefaults.standard.string(forKey: "asset.\(id)") != nil { return }
        guard !isDownloaded(id), ProcessInfo.processInfo.environment["TINKERBOX_ASSET"] == nil else { return }
        defer {
            catalog.scanDownloads()
            Task { await storage.refresh() }
        }
        try await downloads.fetch(id)
    }

    func load() async {
        // A model switch supersedes an earlier load; ignore that load's outcome.
        generation += 1
        let current = generation
        forgetLoadedModel()
        defer {
            if current == generation {
                catalog.scanDownloads()
                Task { await storage.refresh() }  // the load may have downloaded files
            }
        }
        do {
            try await fetch(model)
            guard current == generation else { return }
            let reply = try await runtime.start(model: model)
            guard current == generation else { return }
            setupMs = reply.setupMs
            modelName = reply.modelName
            cpuModel = reply.cpuModel
            architecture = reply.architecture
            questionTrees = reply.questionTrees == true
            settings = reply.settings ?? []
            info = reply.info ?? []
            speechParameters = reply.speechParameters ?? []
            speechCapabilities = reply.speechCapabilities
        } catch is CancellationError {
            // Its download was stopped to remove the model.
        } catch {
            if current == generation { self.error = error.localizedDescription }
        }
    }

    private func forgetLoadedModel() {
        error = nil
        modelName = nil
        cpuModel = nil
        architecture = nil
        settings = []
        info = []
        speechParameters = []
        speechCapabilities = nil
        settingValues = Runtime.settings(for: model)
    }

    // The previous model keeps serving Shortcuts until the chosen one has downloaded.
    func choose(_ choice: CatalogModel, asset: URL? = nil) async {
        guard let task = choice.kind, choice.isSupported, !storage.removing.contains(choice.id),
              choice.id != model || !ready || asset != nil else { return }
        generation += 1
        let current = generation
        model = choice.id  // reserve its files before yielding to the runtime
        forgetLoadedModel()  // its settings are not the chosen model's
        do { if asset == nil { try await fetch(choice.id) } } catch {
            if current == generation && !(error is CancellationError) { self.error = error.localizedDescription }
            return
        }
        guard current == generation else { return }
        await runtime.select(model: choice.id, for: task, asset: asset?.path)
        guard current == generation else { return }
        setupMs = nil
        await load()
    }

    func unload() async {
        busy = true
        unloading = true
        defer { busy = false; unloading = false }
        await runtime.unload()
    }

    // The model the runtime has loaded or is loading, and the selected one once downloaded, keep their files.
    func keeps(_ id: String) -> Bool { id == runtime.state.model || (id == model && isDownloaded(id)) }

    // Removing a model stops its download first. Outdated revisions are never loaded, so any model can lose them.
    func remove(_ id: String, outdated: Bool = false) async {
        if !outdated {
            guard !keeps(id) else { return }
            await downloads.stop(id)
            if UserDefaults.standard.string(forKey: "asset.\(id)") != nil {
                UserDefaults.standard.removeObject(forKey: "asset.\(id)")
                catalog.scanDownloads()
                return
            }
        }
        await storage.remove(id, outdated: outdated)
        catalog.scanDownloads()
    }
}

// The Decide page's input, questions and session timings; they outlive visits to other pages.
@MainActor @Observable final class DecisionPlayground {
    let demo: Demo
    var state = "I was charged twice this month. Please refund the duplicate payment."
    var questions = [QuestionDraft(instructions: QuestionType.noul.example)]
    var mode = RunMode.parallel
    var images: [URL] = []
    var error: String?
    var lastRun: RunTotals?
    var times: [Double] = []
    var preview: TokenCount?  // what the next run will read
    var layouts: [String: ImageLayout] = [:]  // by image path
    var previewError: String?

    init(demo: Demo) { self.demo = demo }

    // Everything that changes what a run reads; the preview refreshes when it does.
    struct PreviewInput: Hashable {
        let item: Item
        let questions: [Question]
        let mode: RunMode
        let settings: [String: Int]
        let model: String
        let ready: Bool
    }

    var previewInput: PreviewInput {
        PreviewInput(item: Item(state: state, images: images.map(\.path)), questions: questions.map(\.question),
                     mode: runMode, settings: demo.settingValues, model: demo.model, ready: demo.ready)
    }

    // Counts like `run`: one shared pass, or one pass per question.
    func updatePreview() async {
        let input = previewInput
        guard input.ready, canRun else {
            preview = nil
            previewError = nil
            return
        }
        do {
            let requests = input.mode == .parallel ? [Request(items: [input.item], questions: input.questions)]
                : input.questions.map { Request(items: [input.item], questions: [$0]) }
            var total = TokenCount()
            for request in requests { total = total + (try await demo.runtime.count(request)[0]) }
            guard input == previewInput else { return }  // edited meanwhile; a newer update follows
            preview = total
            layouts = Dictionary(uniqueKeysWithValues: zip(input.item.images, total.images))
            previewError = nil
        } catch {
            guard input == previewInput else { return }
            preview = nil
            previewError = error.localizedDescription
        }
    }

    // One request lets models without question trees share image encoding across their sequential passes.
    var runMode: RunMode { demo.questionTrees && questions.count > 1 ? mode : .parallel }

    var canRun: Bool {
        !demo.loading && !demo.busy && demo.compatibilityIssue == nil && !(isBlank(state) && images.isEmpty)
            && questions.allSatisfy { $0.validation == nil && !isBlank($0.instructions) }
    }

    // Answers and timings belong to the model that produced them.
    func reset() {
        clearAnswers()
        times = []
        error = nil
        if demo.current?.acceptsImages != true { images = [] }
    }

    func add(_ urls: [URL]) {
        images += urls.filter { url in
            UTType(filenameExtension: url.pathExtension)?.conforms(to: .image) == true && !images.contains(url)
        }
    }

    // Copied image files and pictures join the image strip, and text the input, wherever the focus is.
    // A focused text field pastes text itself. Returns whether the paste was handled.
    func paste(_ pasteboard: NSPasteboard, textFocused: Bool) -> Bool {
        guard !demo.busy else { return false }
        let acceptsImages = demo.current?.acceptsImages == true
        let options: [NSPasteboard.ReadingOptionKey: Any] = [
            .urlReadingFileURLsOnly: true, .urlReadingContentsConformToTypes: [UTType.image.identifier],
        ]
        // Finder also copies the file names as text, so files come first.
        if let files = pasteboard.readObjects(forClasses: [NSURL.self], options: options) as? [URL], !files.isEmpty {
            guard acceptsImages else { return false }
            add(files)
            return true
        }
        // Some apps copy a picture of their text too, so text comes before image data.
        if let text = pasteboard.string(forType: .string) {
            guard !textFocused else { return false }
            state = isBlank(state) ? text : state + "\n" + text
            return true
        }
        // The worker opens images by path, so pictures without a file are saved first.
        for (type, ext) in [(NSPasteboard.PasteboardType.png, "png"), (.tiff, "tiff")] {
            guard acceptsImages, let data = pasteboard.data(forType: type) else { continue }
            let folder = FileManager.default.temporaryDirectory.appending(path: "pasted")
            let url = folder.appending(path: "\(UUID().uuidString).\(ext)")
            do {
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                try data.write(to: url)
            } catch {
                self.error = error.localizedDescription
                return true
            }
            add([url])
            return true
        }
        return false
    }

    func addQuestion() {
        questions.append(QuestionDraft())
    }

    func clearAnswers() {
        lastRun = nil
        for index in questions.indices {
            questions[index].answer = nil
            questions[index].elapsedMs = nil
        }
    }

    // Editing is disabled while busy, so question indices stay stable for the whole run.
    func run() async {
        demo.busy = true
        error = nil
        clearAnswers()
        defer { demo.busy = false }
        let item = Item(state: state, images: images.map(\.path))
        var totals = RunTotals()
        do {
            if !demo.ready {
                await demo.load()
                guard demo.ready else { return }
            }
            if runMode == .parallel {
                let result = try await demo.runtime.decide(Request(items: [item], questions: questions.map(\.question)))
                for (index, answer) in result.results[0].answers.enumerated() { questions[index].answer = answer }
                totals.add(result.results[0], elapsedMs: result.elapsedMs)
            } else {
                // Answers appear as each pass finishes.
                for index in questions.indices {
                    let result = try await demo.runtime.decide(Request(items: [item], questions: [questions[index].question]))
                    questions[index].answer = result.results[0].answers[0]
                    questions[index].elapsedMs = result.elapsedMs
                    totals.add(result.results[0], elapsedMs: result.elapsedMs)
                }
            }
            lastRun = totals
            times.append(totals.elapsedMs)
        } catch {
            self.error = error.localizedDescription
        }
    }

    /// Rendered input tokens per second over the last run.
    var throughput: Double? {
        guard let lastRun, lastRun.elapsedMs > 0 else { return nil }
        return Double(lastRun.tokens) * 1000 / lastRun.elapsedMs
    }

    /// Counted FLOPs per second of Neural Engine time over the last run, in TFLOPS.
    var computeRate: Double? {
        guard let flops = lastRun?.flops, let milliseconds = lastRun?.neuralEngineMs, milliseconds > 0 else { return nil }
        return flops / milliseconds / 1e9
    }

    func percentile(_ fraction: Double) -> Double? {
        guard !times.isEmpty else { return nil }
        let sorted = times.sorted()
        let index = Double(sorted.count - 1) * fraction
        let lower = Int(index)
        let upper = min(lower + 1, sorted.count - 1)
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - Double(lower))
    }
}

/// Downsamples with ImageIO so previews are filtered rather than shrunk from full resolution by the GPU.
func thumbnail(_ url: URL, side: CGFloat) -> CGImage? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
          let width = properties[kCGImagePropertyPixelWidth] as? CGFloat,
          let height = properties[kCGImagePropertyPixelHeight] as? CGFloat else { return nil }
    // The preview fills a square, so the short edge needs `side` pixels.
    let options: [CFString: Any] = [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: side * max(width, height) / min(width, height),
    ]
    return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
}

/// The picture upright and resized to exactly `width` × `height`, as the image processor does; the aspect may change.
func resampled(_ url: URL, width: Int, height: Int) -> CGImage? {
    let options: [CFString: Any] = [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: 2 * max(width, height),
    ]
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary),
          let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    context.interpolationQuality = .high
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    return context.makeImage()
}

extension ImageLayout {
    var grid: String { cols * rows > 1 ? "\(cols)×\(rows) tiles" : "1 view" }

    // The share of the picture's resolution the model keeps, as a linear scale; upscaling counts as all.
    var keptPercent: String {
        let kept = min(1, (Double(size[0] * size[1]) / Double(source[0] * source[1])).squareRoot())
        return kept.formatted(.percent.precision(.fractionLength(0)))
    }

    // Resizing to the grid changes the aspect slightly; worth mentioning above 3%.
    var stretch: String? {
        let ratio = Double(size[0]) / Double(size[1]) / (Double(source[0]) / Double(source[1]))
        let amount = max(ratio, 1 / ratio) - 1
        guard amount > 0.03 else { return nil }
        return "stretched \(amount.formatted(.percent.precision(.fractionLength(0)))) \(ratio > 1 ? "wider" : "taller")"
    }
}

// Tile boundaries over a picture split into `cols` × `rows`.
struct TileGrid: Shape {
    let cols: Int
    let rows: Int

    func path(in rect: CGRect) -> Path {
        Path { path in
            for col in 1..<max(cols, 1) {
                let x = rect.minX + rect.width * CGFloat(col) / CGFloat(cols)
                path.move(to: CGPoint(x: x, y: rect.minY))
                path.addLine(to: CGPoint(x: x, y: rect.maxY))
            }
            for row in 1..<max(rows, 1) {
                let y = rect.minY + rect.height * CGFloat(row) / CGFloat(rows)
                path.move(to: CGPoint(x: rect.minX, y: y))
                path.addLine(to: CGPoint(x: rect.maxX, y: y))
            }
        }
    }
}

// What the model sees of one picture: the tiles it fills and its overview, at their real pixel size when they fit.
struct ModelView: View {
    let url: URL
    let layout: ImageLayout
    @State private var tiles: CGImage?
    @State private var overview: CGImage?
    @Environment(\.displayScale) private var displayScale

    var body: some View {
        let (width, height) = (CGFloat(layout.size[0]) / displayScale, CGFloat(layout.size[1]) / displayScale)
        let fit = min(1, 800 / width, 560 / height)  // the whole picture, without scrolling
        VStack(alignment: .leading, spacing: 10) {
            Text("What the model sees").font(.headline)
            HStack(alignment: .top, spacing: 12) {
                Group {
                    if let tiles {
                        Image(decorative: tiles, scale: displayScale).resizable()
                            .overlay { TileGrid(cols: layout.cols, rows: layout.rows).stroke(.tint, lineWidth: 1) }
                    } else {
                        Color.secondary.opacity(0.1)
                    }
                }
                .frame(width: width * fit, height: height * fit)
                if let overview {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Overview").font(.caption).foregroundStyle(.secondary)
                        Image(decorative: overview, scale: displayScale)
                    }
                }
            }
            let (source, size) = (layout.source, layout.size)
            Text(["\(source[0]) × \(source[1]) → \(size[0]) × \(size[1]) (\(layout.keptPercent))", layout.stretch,
                  fit < 1 ? "shown at \(Double(fit).formatted(.percent.precision(.fractionLength(0))))" : nil]
                 .compactMap { $0 }.joined(separator: " · "))
            Text("\(layout.grid)\(layout.overview == nil ? "" : " + overview") · "
                 + "\(layout.passes) vision \(layout.passes == 1 ? "pass" : "passes") · \(layout.tokens.formatted()) tokens")
                .foregroundStyle(.secondary)
        }
        .font(.callout.monospacedDigit())
        .padding(16)
        .task {
            tiles = resampled(url, width: layout.size[0], height: layout.size[1])
            if let size = layout.overview { overview = resampled(url, width: size[0], height: size[1]) }
        }
    }
}

func duration(_ milliseconds: Double?) -> String {
    guard let milliseconds else { return "—" }
    return milliseconds < 1000
        ? String(format: "%.1f ms", milliseconds)
        : String(format: "%.2f s", milliseconds / 1000)
}

enum Destination: Hashable {
    case task(ModelTask), models, storage
}

struct AppView: View {
    @State private var demo: Demo
    @State private var decisions: DecisionPlayground
    @State private var destination: Destination? = .task(.decision)

    init() {
        let demo = Demo()
        _demo = State(initialValue: demo)
        _decisions = State(initialValue: DecisionPlayground(demo: demo))
    }

    var body: some View {
        NavigationSplitView {
            List(selection: $destination) {
                Section("Tasks") {
                    ForEach(ModelTask.allCases, id: \.self) { task in
                        Label(task.title, systemImage: task.symbol).tag(Destination.task(task))
                    }
                }
                Section("Library") {
                    Label("Models", systemImage: "square.stack.3d.up").tag(Destination.models)
                    Label("Storage", systemImage: "internaldrive").tag(Destination.storage)
                }
            }
            .safeAreaInset(edge: .bottom, spacing: 0) {
                StorageSummary(storage: demo.storage) { destination = .storage }
            }
            .navigationSplitViewColumnWidth(min: 160, ideal: 180)
            .disabled(demo.busy)
        } detail: {
            Group {
                switch destination ?? .task(.decision) {
                case .task(.decision): DecisionView(playground: decisions) { destination = .models }
                case .task(.speech): SpeechView(demo: demo) { destination = .models }
                case .models: ModelsView(demo: demo)
                case .storage: StorageView(demo: demo)
                }
            }
            .safeAreaInset(edge: .bottom, spacing: 0) { SystemFooter() }
        }
        .task(id: destination) {
            if case .task(let task) = destination {
                let selected = Runtime.selectedModel(for: task)
                if demo.model == selected { if !demo.ready { await demo.load() } }
                else if let model = demo.catalog.models.first(where: { $0.id == selected }) { await demo.choose(model) }
            }
        }
        .task { await demo.catalog.refresh() }
        .task { await Updates.shared.check() }
        .onChange(of: demo.model) { decisions.reset() }
        .alert("Could not update storage", isPresented: Binding(get: { demo.storage.error != nil },
                                                                set: { if !$0 { demo.storage.error = nil } })) {
        } message: {
            Text(demo.storage.error ?? "")
        }
    }
}

// The loaded model's status, and a menu to switch between this task's downloaded models.
struct ModelToolbar: ToolbarContent {
    let demo: Demo
    let task: ModelTask
    @Binding var showingInspector: Bool
    let showModels: () -> Void

    var body: some ToolbarContent {
        ToolbarItem {
            if demo.unloading {
                Text("Unloading model…").foregroundStyle(.secondary)
            } else if demo.loading {
                HStack(spacing: 8) {
                    if let progress = demo.progress {
                        ProgressView(value: progress).frame(width: 80)
                        Text("\(demo.status) \(Int(progress * 100))%").monospacedDigit()
                    } else {
                        ProgressView().controlSize(.small)
                        Text(demo.status)
                    }
                }
                .foregroundStyle(.secondary)
                .fixedSize()
            } else if demo.ready {
                Label("Model ready", systemImage: "checkmark.circle.fill")
                    .labelStyle(.titleAndIcon)
                    .foregroundStyle(.green)
                    .fixedSize()
                    .help("Unloads after 5 minutes without inference.")
            } else if let issue = demo.compatibilityIssue {
                Label("Model unavailable", systemImage: "exclamationmark.circle")
                    .foregroundStyle(.secondary)
                    .help(issue)
            } else {
                Label("Model unloaded", systemImage: "moon")
                    .foregroundStyle(.secondary)
                    .help("Loads automatically on the next run. Unloads after 5 minutes without inference.")
            }
        }
        // Status text, not a control, so it sits outside the glass capsule.
        .sharedBackgroundVisibility(.hidden)
        ToolbarItem {
            Menu {
                Picker("Model for Decision and Shortcuts", selection: Binding(get: { demo.model }, set: { id in
                    guard let model = demo.catalog.models.first(where: { $0.id == id }) else { return }
                    Task { await demo.choose(model) }
                })) {
                    ForEach(demo.models(for: task)) { Text($0.name).tag($0.id) }
                }
                .pickerStyle(.inline)
                Divider()
                Button("Manage Models…", action: showModels)
            } label: {
                Label(demo.current.map { "Model: \($0.name)" } ?? "Select Model", systemImage: "square.stack.3d.up")
                    .labelStyle(.titleAndIcon)
            }
            .disabled(demo.busy)
            .help("Used by Decision and Shortcuts. Open to select or download a model.")
        }
        ToolbarItem {
            Button("Unload", systemImage: "eject") { Task { await demo.unload() } }
                .labelStyle(.titleAndIcon)
                .disabled(!demo.ready || demo.loading || demo.busy)
                .help("Unload the model from memory. The next run loads it again.")
        }
        ToolbarItem {
            Toggle("Model Settings", systemImage: "gearshape", isOn: $showingInspector)
                .help("Model settings")
        }
    }
}

// The loaded model's settings and fixed properties, beside a task page.
struct ModelInspector: View {
    let demo: Demo

    var body: some View {
        Form {
            Section("Settings") {
                if demo.settings.isEmpty {
                    Text(demo.loading ? "Loading settings…" : demo.ready ? "No model settings." : "Load the model to view settings.")
                        .foregroundStyle(.secondary)
                }
                ForEach(demo.settings) { setting in
                    let value = Binding {
                        demo.settingValues[setting.key] ?? setting.defaultValue
                    } set: {
                        demo.settingValues[setting.key] = $0 == setting.defaultValue ? nil : $0
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        Stepper(value: value, in: setting.min...setting.max, step: setting.step ?? 1) {
                            LabeledContent(setting.title, value: "\(value.wrappedValue)")
                        }
                        Text(setting.help).font(.caption).foregroundStyle(.secondary)
                    }
                }
                if !demo.settingValues.isEmpty {
                    Button("Reset to Defaults") { demo.settingValues = [:] }
                }
            }
            if !demo.info.isEmpty {
                Section("Model") {
                    ForEach(demo.info, id: \.label) { LabeledContent($0.label, value: $0.value) }
                }
            }
        }
        .formStyle(.grouped)
        .disabled(demo.busy)
        .inspectorColumnWidth(min: 240, ideal: 280, max: 360)
    }
}

struct ModelAvailability: View {
    let demo: Demo
    let showModels: () -> Void

    var body: some View {
        if let issue = demo.compatibilityIssue {
            HStack(alignment: .top, spacing: 12) {
                Label(issue, systemImage: "info.circle")
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Manage Models…", action: showModels)
            }
        }
    }
}

struct DecisionView: View {
    @Bindable var playground: DecisionPlayground
    let showModels: () -> Void
    @State private var importingImages = false
    @State private var textCollapsed = false
    @State private var pasteMonitor: Any?
    @State private var inspectedImage: URL?
    @AppStorage("showingInspector") private var showingInspector = false
    @Environment(\.displayScale) private var displayScale

    private var demo: Demo { playground.demo }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                ModelAvailability(demo: demo, showModels: showModels)
                if demo.compatibilityIssue == nil && !demo.ready && (demo.loading || demo.error != nil) {
                    HStack {
                        Text("First launch downloads Python, dependencies and model files.")
                            .font(.caption).foregroundStyle(.secondary)
                        Spacer()
                        Button("View log") { NSWorkspace.shared.open(Runtime.logURL) }
                        if !demo.loading {
                            Button("Retry") { Task { await demo.load() } }
                        }
                    }
                }

                VStack(alignment: .leading, spacing: 8) {
                    if demo.current?.acceptsImages == true {
                        imageInput
                    }
                    if playground.images.isEmpty {
                        Text("Input text").font(.headline)
                    } else {
                        // Text is optional with images, so the editor can fold away.
                        Button {
                            withAnimation { textCollapsed.toggle() }
                        } label: {
                            Label("Input text (optional)",
                                  systemImage: textCollapsed ? "chevron.right" : "chevron.down")
                                .font(.headline)
                        }
                        .buttonStyle(.plain)
                    }
                    if playground.images.isEmpty || !textCollapsed {
                        TextEditor(text: $playground.state)
                            .font(.body)
                            .scrollContentBackground(.hidden)
                            .padding(10)
                            .background(.background, in: RoundedRectangle(cornerRadius: 8))
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                            .frame(height: 110)
                            .accessibilityLabel("Input text")
                    }
                }
                .disabled(demo.busy)

                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Text("Questions").font(.headline)
                        Spacer()
                        Button("Add question", systemImage: "plus") { playground.addQuestion() }
                    }
                    ForEach($playground.questions) { $draft in
                        // Read through the binding here: inside removeAll it would overlap the mutation.
                        let id = draft.id
                        let number = (playground.questions.firstIndex { $0.id == id } ?? 0) + 1
                        QuestionCard(draft: $draft, number: number, removable: playground.questions.count > 1) {
                            playground.questions.removeAll { $0.id == id }
                        }
                    }
                }
                .disabled(demo.busy)

                HStack(spacing: 14) {
                    Button {
                        Task { await playground.run() }
                    } label: {
                        Label(demo.busy && !demo.unloading ? "Running…" : playground.questions.count == 1 ? "Run decision"
                              : "Run \(playground.questions.count) questions", systemImage: "play.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(!playground.canRun)
                    if playground.questions.count > 1 {
                        if demo.questionTrees {
                            Picker("Mode", selection: $playground.mode) {
                                ForEach(RunMode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                            }
                            .pickerStyle(.segmented)
                            .labelsHidden()
                            .fixedSize()
                            .disabled(demo.busy)
                            .help("Parallel shares one pass across questions. Sequential runs each question separately.")
                        } else {
                            Text("One pass per question")
                                .font(.caption).foregroundStyle(.secondary)
                                .help("Each question runs sequentially and reads the full input again.")
                        }
                    }
                    if demo.busy && !demo.unloading { LogoActivity().frame(width: 36, height: 36).transition(.opacity) }
                    if let error = playground.previewError {
                        Text(error).font(.caption).foregroundStyle(.red).lineLimit(2)
                    } else if let preview = playground.preview {
                        Text(preview.tiles > 0 ? "\(preview.tokens.formatted()) tokens · \(preview.tiles) image tiles"
                             : "\(preview.tokens.formatted()) tokens")
                            .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                            .help("Input for the next run; each image tile uses one vision pass")
                    }
                    Spacer()
                    if let setup = demo.setupMs {
                        Text("Model setup \(duration(setup))").font(.caption).foregroundStyle(.secondary)
                    }
                }
                .onChange(of: playground.runMode) { playground.times = [] }
                .onChange(of: demo.settingValues) { playground.times = [] }
                .task(id: playground.previewInput) {
                    // Wait for typing to pause before counting.
                    try? await Task.sleep(for: .milliseconds(300))
                    if !Task.isCancelled { await playground.updatePreview() }
                }

                if let error = playground.error ?? demo.error, demo.compatibilityIssue == nil {
                    Label(error, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.red).textSelection(.enabled)
                }

                HStack(spacing: 40) {
                    metric("Last run", duration(playground.lastRun?.elapsedMs))
                    metric("Compute", playground.computeRate.map { String(format: "%.1f TFLOPS", $0) } ?? "—")
                        .help(playground.lastRun?.neuralEngineMs.map {
                            "Matmul, convolution and attention FLOPs counted from the model graph, including padding, divided by \(duration($0)) of Neural Engine time"
                        } ?? "Neural Engine compute rate of the last run")
                    metric("Throughput", playground.throughput.map { String(format: "%.0f tok/s", $0) } ?? "—")
                    metric("Median", duration(playground.percentile(0.5)))
                    metric("p95", duration(playground.percentile(0.95)))
                    Spacer()
                    VStack(alignment: .trailing, spacing: 5) {
                        Text("Runs").font(.caption).foregroundStyle(.secondary)
                        Text("\(playground.times.count)").font(.title3.monospacedDigit())
                    }
                }
                .help(playground.lastRun.map { "\($0.tokens) rendered input tokens in the last run" } ?? "")
            }
            .padding(28)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(minWidth: 650, minHeight: 650)
        .background(.background.secondary)
        .navigationTitle(ModelTask.decision.title)
        .navigationSubtitle([demo.modelName, demo.cpuModel, demo.architecture].compactMap { $0 }.joined(separator: " · "))
        .toolbar { ModelToolbar(demo: demo, task: .decision, showingInspector: $showingInspector, showModels: showModels) }
        .inspector(isPresented: $showingInspector) { ModelInspector(demo: demo) }
        .fileImporter(isPresented: $importingImages, allowedContentTypes: [.image], allowsMultipleSelection: true) {
            if case .success(let urls) = $0 { playground.add(urls) }
        }
        // ⌘V anywhere on the page, not only in a focused field; panels and sheets paste as usual.
        .onAppear {
            pasteMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
                guard event.modifierFlags.intersection(.deviceIndependentFlagsMask) == .command,
                      event.charactersIgnoringModifiers == "v", let window = event.window,
                      !(window is NSPanel), window.attachedSheet == nil else { return event }
                return playground.paste(.general, textFocused: window.firstResponder is NSText) ? nil : event
            }
        }
        .onDisappear {
            if let pasteMonitor { NSEvent.removeMonitor(pasteMonitor) }
        }
    }

    private var imageInput: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Images").font(.headline)
            ScrollView(.horizontal) {
                HStack(alignment: .top, spacing: 10) {
                    ForEach(playground.images, id: \.self) { url in
                        let layout = playground.layouts[url.path]
                        VStack(spacing: 4) {
                            ZStack(alignment: .topTrailing) {
                                Button {
                                    inspectedImage = layout == nil ? nil : url
                                } label: {
                                    Group {
                                        if let image = thumbnail(url, side: 72 * displayScale) {
                                            Image(decorative: image, scale: displayScale).resizable().scaledToFill()
                                                .overlay {
                                                    if let layout {
                                                        TileGrid(cols: layout.cols, rows: layout.rows)
                                                            .stroke(.white.opacity(0.9), lineWidth: 1).shadow(radius: 1)
                                                    }
                                                }
                                        } else {
                                            Image(systemName: "photo").foregroundStyle(.secondary)
                                        }
                                    }
                                    .frame(width: 72, height: 72)
                                    .clipShape(RoundedRectangle(cornerRadius: 8))
                                }
                                .buttonStyle(.plain)
                                .help(layout == nil ? url.lastPathComponent : "\(url.lastPathComponent): show what the model sees")
                                .popover(isPresented: Binding { inspectedImage == url } set: { if !$0 { inspectedImage = nil } }) {
                                    if let layout { ModelView(url: url, layout: layout) }
                                }
                                Button {
                                    playground.images.removeAll { $0 == url }
                                } label: {
                                    Image(systemName: "xmark.circle.fill").symbolRenderingMode(.palette)
                                        .foregroundStyle(.white, .black.opacity(0.6))
                                }
                                .buttonStyle(.plain)
                                .padding(4)
                                .accessibilityLabel("Remove \(url.lastPathComponent)")
                            }
                            // The grid and how much resolution the model keeps.
                            Text(layout.map { "\($0.cols)×\($0.rows) · \($0.keptPercent)" } ?? " ")
                                .font(.caption2.monospacedDigit()).foregroundStyle(.secondary)
                        }
                    }
                    Button {
                        importingImages = true
                    } label: {
                        VStack(spacing: 4) {
                            Image(systemName: "photo.badge.plus").font(.title3)
                            Text("Add").font(.caption)
                        }
                        .frame(width: 72, height: 72)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Add images")
                    .background(.background, in: RoundedRectangle(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary, style: StrokeStyle(dash: [4])))
                }
            }
            .scrollIndicators(.never)
            .dropDestination(for: URL.self) { urls, _ in
                playground.add(urls)
                return true
            }
            Text("Drop or paste images here.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(.bottom, 6)
    }

    private func metric(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.title3.monospacedDigit())
        }
    }
}

// One question's editor, with its answer underneath once it has run.
struct QuestionCard: View {
    @Binding var draft: QuestionDraft
    let number: Int
    let removable: Bool
    let remove: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Text("\(number)").font(.headline.monospacedDigit()).foregroundStyle(.secondary)
                Picker("Type", selection: $draft.type) {
                    ForEach(QuestionType.allCases, id: \.self) { Text($0.title).tag($0) }
                }
                .labelsHidden()
                .fixedSize()
                TextField(draft.type.example, text: $draft.instructions)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel("Question \(number) instructions")
                if removable {
                    Button("Remove question \(number)", systemImage: "xmark", action: remove)
                        .labelStyle(.iconOnly)
                        .buttonStyle(.borderless)
                }
            }
            .onChange(of: draft.type) { old, type in
                // Swap the example along with the type, but keep text the user wrote.
                if draft.instructions == old.example { draft.instructions = type.example }
                draft.answer = nil
                draft.elapsedMs = nil
            }
            if draft.type != .noul {
                Text(draft.type == .choice ? "Options · one per line" : "Levels · one per line, lowest to highest")
                    .font(.subheadline.weight(.semibold))
                TextEditor(text: draft.type == .choice ? $draft.choices : $draft.levels)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .padding(8)
                    .background(.background.secondary, in: RoundedRectangle(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                    .frame(height: 80)
                    .accessibilityLabel(draft.type == .choice ? "Question \(number) options" : "Question \(number) levels")
                if let validation = draft.validation {
                    Text(validation).font(.caption).foregroundStyle(.red)
                } else if draft.type == .score {
                    Text("Levels start at 0. The score is a probability-weighted average.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            if let answer = draft.answer {
                Divider()
                HStack {
                    Text(answer.type == .score ? "Expected score" : "Answer").foregroundStyle(.secondary)
                    Spacer()
                    if let score = answer.score {
                        Text(String(format: "%.2f / %d", score, answer.probabilities.count - 1)).font(.title3.bold())
                    } else if let choice = answer.choice {
                        Text(choice).font(.title3.bold()).textSelection(.enabled)
                    } else if let probability = answer.noul {
                        Text(probability >= 0.5 ? "Yes" : "No").font(.title3.bold())
                    }
                    if let elapsed = draft.elapsedMs {
                        Text(duration(elapsed)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                            .frame(minWidth: 60, alignment: .trailing)
                    }
                }
                ForEach(answer.probabilities.indices, id: \.self) { index in
                    probabilityRow(answer.probabilities[index])
                }
            }
        }
        .padding(14)
        .background(.background, in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(.quaternary))
    }

    private func probabilityRow(_ probability: Probability) -> some View {
        HStack(spacing: 12) {
            Text(probability.label).frame(width: 180, alignment: .leading)
            ProgressView(value: probability.value)
            Text(String(format: "%.1f%%", probability.value * 100))
                .monospacedDigit().frame(width: 65, alignment: .trailing)
        }
    }
}

// Every catalog model, grouped by task, with downloads in progress shown in their row.
struct ModelsView: View {
    let demo: Demo
    @State private var removalError: String?
    @State private var pending: CatalogModel?
    @Environment(\.openURL) private var openURL

    // In catalog order; the remote catalog can list tasks this app does not run yet.
    private var tasks: [String] {
        demo.catalog.models.map(\.task).reduce(into: []) { tasks, task in
            if !tasks.contains(task) { tasks.append(task) }
        }
    }

    var body: some View {
        List {
            ForEach(tasks, id: \.self) { task in
                Section(ModelTask(rawValue: task)?.heading ?? task.capitalized) {
                    ForEach(demo.catalog.models.filter { $0.task == task }) { model in
                        row(model).padding(.vertical, 6)
                    }
                }
            }
        }
        .frame(minWidth: 650, minHeight: 650)
        .navigationTitle("Models")
        .navigationSubtitle(demo.catalog.isRemote
                            ? "Catalog from github.com/WeGoToMars/tinkerbox"
                            : "Showing built-in catalog")
        .toolbar {
            Button("Refresh", systemImage: "arrow.clockwise") { Task { await demo.catalog.refresh() } }
        }
        .alert("Model files", isPresented: Binding(get: { removalError != nil }, set: { if !$0 { removalError = nil } })) {
        } message: { Text(removalError ?? "") }
        .confirmationDialog("Delete \(pending?.name ?? "")?", isPresented: Binding(
            get: { pending != nil }, set: { if !$0 { pending = nil } }), presenting: pending) { model in
            Button("Delete", role: .destructive) { Task { await demo.remove(model.id) } }
        } message: { model in
            if let stored = demo.storage.report?.models.first(where: { $0.id == model.id }) {
                Text(StorageView.removal(of: stored))
            } else {
                Text("Deletes the model's bundles, compiled copies and checkpoint. Using this model again requires a download.")
            }
        }
        .task { await demo.catalog.refresh() }
    }

    private func row(_ model: CatalogModel) -> some View {
        let downloaded = demo.isDownloaded(model.id)
        let downloading = demo.downloads.contains(model.id)
        let issue = Catalog.availabilityIssue(for: model)
        // Partial downloads count, so an interrupted one can be deleted.
        let stored = demo.catalog.downloaded.contains(model.id) || downloading
            || demo.storage.report?.models.contains { $0.id == model.id } == true
        let inputs = model.inputs?.map { $0 == "image" ? "Images" : $0.capitalized }.joined(separator: " + ")
        let details = [inputs, model.context.map { "\($0 / 1024)K-token context" },
                       model.downloadBytes.map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) },
                       model.license].compactMap { $0 }
        return HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .firstTextBaseline) {
                    Text(model.name).font(.headline)
                    Text(model.publisher).foregroundStyle(.secondary)
                }
                Text(model.summary)
                Text(details.joined(separator: " · ")).font(.caption).foregroundStyle(.secondary)
                if model.isSupported, let issue {
                    Text(issue).font(.caption).foregroundStyle(.secondary)
                }
            }
            Spacer()
            Menu("Hugging Face", systemImage: "arrow.up.right.square") {
                let repositories = [("Base model", model.baseModel), ("Original model", model.originalModel),
                                    (model.localOnly == true ? "Model source" : "Converted model", model.repo)]
                ForEach(repositories, id: \.0) { title, repository in
                    if let repository {
                        Button("\(title) · \(repository)") {
                            openURL(URL(string: "https://huggingface.co")!.appending(path: repository))
                        }
                    }
                }
            }
            .labelStyle(.iconOnly)
            .menuIndicator(.hidden)
            .menuStyle(.button)
            .buttonStyle(.borderless)
            .fixedSize()
            .help("Open on Hugging Face")
            if !model.isSupported {
                Text(Catalog.isNewer(model.minVersion, than: Catalog.appVersion)
                     ? "Requires Tinkerbox \(model.minVersion)" : "Update Tinkerbox to use this model")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                if issue != nil {
                    Button("Choose bundle…") { chooseBundle(model) }
                        .disabled(demo.busy || demo.storage.removing.contains(model.id))
                } else if model.id == demo.model && demo.loading {
                    VStack(alignment: .trailing, spacing: 4) {
                        Text(demo.status).font(.caption).foregroundStyle(.secondary)
                        ProgressView(value: demo.progress).frame(width: 120)
                    }
                } else if downloading {
                    VStack(alignment: .trailing, spacing: 4) {
                        Text("Downloading in background…").font(.caption).foregroundStyle(.secondary)
                        ProgressView(value: demo.downloads.progress[model.id]).frame(width: 120)
                    }
                } else if model.id == demo.model {
                    Label("Selected", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(demo.ready ? .green : .secondary)
                        .help("Selected for \(model.kind?.title ?? model.task)")
                    if model.localOnly == true {
                        Button("Choose bundle…") { chooseBundle(model) }.disabled(demo.busy)
                    }
                }
                if demo.storage.removing.contains(model.id) {
                    ProgressView().controlSize(.small)
                } else if stored && !demo.keeps(model.id) {
                    Button(role: .destructive) { pending = model } label: { Image(systemName: "trash") }
                        .buttonStyle(.borderless)
                        .help(downloading ? "Stop the download and delete its files" : "Delete model files and cached copies")
                        .disabled(!demo.storage.removing.isEmpty)
                }
                if model.id != demo.model && issue == nil {
                    Button(downloaded ? "Select" : model.localOnly == true ? "Choose bundle…" : "Download & Select") {
                        if model.localOnly == true && !downloaded { chooseBundle(model) }
                        else { Task { await demo.choose(model) } }
                    }
                    .disabled(demo.busy || demo.storage.removing.contains(model.id))
                }
            }
        }
    }
    private func chooseBundle(_ model: CatalogModel) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.message = "Choose the compiled model bundle directory containing model.json."
        if panel.runModal() == .OK, let url = panel.url {
            if let issue = Catalog.bundleIssue(at: url) { removalError = issue }
            else { Task { await demo.choose(model, asset: url) } }
        }
    }
}

// The newest published release, checked once per launch; nil until known, or offline.
@MainActor @Observable final class Updates {
    static let shared = Updates()
    static let repository = URL(string: "https://github.com/WeGoToMars/tinkerbox")!
    static let releases = repository.appending(path: "releases/latest")
    private(set) var latest: String?

    var available: String? { latest.flatMap { Catalog.isNewer($0, than: Catalog.appVersion) ? $0 : nil } }

    func check() async {
        guard latest == nil else { return }
        // Drafts and prereleases are not "latest".
        var request = URLRequest(url: URL(string: "https://api.github.com/repos/WeGoToMars/tinkerbox/releases/latest")!)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        struct Release: Decodable { let tag_name: String }
        guard let (data, response) = try? await URLSession.shared.data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let release = try? JSONDecoder().decode(Release.self, from: data) else { return }
        latest = release.tag_name.hasPrefix("v") ? String(release.tag_name.dropFirst()) : release.tag_name
    }
}

struct SystemFooter: View {
    @State private var thermalState = ProcessInfo.processInfo.thermalState
    @State private var lowPowerMode = ProcessInfo.processInfo.isLowPowerModeEnabled
    @Environment(\.openURL) private var openURL
    @Environment(\.colorScheme) private var colorScheme

    // GitHub's mark (Primer Octicons, MIT), drawn in the text color.
    private static let githubMark: NSImage = {
        let svg = """
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 \
            1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 \
            0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 \
            .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 \
            3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 \
            1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 \
            8-8Z"/></svg>
            """
        let image = NSImage(data: Data(svg.utf8)) ?? NSImage()
        image.isTemplate = true
        return image
    }()

    private var thermalDescription: String {
        switch thermalState {
        case .nominal: "Nominal"
        case .fair: "Fair"
        case .serious: "Serious"
        case .critical: "Critical"
        @unknown default: "Unknown"
        }
    }

    var body: some View {
        let system = ProcessInfo.processInfo
        let version = system.operatingSystemVersion
        let memory = ByteCountFormatter.string(fromByteCount: Int64(system.physicalMemory), countStyle: .memory)
        VStack(spacing: 0) {
            Divider()
            HStack(spacing: 8) {
                Text("\(memory) memory · macOS \(version.majorVersion).\(version.minorVersion).\(version.patchVersion)"
                     + " · Thermal: \(thermalDescription) · Low Power Mode: \(lowPowerMode ? "On" : "Off")")
                Spacer()
                if let newer = Updates.shared.available {
                    Button("Update to \(newer)") { openURL(Updates.releases) }
                        .buttonStyle(.plain)
                        .bold()
                        .foregroundStyle(colorScheme == .dark ? Color.white : Color.black)
                        .help("Download the new version from GitHub")
                }
                Text("Tinkerbox \(Catalog.appVersion)")
                Button { openURL(Updates.repository) } label: {
                    Image(nsImage: Self.githubMark).resizable().frame(width: 12, height: 12)
                }
                .buttonStyle(.plain)
                .help("Tinkerbox on GitHub")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(.horizontal, 28)
            .padding(.vertical, 12)
        }
        .background(.background)
        .onReceive(NotificationCenter.default.publisher(for: ProcessInfo.thermalStateDidChangeNotification)) { _ in
            thermalState = system.thermalState
        }
        .onReceive(NotificationCenter.default.publisher(for: .NSProcessInfoPowerStateDidChange)) { _ in
            lowPowerMode = system.isLowPowerModeEnabled
        }
    }
}

@MainActor final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    func applicationWillTerminate(_ notification: Notification) {
        Runtime.stop(Runtime.workerProcess)
        Downloads.processes.values.forEach { Runtime.stop($0) }
    }
}

@main struct TinkerboxDemoApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        Window("Tinkerbox", id: "demo") {
            AppView()
        }
        .defaultSize(width: 940, height: 820)
    }
}
