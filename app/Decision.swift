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

// The Decide page's input, questions and latest run; they outlive visits to other pages.
@MainActor @Observable final class DecisionPlayground {
    let demo: Demo
    var input = AttributedString("I was charged twice this month. Please refund the duplicate payment.")
    var selection = AttributedTextSelection()
    var state: String {
        get { String(input.characters) }
        set {
            input = AttributedString(newValue)
            selection = AttributedTextSelection(insertionPoint: input.endIndex)
        }
    }
    var questions = [QuestionDraft(instructions: QuestionType.noul.example)]
    var mode = RunMode.parallel
    var images: [URL] = []
    var error: String?
    var lastRun: RunTotals?
    private var answersModel: String?  // the decision model the answers came from
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
    var runMode: RunMode { demo.loaded?.questionTrees == true && questions.count > 1 ? mode : .parallel }

    var canRun: Bool {
        !demo.loading && !demo.busy && demo.compatibilityIssue == nil && !(isBlank(state) && images.isEmpty)
            && questions.allSatisfy { $0.validation == nil && !isBlank($0.instructions) }
    }

    // Answers belong to the model that produced them; visiting another task's page keeps them.
    func modelChanged() {
        guard demo.current?.kind == .decision, demo.model != answersModel else { return }
        clearAnswers()
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
            answersModel = demo.model
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
}

/// Downsamples with ImageIO so previews are filtered rather than shrunk from full resolution by the GPU.
nonisolated func thumbnail(_ url: URL, side: CGFloat) -> CGImage? {
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
nonisolated func resampled(_ url: URL, width: Int, height: Int) -> CGImage? {
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

// A 72-point square preview, decoded off the main thread.
struct Thumbnail: View {
    let url: URL
    @State private var image: CGImage?
    @Environment(\.displayScale) private var displayScale

    var body: some View {
        Group {
            if let image {
                Image(decorative: image, scale: displayScale).resizable().scaledToFill()
            } else {
                Image(systemName: "photo").foregroundStyle(.secondary)
            }
        }
        .task(id: url) {
            let (url, side) = (url, 72 * displayScale)
            image = await Task.detached { thumbnail(url, side: side) }.value
        }
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
            let (url, layout) = (url, layout)
            (tiles, overview) = await Task.detached {
                (resampled(url, width: layout.size[0], height: layout.size[1]),
                 layout.overview.flatMap { resampled(url, width: $0[0], height: $0[1]) })
            }.value
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

    private var demo: Demo { playground.demo }

    var body: some View {
        TaskPage {
            ModelAvailability(demo: demo, showModels: showModels)
            if demo.compatibilityIssue == nil && !demo.ready && (demo.loading || demo.error != nil) {
                HStack {
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
                    TaskTextEditor(text: $playground.input, selection: $playground.selection, demo: demo)
                        .accessibilityLabel("Input text")
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
                if playground.questions.count > 1 && demo.loaded?.questionTrees == true {
                    Picker("Mode", selection: $playground.mode) {
                        ForEach(RunMode.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .fixedSize()
                    .disabled(demo.busy)
                    .help("Parallel shares one pass across questions. Sequential runs each question separately.")
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
                if let setup = demo.loaded?.setupMs {
                    Text("Model setup \(duration(setup))").font(.caption).foregroundStyle(.secondary)
                }
            }
            .task(id: playground.previewInput) {
                // Wait for typing to pause before counting.
                try? await Task.sleep(for: .milliseconds(300))
                if !Task.isCancelled { await playground.updatePreview() }
            }

            if let error = playground.error ?? demo.error, demo.compatibilityIssue == nil {
                Label(error, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red).textSelection(.enabled)
            }

            LazyVGrid(columns: [GridItem(.adaptive(minimum: 100), alignment: .leading)], alignment: .leading, spacing: 16) {
                metric("Last run", duration(playground.lastRun?.elapsedMs))
                metric("Compute", playground.computeRate.map { String(format: "%.1f TFLOPS", $0) } ?? "—")
                    .help(playground.lastRun?.neuralEngineMs.map {
                        "Matmul, convolution and attention FLOPs counted from the model graph, including padding, divided by \(duration($0)) of Neural Engine time"
                    } ?? "Neural Engine compute rate of the last run")
                metric("Throughput", playground.throughput.map { String(format: "%.0f tok/s", $0) } ?? "—")
            }
            .help(playground.lastRun.map { "\($0.tokens) rendered input tokens in the last run" } ?? "")
        } settings: {
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
        }
        .navigationTitle(ModelTask.decision.title)
        .navigationSubtitle(demo.loaded.map { "\($0.modelName) · \($0.cpuModel) · \($0.architecture)" } ?? "")
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
                                    Thumbnail(url: url)
                                        .overlay {
                                            if let layout {
                                                TileGrid(cols: layout.cols, rows: layout.rows)
                                                    .stroke(.white.opacity(0.9), lineWidth: 1).shadow(radius: 1)
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
                }
            }
            if let answer = draft.answer {
                Divider()
                HStack {
                    Text(answer.type == .score ? "Expected score" : "Answer").foregroundStyle(.secondary)
                        .help(answer.type == .score ? "The probability-weighted average level, counting from 0." : "")
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

