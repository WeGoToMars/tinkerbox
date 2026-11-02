import Foundation

enum QuestionType: String, CaseIterable, Codable, Sendable {
    case noul, choice, score

    var title: String {
        switch self {
        case .noul: "Yes/No"
        case .choice: "Choice"
        case .score: "Score"
        }
    }

    var example: String {
        switch self {
        case .noul: "Is this a refund request?"
        case .choice: "What is the topic of this message?"
        case .score: "How urgent is this request?"
        }
    }
}

func isBlank(_ text: String) -> Bool {
    text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
}

struct TinkerboxError: LocalizedError {
    let errorDescription: String?
    init(_ message: String) { errorDescription = message }
}

struct Question: Encodable, Hashable, Sendable {
    let type: QuestionType
    let instructions: String
    var criteria: [String] = []
}

struct Item: Encodable, Hashable, Sendable {
    let state: String
    var images: [String] = []  // file paths
}

// Every question is asked about every item. Several questions about one item share a single tree
// pass; one question about many items is packed into one batch. The worker validates requests.
struct Request: Encodable, Sendable {
    let items: [Item]
    let questions: [Question]
    var settings: [String: Int] = [:]  // filled in by Runtime from the model's saved values
    var count = false  // report tokens and image tiles instead of answering
}

struct Answer: Decodable, Sendable {
    let type: QuestionType
    let noul: Double?
    let choice: String?
    let score: Double?
    let confidence: Double?
    let probabilities: [Probability]  // in option order: Yes/No, the options, or the levels
}

struct Probability: Decodable, Sendable {
    let label: String
    let value: Double
}

// One item's answers, in the request's question order, and the tokens it read.
struct ItemResult: Decodable, Sendable {
    let answers: [Answer]
    let tokens: Int
    let flops: Double?  // matmul, convolution and attention FLOPs of the functions it ran
    let neuralEngineMs: Double?
}

struct Decisions: Decodable, Sendable {
    let results: [ItemResult]  // in the request's item order
    let elapsedMs: Double
}

// A run of input text that one or more tokens cover, in UTF-16 offsets.
struct TokenSpan: Decodable, Equatable, Sendable {
    let startUtf16: Int
    let endUtf16: Int
    let tokenIds: [Int]
}

// What one item of a request reads: input tokens and image tiles, each tile one vision pass.

struct TokenCount: Decodable, Equatable, Sendable {
    var tokens = 0
    var tiles = 0
    var images: [ImageLayout] = []  // in the item's image order

    // Several passes over the same item: they add up, and its pictures are laid out alike in each.
    static func + (a: Self, b: Self) -> Self {
        Self(tokens: a.tokens + b.tokens, tiles: a.tiles + b.tiles, images: a.images.isEmpty ? b.images : a.images)
    }
}

// How one picture reaches the model, as its image processor reports it. Sizes are width, height.
struct ImageLayout: Decodable, Equatable, Sendable {
    let source: [Int]  // upright, as given
    let cols: Int
    let rows: Int
    let size: [Int]  // resized to fill its tiles, or its single view
    let overview: [Int]?  // the whole picture at a single view's size, for split pictures
    let passes: Int
    let tokens: Int
}

// A value the model lets the user change, declared by its Python package.
struct ModelSetting: Decodable, Identifiable, Sendable {
    let key: String
    let title: String
    let help: String
    let min: Int
    let max: Int
    let step: Int?
    let defaultValue: Int

    var id: String { key }

    enum CodingKeys: String, CodingKey {
        case key, title, help, min, max, step, defaultValue = "default"
    }
}

// A fixed property of the model, shown beside its settings.
struct ModelInfo: Decodable, Sendable {
    let label: String
    let value: String
}

struct CompletionRequest: Encodable, Sendable {
    var prompt: [String]
    var maxTokens = 16
    var temperature = 1.0
    var topK = 0
    var topP = 1.0
    var seed: Int64?
    var batchSize: Int?
    var stream = false
    var count = false
}

struct CompletionMetrics: Decodable, Sendable {
    let inputTokens: Int
    let outputTokens: Int
    let batchSize: Int
}

struct TextCompletion: Decodable, Sendable {
    let text: String
    let finishReason: String
    let tokenIds: [Int]
    let metrics: CompletionMetrics
}

struct CompletionDelta: Decodable, Sendable {
    let index: Int
    let text: String
    let tokens: Int  // generated since the previous piece
}

struct CompletionChunk: Decodable, Sendable {
    let choices: [CompletionDelta]
}

struct TextCompletions: Decodable, Sendable {
    struct Timing: Decodable, Sendable { let elapsedMs: Double }
    let choices: [TextCompletion]  // in prompt order
    let metrics: Timing
}

// The loaded model, as its worker describes it once ready.
struct ReadyReply: Decodable, Sendable {
    let modelId: String
    let modelName: String
    let cpuModel: String
    let architecture: String
    let questionTrees: Bool  // several questions can share one pass
    let settings: [ModelSetting]
    let info: [ModelInfo]
    let batchSizes: [Int]  // text completion only
    let setupMs: Double
}

struct SpeechReply: Decodable, Sendable {
    let audio: [SpeechAudio]  // in item order
    let seed: Int64
    let cloneMs: Double
    let audioMs: Double
    let aneMs: Double
    let firstAudioMs: Double?  // streamed requests
}

enum SpeechValue: Codable, Sendable, Equatable {
    case number(Double), boolean(Bool), text(String)

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Bool.self) { self = .boolean(value) }
        else if let value = try? container.decode(String.self) { self = .text(value) }
        else { self = .number(try container.decode(Double.self)) }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .number(let value): try container.encode(value)
        case .boolean(let value): try container.encode(value)
        case .text(let value): try container.encode(value)
        }
    }
}

struct SpeechParameter: Decodable, Identifiable, Sendable {
    enum Kind: String, Decodable, Sendable { case int, float, bool, text, `enum` }
    let key: String
    let title: String
    let type: Kind
    let min: Double?
    let max: Double?
    let defaultValue: SpeechValue?
    let choices: [SpeechChoice]?
    let optional: Bool?
    let help: String?
    var id: String { key }
    enum CodingKeys: String, CodingKey {
        case key, title, type, min, max, choices, optional, help, defaultValue = "default"
    }
}

struct SpeechChoice: Decodable, Identifiable, Sendable {
    let value: String
    let title: String
    var id: String { value }
}

enum Requirement: String, Decodable, Sendable { case none, optional, required }

struct SpeechMode: Decodable, Identifiable, Sendable {
    let id: String
    let title: String
    let reference: Requirement?  // absent: none
    let referenceTranscript: Requirement?  // absent: used when available, recognized on-device otherwise
    let language: Bool?
    let instructionTitle: String?
    let instructionHint: String?
    let instructionRequired: Bool?
    let controls: [SpeechParameter]?
}

// A speech model's modes and controls, bundled at build time so pickers work before setup.
struct SpeechCapabilities: Decodable, Sendable {
    let modes: [SpeechMode]
    let controls: [SpeechParameter]  // per item
    let parameters: [SpeechParameter]  // per request
    let streaming: Bool?  // audio can play while one text is generated
}

struct SpeechItem: Encodable, Sendable {
    var text: String
    var language: String?
    var instruct: String?
    var refAudio: String?
    var refText: String?
    var voiceClonePrompt: String?
    var duration: Double?
    var speed: Double?
    var mode: String?
    var controls: [String: SpeechValue]?
}

struct SpeechRequest: Encodable, Sendable {
    var items: [SpeechItem]
    var parameters: [String: SpeechValue] = [:]
    var seed: Int64?
    var stream: Bool?  // set by Runtime when the caller plays pieces as they arrive
}

struct SpeechVoiceRequest: Encodable, Sendable {
    let operation = "create_voice"
    let refAudio: String
    var refText: String?
    var preprocessPrompt: Bool?  // nil uses the model's default
    var mode: String? = nil
}

struct SpeechAudio: Decodable, Sendable {
    let wav: Data
    let sampleRate: Int
    let duration: Double
}

// One streamed piece: mono little-endian PCM16; in order, the pieces join into the reply's WAV.
struct SpeechChunk: Decodable, Sendable {
    let pcm: Data
    let sampleRate: Int
}

@MainActor @Observable final class RuntimeState {
    nonisolated init() {}
    var reply: ReadyReply?
    var model: String?  // loaded, or being loaded
    var loading = false
    var status = "Preparing runtime…"
    var progress: Double?
}

extension Process {
    // Darwin calls a termination handler that is set after the process exited, too.
    func exited() async {
        await withCheckedContinuation { continuation in terminationHandler = { _ in continuation.resume() } }
    }
}

// Every worker message: status while loading, an error, a streamed piece, or else the reply itself.
private struct Envelope: Decodable {
    let status: String?
    let progress: Double?
    let error: String?
    let chunk: SpeechChunk?
    let completionChunk: CompletionChunk?
}

actor Runtime {
    static let shared = Runtime()
    @MainActor static var workerProcess: Process?
    nonisolated let state = RuntimeState()
    static let logURL = URL.libraryDirectory.appending(path: "Logs/tinkerbox-ane/demo.log")
    nonisolated static let encoder = {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        return encoder
    }()
    nonisolated static let decoder = {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }()
    private var process: Process?
    private var input: Pipe?
    private var replies: AsyncLineSequence<FileHandle.AsyncBytes>.AsyncIterator?
    private var readyReply: ReadyReply?
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private let idleTimeout: Duration
    private var idleDeadline: ContinuousClock.Instant?
    private var idleTask: Task<Void, Never>?

    init(idleTimeout: Duration = .seconds(300)) {
        self.idleTimeout = idleTimeout
        // A dead worker should fail the next write, not kill the app.
        signal(SIGPIPE, SIG_IGN)
    }

    var isConnected: Bool { readyReply != nil && process?.isRunning == true && replies != nil }

    // Workers run in their own process group, so this also stops their native hosts.
    nonisolated static func stop(_ process: Process?, signal: Int32 = SIGTERM) {
        guard let process, process.isRunning else { return }
        kill(-process.processIdentifier, signal)
    }

    // Each task remembers the UI's selection. Speech Shortcuts pass their own saved model per request.
    nonisolated static func selectedModel(for task: ModelTask) -> String {
        let runnable = Catalog.bundled.filter { $0.isSupported && $0.kind == task }.map(\.id)
        let stored = UserDefaults.standard.string(forKey: "model.\(task.rawValue)")
        return stored.flatMap { runnable.contains($0) ? $0 : nil } ?? runnable.first ?? ""
    }

    // Values changed in the app, per model; the UI and Shortcuts send them with every request.
    nonisolated static func settings(for model: String) -> [String: Int] {
        UserDefaults.standard.dictionary(forKey: "settings.\(model)") as? [String: Int] ?? [:]
    }

    nonisolated static func store(settings: [String: Int], for model: String) {
        UserDefaults.standard.set(settings, forKey: "settings.\(model)")
    }

    // Change the saved selection after pending requests finish, then release the previous model.
    func select(model: String, for task: ModelTask, asset: String? = nil) async {
        await acquire()
        defer { release() }
        UserDefaults.standard.set(model, forKey: "model.\(task.rawValue)")
        if let asset { Catalog.setLocalAsset(asset, for: model) }
        if readyReply?.modelId != model || asset != nil { await disconnect(graceful: true) }
    }

    // Queue behind active requests; unloading never interrupts inference or changes the selection.
    func unload() async {
        await acquire()
        defer { release() }
        await disconnect(graceful: true)
    }

    // Actors are reentrant at awaits. Hold the queue across startup AND a full
    // request/reply exchange so UI and Shortcuts calls cannot read each other's replies.
    private func acquire() async {
        idleTask?.cancel()
        idleTask = nil
        if busy {
            await withCheckedContinuation { waiters.append($0) }
        } else {
            busy = true
        }
    }

    private func release(activity: Bool = true) {
        if activity { idleDeadline = isConnected ? .now.advanced(by: idleTimeout) : nil }
        if !waiters.isEmpty {
            waiters.removeFirst().resume()
            return
        }
        busy = false
        guard isConnected, let deadline = idleDeadline else { return }
        idleTask = Task {
            do { try await ContinuousClock().sleep(until: deadline) } catch { return }
            guard !busy, idleDeadline == deadline else { return }
            await unload()
        }
    }

    func start(model: String) async throws -> ReadyReply {
        await acquire()
        defer { release() }
        try Task.checkCancellation()
        return try await prepare(model: model)
    }

    // One model is loaded at a time; a request for another one replaces it.
    private func prepare(model: String) async throws -> ReadyReply {
        if let readyReply, readyReply.modelId == model, isConnected { return readyReply }
        if let selected = Catalog.current(model),
           let issue = Catalog.availabilityIssue(for: selected, asset: Catalog.overrideAsset) {
            throw TinkerboxError(issue)
        }
        // The previous worker releases its Neural Engine memory before the next one loads.
        await disconnect(graceful: true)
        await MainActor.run {
            state.model = model
            state.loading = true
            state.status = "Preparing runtime…"
            state.progress = nil
        }
        do {
            return try await launch(model: model)
        } catch {
            await disconnect()
            throw error
        }
    }

    private func launch(model: String) async throws -> ReadyReply {
        let process = Process(), input = Pipe(), output = Pipe()
        Self.configure(process, python: ["-u", Bundle.main.url(forResource: "worker", withExtension: "py")!.path])
        process.environment!["TINKERBOX_MODEL"] = model
        process.environment!["TINKERBOX_CATALOG_UPDATES"] = Catalog.updatesEnvironment()
        if process.environment!["TINKERBOX_ASSET"] == nil, let asset = Catalog.localAsset(for: model) {
            process.environment!["TINKERBOX_ASSET"] = asset
        }
        try FileManager.default.createDirectory(at: Self.logURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: Self.logURL.path, contents: nil)
        let log = try FileHandle(forWritingTo: Self.logURL)
        defer { log.closeFile() }
        process.standardInput = input
        process.standardOutput = output
        process.standardError = log
        // A worker that crashes or is killed between requests stops counting as loaded.
        process.terminationHandler = { process in Task { await Runtime.shared.exited(process) } }
        self.process = process
        self.input = input
        replies = output.fileHandleForReading.bytes.lines.makeAsyncIterator()
        try await MainActor.run {
            try process.run()
            Runtime.workerProcess = process
        }
        // Let the child see EOF when the app closes its writer or exits.
        input.fileHandleForReading.closeFile()
        output.fileHandleForWriting.closeFile()
        while true {
            let line = try await nextLine()
            let envelope = try Self.decoder.decode(Envelope.self, from: line)
            if let error = envelope.error { throw TinkerboxError(error) }
            if let status = envelope.status {
                await MainActor.run {
                    state.status = status
                    state.progress = envelope.progress
                }
                continue
            }
            let reply = try Self.decoder.decode(ReadyReply.self, from: line)
            readyReply = reply
            await MainActor.run {
                state.reply = reply
                state.loading = false
                state.progress = nil
            }
            return reply
        }
    }

    private func exited(_ process: Process) async {
        if process === self.process { await disconnect() }
    }

    // Runs Python with these arguments in the app's environment: its pinned tinkerbox-ane and dependencies.
    nonisolated static func configure(_ process: Process, python arguments: [String]) {
        process.executableURL = Bundle.main.executableURL!.deletingLastPathComponent().appending(path: "uv")
        process.arguments = [
            "run", "--no-project", "--no-config", "--isolated", "--managed-python",
            "--python", "3.12", "--no-build", "--with-requirements",
            Bundle.main.url(forResource: "requirements", withExtension: "txt")!.path, "python",
        ] + arguments
        var environment = ProcessInfo.processInfo.environment
        for name in ["PYTHONPATH", "PYTHONHOME", "UV_INTERNAL__PARENT_INTERPRETER"] {
            environment.removeValue(forKey: name)
        }
        // Return freed large blocks to macOS: the allocator's cache otherwise keeps a speech request's
        // temporary buffers resident, about 1 GB, for a few percent of speed.
        environment["MallocLargeCache"] = "0"
        process.environment = environment
        process.currentDirectoryURL = URL.homeDirectory
    }

    func decide(_ request: Request) async throws -> Decisions {
        let model = Self.selectedModel(for: .decision)
        var request = request
        request.settings = Self.settings(for: model)
        return try await exchange(request, model: model)
    }

    // What each item would read, from the same prompt and settings as a run, without running the model.
    func count(_ request: Request) async throws -> [TokenCount] {
        let model = Self.selectedModel(for: .decision)
        var request = request
        request.settings = Self.settings(for: model)
        request.count = true
        return try await counts(request, model: model)
    }

    func count(_ request: CompletionRequest) async throws -> [TokenCount] {
        var request = request
        request.count = true
        return try await counts(request, model: Self.selectedModel(for: .completion))
    }

    private struct Counts: Decodable { let counts: [TokenCount] }

    private func counts(_ request: some Encodable & Sendable, model: String) async throws -> [TokenCount] {
        let reply: Counts = try await exchange(request, model: model, load: false)
        return reply.counts
    }

    // With `onText`, each prompt's new text arrives while it is generated, then the complete reply.
    func complete(_ request: CompletionRequest,
                  onText: (@Sendable ([CompletionDelta]) async -> Void)? = nil) async throws -> TextCompletions {
        var request = request
        request.stream = onText != nil
        return try await exchange(request, model: Self.selectedModel(for: .completion)) { envelope in
            if let chunk = envelope.completionChunk { await onText?(chunk.choices) }
        }
    }

    // With `onAudio`, one text's pieces arrive in order while it is generated, then the complete reply.
    func speak(_ request: SpeechRequest, model: String? = nil,
               onAudio: (@Sendable (SpeechChunk) async -> Void)? = nil) async throws -> SpeechReply {
        var request = request
        request.stream = onAudio == nil ? nil : true
        return try await exchange(request, model: model ?? Self.selectedModel(for: .speech)) { envelope in
            if let chunk = envelope.chunk { await onAudio?(chunk) }
        }
    }

    func createVoice(_ request: SpeechVoiceRequest, model: String? = nil) async throws -> Data {
        struct Voice: Decodable { let voicePrompt: Data }
        let reply: Voice = try await exchange(request, model: model ?? Self.selectedModel(for: .speech))
        return reply.voicePrompt
    }

    // Raw input only, with the loaded model's tokenizer.
    func tokenSpans(_ text: String, model: String) async throws -> [TokenSpan] {
        struct Tokenize: Encodable, Sendable {
            let operation = "tokenize"
            let text: String
        }
        struct Spans: Decodable { let tokenSpans: [TokenSpan] }
        let reply: Spans = try await exchange(Tokenize(text: text), model: model, load: false)
        return reply.tokenSpans
    }

    // One request and its reply. Previews (`load: false`) never load a model or extend its idle timeout.
    // Streamed pieces go to `onPiece` until the reply; the exchange drains them even if no one listens.
    private func exchange<T: Decodable>(_ request: some Encodable & Sendable, model: String, load: Bool = true,
                                        onPiece: ((Envelope) async -> Void)? = nil) async throws -> T {
        await acquire()
        defer { release(activity: load) }
        try Task.checkCancellation()
        if load {
            _ = try await prepare(model: model)
        } else if !isConnected || readyReply?.modelId != model {
            throw TinkerboxError("Load the model to preview its tokens.")
        }
        do { try input!.fileHandleForWriting.write(contentsOf: try Self.encoder.encode(request) + [10]) } catch {
            await disconnect()
            throw Self.stopped()
        }
        while true {
            let line = try await nextLine()
            let envelope = try Self.decoder.decode(Envelope.self, from: line)
            if let error = envelope.error { throw TinkerboxError(error) }
            if envelope.chunk != nil || envelope.completionChunk != nil {
                await onPiece?(envelope)
                continue
            }
            return try Self.decoder.decode(T.self, from: line)
        }
    }

    private func nextLine() async throws -> Data {
        guard var replies, let line = try await replies.next() else {
            await disconnect()
            throw Self.stopped()
        }
        self.replies = replies
        return Data(line.utf8)
    }

    private func disconnect(graceful: Bool = false) async {
        idleTask?.cancel()
        idleTask = nil
        idleDeadline = nil
        replies = nil
        readyReply = nil
        input?.fileHandleForWriting.closeFile()
        input = nil
        let process = self.process
        self.process = nil
        if let process, process.isRunning {
            if !graceful { Self.stop(process) }
            // EOF lets the worker close its native host and ANE resources. Bound shutdown if stuck.
            let watchdog = Task {
                do { try await Task.sleep(for: .seconds(12)) } catch { return }
                Self.stop(process, signal: SIGKILL)
            }
            await process.exited()
            watchdog.cancel()
        }
        await MainActor.run {
            state.reply = nil
            state.model = nil
            state.loading = false
            state.progress = nil
            Runtime.workerProcess = nil
        }
    }

    private static func stopped() -> TinkerboxError {
        let lines = ((try? String(contentsOf: logURL, encoding: .utf8)) ?? "").split(separator: "\n").suffix(10)
        return TinkerboxError("The model process quit unexpectedly. From \(logURL.path):\n\(lines.joined(separator: "\n"))")
    }
}
