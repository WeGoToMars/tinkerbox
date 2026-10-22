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
// pass; one question about many items is packed into one batch.
struct Request: Encodable, Sendable {
    let items: [Item]
    let questions: [Question]
    var settings: [String: Int] = [:]  // filled in by Runtime from the model's saved values
    var count = false  // report tokens and image tiles instead of answering

    func validate() throws {
        if items.isEmpty || questions.isEmpty {
            throw DecisionError.invalid("Ask at least one question about at least one input.")
        }
        if items.contains(where: { isBlank($0.state) && $0.images.isEmpty }) {
            throw DecisionError.invalid("Input text must not be empty unless you add an image.")
        }
        for question in questions {
            let (type, criteria) = (question.type, question.criteria)
            if isBlank(question.instructions) { throw DecisionError.invalid("Questions must not be empty.") }
            if type == .choice && (criteria.count < 2 || Set(criteria).count != criteria.count) {
                throw DecisionError.invalid("Add at least two unique options.")
            }
            if type == .score && !(2...10).contains(criteria.count) {
                throw DecisionError.invalid("Add 2–10 levels, ordered from lowest to highest.")
            }
            if type != .noul && criteria.contains(where: isBlank) {
                throw DecisionError.invalid("Options and levels must not be empty.")
            }
        }
    }
}

enum DecisionError: LocalizedError {
    case invalid(String)
    case invalidReply

    var errorDescription: String? {
        switch self {
        case .invalid(let message): message
        case .invalidReply: "The model returned an incomplete response."
        }
    }
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

struct Decisions: Sendable {
    let results: [ItemResult]  // in the request's item order
    let elapsedMs: Double
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

struct Reply: Decodable, Sendable {
    let status: String?
    let progress: Double?
    let ready: Bool?
    let modelId: String?
    let setupMs: Double?
    let modelName: String?
    let cpuModel: String?
    let architecture: String?
    let questionTrees: Bool?  // several questions can share one pass
    let settings: [ModelSetting]?
    let info: [ModelInfo]?
    let results: [ItemResult]?
    let counts: [TokenCount]?  // in the request's item order
    let elapsedMs: Double?
    let error: String?
}

@MainActor @Observable final class RuntimeState {
    nonisolated init() {}
    var reply: Reply?
    var model: String?  // loaded, or being loaded
    var loading = false
    var status = "Preparing runtime…"
    var progress: Double?
}

actor Runtime {
    static let shared = Runtime()
    @MainActor static var workerProcess: Process?
    nonisolated let state = RuntimeState()
    static let logURL = FileManager.default.homeDirectoryForCurrentUser
        .appending(path: "Library/Logs/tinkerbox-ane/demo.log")
    private var process: Process?
    private var input: Pipe?
    private var output: Pipe?
    private var replies: AsyncLineSequence<FileHandle.AsyncBytes>.AsyncIterator?
    private var readyReply: Reply?
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

    nonisolated static func stop(_ process: Process?, signal: Int32 = SIGTERM) {
        guard let process, process.isRunning else { return }
        let pid = process.processIdentifier
        kill(getpgid(pid) == pid ? -pid : pid, signal)
    }

    // Each task remembers its own model; the UI and Shortcuts both use the one chosen in the app.
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
    func select(model: String, for task: ModelTask) async {
        await acquire()
        defer { release() }
        UserDefaults.standard.set(model, forKey: "model.\(task.rawValue)")
        if readyReply?.modelId != model { await disconnect(graceful: true) }
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
        idleTask = Task { [weak self] in
            do { try await ContinuousClock().sleep(until: deadline) } catch { return }
            await self?.unloadIfIdle()
        }
    }

    private func unloadIfIdle() async {
        guard !Task.isCancelled, !busy, let deadline = idleDeadline, .now >= deadline else { return }
        await unload()
    }

    func start(model: String) async throws -> Reply {
        await acquire()
        defer { release() }
        try Task.checkCancellation()
        return try await prepare(model: model)
    }

    // One model is loaded at a time; a request for another one replaces it.
    private func prepare(model: String) async throws -> Reply {
        if let readyReply, readyReply.modelId == model, process?.isRunning == true, replies != nil { return readyReply }
        readyReply = nil
        await MainActor.run {
            state.model = model
            state.loading = true
            state.reply = nil
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

    private func launch(model: String) async throws -> Reply {
        Self.stop(process)
        input?.fileHandleForWriting.closeFile()
        output?.fileHandleForReading.closeFile()
        let process = Process(), input = Pipe(), output = Pipe()
        self.process = process
        self.input = input
        self.output = output
        replies = output.fileHandleForReading.bytes.lines.makeAsyncIterator()
        Self.configure(process, python: ["-u", Bundle.main.resourceURL!.appending(path: "worker.py").path])
        process.environment!["TINKERBOX_MODEL"] = model
        try FileManager.default.createDirectory(at: Self.logURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: Self.logURL.path, contents: nil)
        let log = try FileHandle(forWritingTo: Self.logURL)
        defer { log.closeFile() }
        process.standardInput = input
        process.standardOutput = output
        process.standardError = log
        try await MainActor.run {
            try process.run()
            Runtime.workerProcess = process
        }
        // Let the child see EOF when the app closes its writer or exits.
        input.fileHandleForReading.closeFile()
        output.fileHandleForWriting.closeFile()
        while true {
            let reply = try await readReply()
            if let message = reply.status {
                await MainActor.run {
                    state.status = message
                    state.progress = reply.progress
                }
            } else {
                guard reply.ready == true else { throw DecisionError.invalidReply }
                readyReply = reply
                await MainActor.run {
                    state.reply = reply
                    state.loading = false
                    state.progress = nil
                }
                return reply
            }
        }
    }

    // Runs Python with these arguments in the app's environment: its pinned tinkerbox-ane and dependencies.
    nonisolated static func configure(_ process: Process, python arguments: [String]) {
        process.executableURL = Bundle.main.executableURL!.deletingLastPathComponent().appending(path: "uv")
        process.arguments = [
            "run", "--no-project", "--no-config", "--isolated", "--managed-python",
            "--python", "3.12", "--no-build", "--with-requirements",
            Bundle.main.resourceURL!.appending(path: "requirements.txt").path, "python",
        ] + arguments
        var environment = ProcessInfo.processInfo.environment
        for name in ["PYTHONPATH", "PYTHONHOME", "UV_INTERNAL__PARENT_INTERPRETER"] {
            environment.removeValue(forKey: name)
        }
        process.environment = environment
        process.currentDirectoryURL = FileManager.default.homeDirectoryForCurrentUser
    }

    func decide(_ request: Request) async throws -> Decisions {
        let reply = try await exchange(request)
        guard let results = reply.results, let elapsed = reply.elapsedMs, results.count == request.items.count,
              results.allSatisfy({ $0.answers.count == request.questions.count }) else {
            throw DecisionError.invalidReply
        }
        return Decisions(results: results, elapsedMs: elapsed)
    }

    // What each item would read, from the same prompt and settings as a run, without running the model.
    func count(_ request: Request) async throws -> [TokenCount] {
        var request = request
        request.count = true
        let reply = try await exchange(request)
        guard let counts = reply.counts, counts.count == request.items.count else { throw DecisionError.invalidReply }
        return counts
    }

    // Sends one request to the decision model, with its saved settings, and reads its reply.
    private func exchange(_ request: Request) async throws -> Reply {
        try request.validate()
        await acquire()
        defer { release(activity: !request.count) }
        try Task.checkCancellation()
        let model = Self.selectedModel(for: .decision)
        if request.count {
            guard isConnected, readyReply?.modelId == model else {
                throw DecisionError.invalid("Load a model before counting tokens.")
            }
        } else {
            _ = try await prepare(model: model)
        }
        var request = request
        request.settings = Self.settings(for: model)
        var data = try JSONEncoder().encode(request)
        data.append(10)
        do { try input!.fileHandleForWriting.write(contentsOf: data) } catch {
            await disconnect()
            throw Self.stopped()
        }
        return try await readReply()
    }

    private func readReply() async throws -> Reply {
        let reply: Reply
        do {
            guard var replies, let line = try await replies.next() else { throw Self.stopped() }
            self.replies = replies
            reply = try JSONDecoder().decode(Reply.self, from: Data(line.utf8))
        } catch {
            await disconnect()
            throw error
        }
        if let error = reply.error {
            throw NSError(domain: "Tinkerbox", code: 1, userInfo: [NSLocalizedDescriptionKey: error])
        }
        return reply
    }

    private func disconnect(graceful: Bool = false) async {
        idleTask?.cancel()
        idleTask = nil
        idleDeadline = nil
        replies = nil
        readyReply = nil
        input?.fileHandleForWriting.closeFile()
        output?.fileHandleForReading.closeFile()
        let process = self.process
        self.process = nil
        input = nil
        output = nil
        if let process, process.isRunning {
            if !graceful { Self.stop(process) }
            // EOF lets the worker close its native host and ANE resources. Bound shutdown if stuck.
            let watchdog = Task {
                do { try await Task.sleep(for: .seconds(12)) } catch { return }
                Self.stop(process, signal: SIGKILL)
            }
            await Task.detached { process.waitUntilExit() }.value
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

    private static func stopped() -> NSError {
        let details = (try? String(contentsOf: logURL, encoding: .utf8))?.suffix(2000) ?? ""
        return NSError(domain: "Tinkerbox", code: 2,
                       userInfo: [NSLocalizedDescriptionKey: "The runtime stopped.\n\(details)"])
    }
}
