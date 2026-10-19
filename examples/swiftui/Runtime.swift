import Foundation

enum QuestionType: String, CaseIterable, Codable, Sendable {
    case noul, choice, score

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

struct Question: Encodable, Sendable {
    let type: QuestionType
    let instructions: String
    var criteria: [String] = []
}

struct Item: Encodable, Sendable {
    let state: String
    var images: [String] = []  // file paths
}

// Every question is asked about every item. Several questions about one item share a single tree
// pass; one question about many items is packed into one batch.
struct Request: Encodable, Sendable {
    let items: [Item]
    let questions: [Question]

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
}

struct Decisions: Sendable {
    let results: [ItemResult]  // in the request's item order
    let elapsedMs: Double
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
    let results: [ItemResult]?
    let elapsedMs: Double?
    let error: String?
}

actor Runtime {
    static let shared = Runtime()
    @MainActor static var setupProcess: Process?
    static let logURL = FileManager.default.homeDirectoryForCurrentUser
        .appending(path: "Library/Logs/tinkerbox-ane/demo.log")
    private static let modelKey = "model"
    private var process: Process?
    private var input: Pipe?
    private var output: Pipe?
    private var replies: AsyncLineSequence<FileHandle.AsyncBytes>.AsyncIterator?
    private var readyReply: Reply?
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    private init() {
        // A dead worker should fail the next write, not kill the app.
        signal(SIGPIPE, SIG_IGN)
    }

    var isConnected: Bool { readyReply != nil && process?.isRunning == true && replies != nil }

    // The UI and Shortcuts both use the model chosen in the app.
    nonisolated static var selectedModel: String {
        let runnable = Catalog.bundled.filter(\.isSupported).map(\.id)
        let stored = UserDefaults.standard.string(forKey: modelKey)
        return stored.flatMap { runnable.contains($0) ? $0 : nil } ?? runnable.first ?? ""
    }

    // Takes effect on the next start or decision. Stopping the worker also cancels a download or
    // load in progress; the queued caller then fails and the next one launches the new model.
    func select(model: String) {
        guard model != Self.selectedModel || readyReply?.modelId != model else { return }
        UserDefaults.standard.set(model, forKey: Self.modelKey)
        disconnect()
    }

    // Actors are reentrant at awaits. Hold the queue across startup AND a full
    // request/reply exchange so UI and Shortcuts calls cannot read each other's replies.
    private func acquire() async {
        if busy {
            await withCheckedContinuation { waiters.append($0) }
        } else {
            busy = true
        }
    }

    private func release() {
        if waiters.isEmpty { busy = false } else { waiters.removeFirst().resume() }
    }

    func start(status: @MainActor @Sendable (String, Double?) -> Void = { _, _ in }) async throws -> Reply {
        await acquire()
        defer { release() }
        try Task.checkCancellation()
        return try await prepare(status: status)
    }

    private func prepare(status: @MainActor @Sendable (String, Double?) -> Void) async throws -> Reply {
        if let readyReply, process?.isRunning == true, replies != nil { return readyReply }
        readyReply = nil
        do {
            return try await launch(status: status)
        } catch {
            disconnect()
            await MainActor.run { Runtime.setupProcess = nil }
            throw error
        }
    }

    private func launch(status: @MainActor @Sendable (String, Double?) -> Void) async throws -> Reply {
        if let process, process.isRunning { process.terminate() }
        input?.fileHandleForWriting.closeFile()
        output?.fileHandleForReading.closeFile()
        let process = Process(), input = Pipe(), output = Pipe()
        self.process = process
        self.input = input
        self.output = output
        replies = output.fileHandleForReading.bytes.lines.makeAsyncIterator()
        let resources = Bundle.main.resourceURL!
        process.executableURL = Bundle.main.executableURL!.deletingLastPathComponent().appending(path: "uv")
        process.arguments = [
            "run", "--no-project", "--no-config", "--isolated", "--managed-python",
            "--python", "3.12", "--no-build", "--with-requirements",
            resources.appending(path: "requirements.txt").path,
            "python", "-u", resources.appending(path: "worker.py").path,
        ]
        var environment = ProcessInfo.processInfo.environment
        for name in ["PYTHONPATH", "PYTHONHOME", "UV_INTERNAL__PARENT_INTERPRETER"] {
            environment.removeValue(forKey: name)
        }
        environment["TINKERBOX_MODEL"] = Self.selectedModel
        process.environment = environment
        process.currentDirectoryURL = FileManager.default.homeDirectoryForCurrentUser
        try FileManager.default.createDirectory(at: Self.logURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: Self.logURL.path, contents: nil)
        let log = try FileHandle(forWritingTo: Self.logURL)
        defer { log.closeFile() }
        process.standardInput = input
        process.standardOutput = output
        process.standardError = log
        try await MainActor.run {
            try process.run()
            Runtime.setupProcess = process
        }
        // Let the child see EOF when the app closes its writer or exits.
        input.fileHandleForReading.closeFile()
        output.fileHandleForWriting.closeFile()
        while true {
            let reply = try await readReply()
            if let message = reply.status {
                await status(message, reply.progress)
            } else {
                guard reply.ready == true else { throw DecisionError.invalidReply }
                await MainActor.run { Runtime.setupProcess = nil }
                readyReply = reply
                return reply
            }
        }
    }

    func decide(_ request: Request) async throws -> Decisions {
        try request.validate()
        await acquire()
        defer { release() }
        try Task.checkCancellation()
        _ = try await prepare(status: { _, _ in })
        var data = try JSONEncoder().encode(request)
        data.append(10)
        do { try input!.fileHandleForWriting.write(contentsOf: data) } catch {
            disconnect()
            throw Self.stopped()
        }
        let reply = try await readReply()
        guard let results = reply.results, let elapsed = reply.elapsedMs, results.count == request.items.count,
              results.allSatisfy({ $0.answers.count == request.questions.count }) else {
            throw DecisionError.invalidReply
        }
        return Decisions(results: results, elapsedMs: elapsed)
    }

    private func readReply() async throws -> Reply {
        let reply: Reply
        do {
            guard var replies, let line = try await replies.next() else { throw Self.stopped() }
            self.replies = replies
            reply = try JSONDecoder().decode(Reply.self, from: Data(line.utf8))
        } catch {
            disconnect()
            throw error
        }
        if let error = reply.error {
            throw NSError(domain: "Tinkerbox", code: 1, userInfo: [NSLocalizedDescriptionKey: error])
        }
        return reply
    }

    private func disconnect() {
        replies = nil
        readyReply = nil
        input?.fileHandleForWriting.closeFile()
        output?.fileHandleForReading.closeFile()
        if let process, process.isRunning { process.terminate() }
    }

    private static func stopped() -> NSError {
        let details = (try? String(contentsOf: logURL, encoding: .utf8))?.suffix(2000) ?? ""
        return NSError(domain: "Tinkerbox", code: 2,
                       userInfo: [NSLocalizedDescriptionKey: "The runtime stopped.\n\(details)"])
    }
}
