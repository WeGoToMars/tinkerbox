import Foundation

// What a model does. Each task has its own page in the app and its own selected model.
enum ModelTask: String, CaseIterable, Sendable {
    case decision
    case speech

    var title: String {
        switch self {
        case .decision: "Decision (System One)"
        case .speech: "Text to Speech"
        }
    }

    var symbol: String {
        switch self {
        case .decision: "arrow.triangle.branch"
        case .speech: "waveform"
        }
    }

    // Its section in the Models library.
    var heading: String {
        switch self {
        case .decision: "Decision"
        case .speech: "Speech"
        }
    }
}

// One entry of tinkerbox_ane/models.json. Released apps keep reading the remote copy, so only
// identity and display fields are required; entries for other tasks may omit the rest.
struct CatalogModel: Decodable, Identifiable, Hashable, Sendable {
    let id: String
    let name: String
    let publisher: String
    let summary: String
    let task: String  // a ModelTask; the remote catalog may list tasks newer than this app
    // Hugging Face repositories: the foundation it was fine-tuned from, the model that was
    // converted, and the converted Core AI bundles.
    let baseModel: String?
    let originalModel: String?
    let repo: String
    let revision: String
    let inputs: [String]?
    let context: Int?
    let downloadBytes: Int64?
    let license: String?
    let localOnly: Bool?
    let minVersion: String

    var kind: ModelTask? { ModelTask(rawValue: task) }
    var acceptsImages: Bool { inputs?.contains("image") == true }
    // This app can run it: its package ships the model, and the task has a page here.
    var isSupported: Bool { kind != nil && Catalog.bundled.contains { $0.id == id } }
}

@MainActor @Observable final class Catalog {
    nonisolated static let remoteURL = URL(string: "https://raw.githubusercontent.com/WeGoToMars/tinkerbox/main/src/tinkerbox_ane/models.json")!
    nonisolated static let appVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0"
    // Matches tinkerbox_ane.bundle.CACHE: <model>/<revision>/macos27/<architecture>/model.json.
    nonisolated static let cacheURL = FileManager.default.homeDirectoryForCurrentUser
        .appending(path: "Library/Caches/tinkerbox-ane/models")

    // The bundled catalog lists exactly the models this app's Python package can run.
    nonisolated static let bundled: [CatalogModel] = {
        guard let url = Bundle.main.url(forResource: "models", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return [] }
        return (try? decode(data)) ?? []
    }()

    var models = Catalog.bundled
    var isRemote = false
    var downloaded: Set<String> = []

    // The catalog format this app reads; a newer one is ignored instead of misread.
    nonisolated static let schema = 1

    nonisolated static func decode(_ data: Data) throws -> [CatalogModel] {
        // An entry this app cannot read is skipped, so a newer kind of entry never hides the others.
        struct Entry: Decodable {
            let model: CatalogModel?
            init(from decoder: Decoder) throws { model = try? CatalogModel(from: decoder) }
        }
        struct File: Decodable { let schema: Int; let models: [Entry] }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let file = try decoder.decode(File.self, from: data)
        guard file.schema <= schema else { throw URLError(.cannotDecodeContentData) }
        return file.models.compactMap(\.model)
    }

    nonisolated static func isNewer(_ version: String, than other: String) -> Bool {
        let parts = { (text: String) in text.split(separator: ".").map { Int($0) ?? 0 } }
        let (a, b) = (parts(version), parts(other))
        for index in 0..<max(a.count, b.count) {
            let (x, y) = (index < a.count ? a[index] : 0, index < b.count ? b[index] : 0)
            if x != y { return x > y }
        }
        return false
    }

    func refresh() async {
        scanDownloads()
        do {
            let (data, response) = try await URLSession.shared.data(from: Self.remoteURL)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
            let remote = try Self.decode(data)
            // Bundled entries describe what this app runs; the remote catalog only adds models to discover.
            models = Self.bundled + remote.filter { model in !Self.bundled.contains { $0.id == model.id } }
            isRemote = true
        } catch {
            models = Self.bundled
            isRemote = false
        }
        scanDownloads()
    }

    func scanDownloads() {
        let files = FileManager.default
        // Only the catalog's revision counts; an older one would be downloaded again on load.
        downloaded = Set(models.filter { model in
            if let local = UserDefaults.standard.string(forKey: "asset.\(model.id)"),
               files.fileExists(atPath: URL(fileURLWithPath: local).appending(path: "model.json").path) {
                return true
            }
            let folder = Self.cacheURL.appending(path: model.id).appending(path: model.revision).appending(path: "macos27")
            let bundles = (try? files.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
            return bundles.contains { files.fileExists(atPath: $0.appending(path: "model.json").path) }
        }.map(\.id))
    }
}

// Bundles download in their own processes, beside the runtime: switching models neither waits for
// nor stops a download, and the previous model stays loaded until the next one is ready.
@MainActor @Observable final class Downloads {
    static var processes: [String: Process] = [:]  // by model; stopped when the app quits
    private(set) var progress: [String: Double] = [:]  // by model, once the download size is known
    private var tasks: [String: Task<Void, Error>] = [:]
    private var stopped: Set<String> = []

    func contains(_ model: String) -> Bool { tasks[model] != nil }

    // Waits for a model's download, starting it unless it is already running.
    func fetch(_ model: String) async throws {
        if let task = tasks[model] { return try await task.value }
        let task = Task {
            defer {
                tasks[model] = nil
                progress[model] = nil
            }
            try await run(model)
        }
        tasks[model] = task
        try await task.value
    }

    // Stops a download and waits for it to end; its partial files stay until the model is removed.
    // Those waiting for it see a CancellationError.
    func stop(_ model: String) async {
        guard let task = tasks[model] else { return }
        stopped.insert(model)
        Runtime.stop(Self.processes[model])
        _ = try? await task.value
    }

    private func run(_ model: String) async throws {
        struct Update: Decodable { let progress: Double? }
        defer { stopped.remove(model) }
        let process = Process(), output = Pipe(), errors = Pipe()
        Runtime.configure(process, python: ["-u", "-m", "tinkerbox_ane.download", model])
        process.environment!["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = output
        process.standardError = errors
        try process.run()
        Self.processes[model] = process
        defer { Self.processes[model] = nil }
        // Read both pipes at once so neither fills up and stalls the other.
        async let failure = Task.detached { errors.fileHandleForReading.readDataToEndOfFile() }.value
        for try await line in output.fileHandleForReading.bytes.lines {
            if let value = (try? JSONDecoder().decode(Update.self, from: Data(line.utf8)))?.progress {
                progress[model] = value
            }
        }
        let details = String(decoding: await failure, as: UTF8.self)
        await Task.detached { process.waitUntilExit() }.value
        if stopped.contains(model) { throw CancellationError() }
        guard process.terminationStatus == 0 else {
            throw NSError(domain: "Tinkerbox", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Could not download the model.\n\(details.suffix(2000))"])
        }
    }
}
