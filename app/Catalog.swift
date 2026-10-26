import Foundation
import CoreAI

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
    let nonVerbalTags: [String]?
    let context: Int?
    let downloadBytes: Int64?
    let license: String?
    let localOnly: Bool?
    let architectures: [String]?
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
    nonisolated static let deviceArchitecture = AIModel.deviceArchitectureName
    nonisolated private static let bundledData = Bundle.main.url(forResource: "models", withExtension: "json")
        .flatMap { try? Data(contentsOf: $0) }
    nonisolated static let architectures: [String: String] = {
        guard let data = bundledData,
              let catalog = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [:] }
        return catalog["architectures"] as? [String: String] ?? [:]
    }()

    // The bundled catalog lists exactly the models this app's Python package can run.
    nonisolated static let bundled: [CatalogModel] = {
        guard let data = bundledData else { return [] }
        return (try? decode(data)) ?? []
    }()

    // A published revision the user accepted, newer than the one this app ships; kept as the raw
    // catalog entry because the worker and downloader read it from TINKERBOX_CATALOG_ENTRY.
    nonisolated static func pinnedEntry(_ id: String) -> String? {
        UserDefaults.standard.string(forKey: "catalog.entry.\(id)")
    }

    // Entries of every accepted update, in the form tinkerbox_ane.models reads; `adding` is one being downloaded.
    nonisolated static func updatesEnvironment(adding: String? = nil) -> String {
        let entries = bundled.compactMap { model in current(model.id).flatMap { $0.revision == model.revision ? nil : pinnedEntry(model.id) } }
        return "[" + (entries + [adding].compactMap { $0 }).joined(separator: ",") + "]"
    }

    nonisolated static func pin(_ raw: String, for id: String) {
        UserDefaults.standard.set(raw, forKey: "catalog.entry.\(id)")
    }

    // The entry this app runs for a bundled model: its own, unless an accepted update is newer.
    nonisolated static func current(_ id: String) -> CatalogModel? {
        guard let bundled = bundled.first(where: { $0.id == id }) else { return nil }
        guard let raw = pinnedEntry(id), let pinned = try? decode(Data("{\"schema\":1,\"models\":[\(raw)]}".utf8)).first,
              isNewer(pinned.revision, than: bundled.revision), !isNewer(pinned.minVersion, than: appVersion)
        else { return bundled }
        return pinned
    }

    // Whether an earlier revision of the model is installed, so the current one is an update.
    nonisolated static func hasOlderBundle(_ model: CatalogModel, architecture: String = deviceArchitecture) -> Bool {
        let folders = (try? FileManager.default.contentsOfDirectory(
            at: cacheURL.appending(path: model.id), includingPropertiesForKeys: nil)) ?? []
        return folders.contains { folder in
            folder.lastPathComponent != model.revision && bundleIssue(
                at: folder.appending(path: "macos27").appending(path: architecture), architecture: architecture) == nil
        }
    }

    nonisolated static func bundleIssue(at url: URL, architecture: String = deviceArchitecture) -> String? {
        guard let data = try? Data(contentsOf: url.appending(path: "model.json")),
              let manifest = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              manifest["version"] as? Int == 2,
              let target = manifest["architecture"] as? String else {
            return "Choose a compiled bundle directory containing model.json."
        }
        guard target == architecture else {
            return "This bundle targets \(architectures[target] ?? target). Choose a bundle compiled for this Mac."
        }
        return nil
    }

    nonisolated static func cachedBundle(for model: CatalogModel, architecture: String = deviceArchitecture) -> URL {
        cacheURL.appending(path: model.id).appending(path: model.revision)
            .appending(path: "macos27").appending(path: architecture)
    }

    nonisolated static func localBundle(for model: CatalogModel, architecture: String = deviceArchitecture) -> URL? {
        let candidate = UserDefaults.standard.string(forKey: "asset.\(model.id)").map { URL(fileURLWithPath: $0) }
            ?? cachedBundle(for: model, architecture: architecture)
        return bundleIssue(at: candidate, architecture: architecture) == nil ? candidate : nil
    }

    nonisolated static func availabilityIssue(for model: CatalogModel, asset: URL? = nil,
                                             architecture: String = deviceArchitecture) -> String? {
        guard architectures[architecture] != nil else {
            return "This Mac's Neural Engine is not supported by this version of Tinkerbox. Update Tinkerbox to use this Mac."
        }
        if let asset { return bundleIssue(at: asset, architecture: architecture) }
        if let selected = UserDefaults.standard.string(forKey: "asset.\(model.id)") {
            return bundleIssue(at: URL(fileURLWithPath: selected), architecture: architecture)
        }
        if localBundle(for: model, architecture: architecture) != nil { return nil }
        if let supported = model.architectures, !supported.contains(architecture) {
            return "No \(model.name) bundle is available for this Mac. Choose another model or a matching local bundle."
        }
        if model.localOnly == true { return "Choose a local \(model.name) bundle to use this model." }
        return nil
    }

    var models = Catalog.bundled.map { Catalog.current($0.id) ?? $0 }
    var isRemote = false
    // Newer published revisions of installed models this app can run, by model, with the raw entry.
    var offers: [String: (model: CatalogModel, raw: String)] = [:]
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
            let current = Self.bundled.map { Self.current($0.id) ?? $0 }
            models = current + remote.filter { model in !Self.bundled.contains { $0.id == model.id } }
            offers = Self.offers(in: data, remote: remote, current: current)
            isRemote = true
        } catch {
            models = Self.bundled
            isRemote = false
        }
        scanDownloads()
    }

    // A published revision is offered when it is newer than the one this app runs and this app is new enough for it.
    nonisolated private static func offers(in data: Data, remote: [CatalogModel], current: [CatalogModel])
        -> [String: (model: CatalogModel, raw: String)] {
        let entries = ((try? JSONSerialization.jsonObject(with: data)) as? [String: Any])?["models"] as? [[String: Any]] ?? []
        var offers: [String: (model: CatalogModel, raw: String)] = [:]
        for model in current {
            guard let newer = remote.first(where: { $0.id == model.id }),
                  isNewer(newer.revision, than: model.revision), !isNewer(newer.minVersion, than: appVersion),
                  let entry = entries.first(where: { $0["id"] as? String == model.id }),
                  let raw = try? JSONSerialization.data(withJSONObject: entry),
                  let text = String(data: raw, encoding: .utf8) else { continue }
            offers[model.id] = (newer, text)
        }
        return offers
    }

    // What the model's row offers to replace: an installed older revision, or the newer published one.
    func update(for model: CatalogModel) -> CatalogModel? {
        if let offer = offers[model.id], Self.localBundle(for: model) != nil || Self.hasOlderBundle(model)
            || Self.hasOlderBundle(offer.model) { return offer.model }
        return !downloaded.contains(model.id) && Self.hasOlderBundle(model) ? model : nil
    }

    // Switches a model to its offered revision once that is downloaded; every launch of the app and worker follows.
    func adopt(_ id: String) {
        guard let offer = offers[id] else { return }
        Self.pin(offer.raw, for: id)
        models = models.map { $0.id == id ? offer.model : $0 }
        offers[id] = nil
        scanDownloads()
    }

    func scanDownloads() {
        // Only the catalog's revision counts; an older one would be downloaded again on load.
        downloaded = Set(models.filter { Self.localBundle(for: $0) != nil }.map(\.id))
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
    func fetch(_ model: String, entry: String? = nil) async throws {
        if let task = tasks[model] { return try await task.value }
        let task = Task {
            defer {
                tasks[model] = nil
                progress[model] = nil
            }
            try await run(model, entry: entry)
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

    private func run(_ model: String, entry: String?) async throws {
        struct Update: Decodable { let progress: Double? }
        defer { stopped.remove(model) }
        let process = Process(), output = Pipe(), errors = Pipe()
        Runtime.configure(process, python: ["-u", "-m", "tinkerbox_ane.download", model])
        process.environment!["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        process.environment!["TINKERBOX_CATALOG_UPDATES"] = Catalog.updatesEnvironment(adding: entry)
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
