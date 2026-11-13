import Foundation
import CoreAI

// What a model does. Each task has its own page in the app and its own selected model.
enum ModelTask: String, CaseIterable, Sendable {
    case decision
    case speech
    case completion = "text-generation"

    var title: String {
        switch self {
        case .decision: "Decision (System One)"
        case .speech: "Text to Speech"
        case .completion: "Text Completion"
        }
    }

    var symbol: String {
        switch self {
        case .decision: "arrow.triangle.branch"
        case .speech: "waveform"
        case .completion: "text.append"
        }
    }

    // Its section in the Models library.
    var heading: String {
        switch self {
        case .decision: "Decision"
        case .speech: "Speech"
        case .completion: "Text Completion"
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
    let family: String?
    let size: String?
    let quantization: String?
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
    let architectures: [String]?
    let minVersion: String

    var kind: ModelTask? { ModelTask(rawValue: task) }
    var acceptsImages: Bool { inputs?.contains("image") == true }
    var fileSize: String? { downloadBytes.map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) } }
    // This app can run it: its package ships the model, and the task has a page here.
    var isSupported: Bool { kind != nil && Catalog.bundled.contains { $0.id == id } }
}

@MainActor @Observable final class Catalog {
    nonisolated static let remoteURL = URL(
        string: "https://raw.githubusercontent.com/WeGoToMars/tinkerbox/main/src/tinkerbox_ane/models.json")!
    nonisolated static let appVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0"
    // Matches tinkerbox_ane.bundle.CACHE: <model>/<revision>/macos27/<architecture>/model.json.
    nonisolated static let cacheURL = URL.cachesDirectory.appending(path: "tinkerbox-ane/models")
    nonisolated static let deviceArchitecture = AIModel.deviceArchitectureName
    // The bundled catalog lists exactly the models this app's Python package can run, and the chips it knows.
    nonisolated private static let bundledFile = try! decodeFile(
        Data(contentsOf: Bundle.main.url(forResource: "models", withExtension: "json")!))
    nonisolated static let bundled = bundledFile.models.compactMap(\.model)
    nonisolated static let architectures = bundledFile.architectures ?? [:]

    // Backend declarations bundled at build time: Shortcuts pickers need no model or Python setup.
    nonisolated static let speechCapabilities = try! Runtime.decoder.decode(
        [String: SpeechCapabilities].self,
        from: Data(contentsOf: Bundle.main.url(forResource: "speech-capabilities", withExtension: "json")!))

    // A published revision the user accepted, newer than the one this app ships; kept as the raw
    // catalog entry because the worker and downloader read it from TINKERBOX_CATALOG_UPDATES.
    nonisolated static func pinnedEntry(_ id: String) -> String? {
        UserDefaults.standard.string(forKey: "catalog.entry.\(id)")
    }

    // Entries of every accepted update, in the form tinkerbox_ane.models reads; `adding` is one being downloaded.
    nonisolated static func updatesEnvironment(adding: String? = nil) -> String {
        let entries = bundled.compactMap { model in
            current(model.id).flatMap { $0.revision == model.revision ? nil : pinnedEntry(model.id) }
        }
        return "[" + (entries + [adding].compactMap { $0 }).joined(separator: ",") + "]"
    }

    nonisolated static func pin(_ raw: String, for id: String) {
        UserDefaults.standard.set(raw, forKey: "catalog.entry.\(id)")
    }

    // The entry this app runs for a bundled model: its own, unless an accepted update is newer.
    nonisolated static func current(_ id: String) -> CatalogModel? {
        guard let bundled = bundled.first(where: { $0.id == id }) else { return nil }
        guard let raw = pinnedEntry(id),
            let pinned = try? Runtime.decoder.decode(CatalogModel.self, from: Data(raw.utf8)),
            isNewer(pinned.revision, than: bundled.revision), !isNewer(pinned.minVersion, than: appVersion)
        else { return bundled }
        return pinned
    }

    // A locally built bundle chosen for a model, used instead of its download.
    nonisolated static func localAsset(for id: String) -> String? {
        UserDefaults.standard.string(forKey: "asset.\(id)")
    }

    nonisolated static func setLocalAsset(_ path: String?, for id: String) {
        UserDefaults.standard.set(path, forKey: "asset.\(id)")
    }

    // Whether an earlier revision of the model is installed, so the current one is an update.
    nonisolated static func hasOlderBundle(_ model: CatalogModel) -> Bool {
        let folders =
            (try? FileManager.default.contentsOfDirectory(
                at: cacheURL.appending(path: model.id), includingPropertiesForKeys: nil)) ?? []
        return folders.contains { folder in
            folder.lastPathComponent != model.revision
                && bundleIssue(at: folder.appending(path: "macos27").appending(path: deviceArchitecture)) == nil
        }
    }

    nonisolated static func bundleIssue(at url: URL) -> String? {
        struct Manifest: Decodable { let architecture: String }
        guard let data = try? Data(contentsOf: url.appending(path: "model.json")),
            let target = (try? JSONDecoder().decode(Manifest.self, from: data))?.architecture
        else {
            return "Choose a compiled bundle folder containing model.json."
        }
        guard target == deviceArchitecture else {
            return "This bundle targets \(architectures[target] ?? target). Choose a bundle compiled for this Mac."
        }
        return nil
    }

    nonisolated static func cachedBundle(for model: CatalogModel) -> URL {
        cacheURL.appending(path: model.id).appending(path: model.revision)
            .appending(path: "macos27").appending(path: deviceArchitecture)
    }

    // The chosen bundle while it is readable, so a deleted or unmounted one falls back to the download.
    nonisolated static func readableLocalAsset(for id: String) -> String? {
        localAsset(for: id).flatMap { bundleIssue(at: URL(fileURLWithPath: $0)) == nil ? $0 : nil }
    }

    nonisolated static func localBundle(for model: CatalogModel) -> URL? {
        [readableLocalAsset(for: model.id).map { URL(fileURLWithPath: $0) }, cachedBundle(for: model)]
            .compactMap { $0 }.first { bundleIssue(at: $0) == nil }
    }

    // A development bundle that the worker loads for whichever model it runs.
    nonisolated static let overrideAsset = ProcessInfo.processInfo.environment["TINKERBOX_ASSET"].map {
        URL(fileURLWithPath: $0)
    }

    nonisolated static func availabilityIssue(for model: CatalogModel, asset: URL? = nil) -> String? {
        guard architectures[deviceArchitecture] != nil else {
            return
                "This Mac's Neural Engine is not supported by this version of Tinkerbox. Update Tinkerbox to use this Mac."
        }
        if let asset { return bundleIssue(at: asset) }
        if localBundle(for: model) != nil { return nil }
        // Never download over a chosen bundle: it may be on a drive that is not connected.
        if let selected = localAsset(for: model.id) {
            guard FileManager.default.fileExists(atPath: selected) else {
                return
                    "The bundle chosen for \(model.name) is missing. Reconnect it, or delete it in Models to download it."
            }
            return bundleIssue(at: URL(fileURLWithPath: selected))
        }
        if let supported = model.architectures, !supported.contains(deviceArchitecture) {
            return "No \(model.name) bundle is available for this Mac. Choose another model or a matching local bundle."
        }
        return nil
    }

    var models = Catalog.bundled.map { Catalog.current($0.id) ?? $0 }
    var isRemote = false
    // Newer published revisions of installed models this app can run, by model, with the raw entry.
    var offers: [String: (model: CatalogModel, raw: String)] = [:]
    var downloaded: Set<String> = []

    // The catalog format this app reads; a newer one is ignored instead of misread.
    nonisolated static let schema = 1

    // An entry this app cannot read is skipped, so a newer kind of entry never hides the others.
    private struct Entry: Decodable {
        let model: CatalogModel?
        init(from decoder: Decoder) throws { model = try? CatalogModel(from: decoder) }
    }

    private struct File: Decodable {
        let schema: Int
        let architectures: [String: String]?
        let models: [Entry]
    }

    nonisolated private static func decodeFile(_ data: Data) throws -> File {
        // Architecture codes are dictionary keys, which the snake-case strategy leaves alone.
        let file = try Runtime.decoder.decode(File.self, from: data)
        guard file.schema <= schema else { throw URLError(.cannotDecodeContentData) }
        return file
    }

    // Versions and revisions such as "2.3.0" or "v2.3.0".
    nonisolated static func isNewer(_ version: String, than other: String) -> Bool {
        version.compare(other, options: .numeric) == .orderedDescending
    }

    func refresh() async {
        scanDownloads()
        do {
            let (data, response) = try await URLSession.shared.data(from: Self.remoteURL)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
            let remote = try Self.decodeFile(data).models.compactMap(\.model)
            // Bundled entries describe what this app runs; the remote catalog only adds models to discover.
            let current = Self.bundled.map { Self.current($0.id) ?? $0 }
            models = current + remote.filter { model in !Self.bundled.contains { $0.id == model.id } }
            offers = Self.offers(in: data, remote: remote, current: current)
            isRemote = true
        } catch {
            isRemote = false  // offline: keep what this app runs, including accepted updates
        }
        scanDownloads()
    }

    // A published revision is offered when it is newer than the one this app runs and this app is new enough for it.
    nonisolated private static func offers(in data: Data, remote: [CatalogModel], current: [CatalogModel])
        -> [String: (model: CatalogModel, raw: String)]
    {
        let entries =
            ((try? JSONSerialization.jsonObject(with: data)) as? [String: Any])?["models"] as? [[String: Any]] ?? []
        var offers: [String: (model: CatalogModel, raw: String)] = [:]
        for model in current {
            guard let newer = remote.first(where: { $0.id == model.id }),
                isNewer(newer.revision, than: model.revision), !isNewer(newer.minVersion, than: appVersion),
                let entry = entries.first(where: { $0["id"] as? String == model.id }),
                let raw = try? JSONSerialization.data(withJSONObject: entry),
                let text = String(data: raw, encoding: .utf8)
            else { continue }
            offers[model.id] = (newer, text)
        }
        return offers
    }

    // What the model's row offers to replace: an installed older revision, or the newer published one.
    func update(for model: CatalogModel) -> CatalogModel? {
        if let offer = offers[model.id],
            Self.localBundle(for: model) != nil || Self.hasOlderBundle(model)
                || Self.hasOlderBundle(offer.model)
        {
            return offer.model
        }
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
    private(set) var completed = 0  // downloads that finished since launch
    private var tasks: [String: Task<Void, Error>] = [:]

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
            completed += 1
        }
        tasks[model] = task
        try await task.value
    }

    // Stops a download and waits for it to end; its partial files stay until the model is removed.
    // Those waiting for it see a CancellationError.
    func stop(_ model: String) async {
        guard let task = tasks[model] else { return }
        task.cancel()
        _ = try? await task.value
    }

    private func run(_ model: String, entry: String?) async throws {
        struct Update: Decodable { let progress: Double? }
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
        try await withTaskCancellationHandler {
            // Read both pipes at once so neither fills up and stalls the other.
            async let failure = errors.fileHandleForReading.bytes.reduce(into: Data()) { $0.append($1) }
            for try await line in output.fileHandleForReading.bytes.lines {
                if let value = (try? JSONDecoder().decode(Update.self, from: Data(line.utf8)))?.progress {
                    progress[model] = value
                }
            }
            let details = String(decoding: try await failure, as: UTF8.self)
            await process.exited()
            try Task.checkCancellation()
            guard process.terminationStatus == 0 else {
                throw TinkerboxError("Could not download the model.\n\(details.suffix(2000))")
            }
        } onCancel: {
            Runtime.stop(process)
        }
    }
}
