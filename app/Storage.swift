import Foundation

// What models keep on this Mac, as `python -m tinkerbox_ane.storage` reports it. Sizes are allocated bytes.
struct StorageReport: Decodable, Sendable {
    struct Files: Decodable, Sendable {
        let path: String
        let bytes: Int64
    }

    // Core AI's copies of assets, specialized for this Mac's Neural Engine; one folder per asset.
    struct Compiled: Decodable, Hashable, Sendable {
        let paths: [String]
        let bytes: Int64
    }

    struct Bundle: Decodable, Hashable, Sendable {
        let revision: String
        let path: String
        let bytes: Int64
        let current: Bool  // the revision the catalog loads; others are left from earlier versions
        let compiled: Compiled
    }

    struct Checkpoint: Decodable, Sendable {
        let repo: String
        let path: String
        let bytes: Int64
    }

    struct Model: Decodable, Identifiable, Sendable {
        let id: String
        let name: String
        let bundles: [Bundle]
        let checkpoint: Checkpoint?  // upstream, in the Hugging Face cache: exports and earlier versions download it
        let bytes: Int64
    }

    let models: [Model]  // models with files on this Mac
    // Of deleted bundles, local exports and earlier macOS builds.
    let unusedCompiled: Compiled
    let python: Files
    let packages: Files
    let bytes: Int64
}

func formatBytes(_ bytes: Int64) -> String {
    ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
}

@MainActor @Observable final class Storage {
    private(set) var report: StorageReport?
    private(set) var removing: Set<String> = []
    var error: String?
    private var generation = 0

    // A key of `removing` while unused compiled copies are deleted.
    static let unused = "unused compiled copies"

    func refresh() async {
        guard removing.isEmpty else { return }
        await update([])
    }

    func removeUnused() async {
        guard removing.isEmpty else { return }
        removing.insert(Self.unused)
        defer { removing.remove(Self.unused) }
        await update(["--remove-unused"])
    }

    // Deletes a model's bundles, their compiled copies and its checkpoint; with `outdated`, only
    // bundles of revisions the catalog no longer loads.
    func remove(_ model: String, outdated: Bool = false) async {
        guard removing.isEmpty else { return }
        removing.insert(model)
        defer { removing.remove(model) }
        await update(["--remove", model] + (outdated ? ["--outdated"] : []))
    }

    // The latest call's report wins, so a slow scan never replaces a newer one.
    private func update(_ arguments: [String]) async {
        generation += 1
        let current = generation
        do {
            let report = try await Self.run(arguments)
            if current == generation {
                self.report = report
                self.error = nil
            }
        } catch {
            if current == generation { self.error = error.localizedDescription }
        }
    }

    nonisolated private static func run(_ arguments: [String]) async throws -> StorageReport {
        let process = Process(), output = Pipe(), errors = Pipe()
        Runtime.configure(process, python: ["-m", "tinkerbox_ane.storage"] + arguments)
        process.environment!["TINKERBOX_CATALOG_UPDATES"] = Catalog.updatesEnvironment()
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = output
        process.standardError = errors
        try process.run()
        // Read both pipes at once so neither fills up and stalls the other.
        async let failure = errors.fileHandleForReading.bytes.reduce(into: Data()) { $0.append($1) }
        let data = try await output.fileHandleForReading.bytes.reduce(into: Data()) { $0.append($1) }
        let details = String(decoding: try await failure, as: UTF8.self)
        await process.exited()
        guard process.terminationStatus == 0 else {
            throw TinkerboxError("Could not measure storage.\n\(details.suffix(2000))")
        }
        return try Runtime.decoder.decode(StorageReport.self, from: data)
    }

    // What deleting a model removes, for its confirmation.
    static func removal(of model: StorageReport.Model) -> String {
        var parts = model.bundles.map { "the \($0.revision) bundle (\(formatBytes($0.bytes)))" }
        let compiled = model.bundles.map(\.compiled.bytes).reduce(0, +)
        if compiled > 0 { parts.append("its compiled copies (\(formatBytes(compiled)))") }
        if let checkpoint = model.checkpoint {
            parts.append("the \(checkpoint.repo) checkpoint (\(formatBytes(checkpoint.bytes)))")
        }
        return "Deletes \(ListFormatter.localizedString(byJoining: parts))."
    }
}
