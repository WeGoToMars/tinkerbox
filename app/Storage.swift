import AppKit
import SwiftUI

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

    var downloadedModels: Int { report?.models.filter { !$0.bundles.isEmpty }.count ?? 0 }

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
        try await Task.detached {
            let process = Process(), output = Pipe(), errors = Pipe()
            Runtime.configure(process, python: ["-m", "tinkerbox_ane.storage"] + arguments)
            process.environment!["TINKERBOX_CATALOG_UPDATES"] = Catalog.updatesEnvironment()
            process.standardInput = FileHandle.nullDevice
            process.standardOutput = output
            process.standardError = errors
            try process.run()
            // Read both pipes at once so neither fills up and stalls the other.
            async let failure = Task.detached { errors.fileHandleForReading.readDataToEndOfFile() }.value
            let data = output.fileHandleForReading.readDataToEndOfFile()
            let details = String(decoding: await failure, as: UTF8.self)
            process.waitUntilExit()
            guard process.terminationStatus == 0 else {
                throw NSError(domain: "Tinkerbox", code: 3, userInfo: [
                    NSLocalizedDescriptionKey: "Could not measure storage.\n\(details.suffix(2000))"])
            }
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            return try decoder.decode(StorageReport.self, from: data)
        }.value
    }
}

// The bottom of the sidebar: how much the models use, opening the Storage page.
struct StorageSummary: View {
    let storage: Storage
    let open: () -> Void

    var body: some View {
        Button(action: open) {
            VStack(alignment: .leading, spacing: 2) {
                Text(storage.report.map { "\(formatBytes($0.bytes)) used" } ?? "Measuring storage…")
                if storage.report != nil {
                    Text("\(storage.downloadedModels) \(storage.downloadedModels == 1 ? "model" : "models") downloaded")
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .monospacedDigit()
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .contentShape(.rect)
        }
        .buttonStyle(.plain)
        .help("Show storage")
    }
}

// Every file the models keep on this Mac, with what deleting a model removes.
struct StorageView: View {
    let demo: Demo
    @State private var pending: StorageReport.Model?
    @State private var confirmingUnused = false

    var body: some View {
        let storage = demo.storage
        List {
            if let report = storage.report {
                Section("Models") {
                    if report.models.isEmpty {
                        Text("No downloaded models.").foregroundStyle(.secondary)
                    }
                    ForEach(report.models) { model in
                        row(model).padding(.vertical, 6)
                    }
                }
                Section("Shared files") {
                    files("Unused compiled copies", detail: "Core AI copies of deleted bundles, local exports and "
                          + "earlier macOS builds that no downloaded model uses.",
                          paths: report.unusedCompiled.paths, bytes: report.unusedCompiled.bytes) {
                        if storage.removing.contains(Storage.unused) {
                            ProgressView().controlSize(.small)
                        } else {
                            Button("Remove") { confirmingUnused = true }
                                .disabled(report.unusedCompiled.paths.isEmpty || !storage.removing.isEmpty)
                        }
                    }
                    files("Python", detail: "The interpreter, managed by uv.",
                          paths: [report.python.path], bytes: report.python.bytes) {}
                    files("Python packages", detail: "tinkerbox-ane and its dependencies, linked from uv's cache.",
                          paths: [report.packages.path].filter { !$0.isEmpty }, bytes: report.packages.bytes) {}
                }
            } else if storage.error == nil {
                ProgressView("Measuring…").frame(maxWidth: .infinity)
            }
        }
        .frame(minWidth: 650, minHeight: 650)
        .navigationTitle("Storage")
        .navigationSubtitle(storage.report.map { "\(formatBytes($0.bytes)) used" } ?? "")
        .toolbar {
            Button("Refresh", systemImage: "arrow.clockwise") { Task { await storage.refresh() } }
        }
        .confirmationDialog("Delete \(pending?.name ?? "")?", isPresented: Binding(
            get: { pending != nil }, set: { if !$0 { pending = nil } }), presenting: pending) { model in
            Button("Delete \(formatBytes(model.bytes))", role: .destructive) {
                Task { await demo.remove(model.id) }
            }
        } message: { model in
            Text(Self.removal(of: model))
        }
        .confirmationDialog("Remove unused compiled copies?", isPresented: $confirmingUnused) {
            Button("Remove \(formatBytes(storage.report?.unusedCompiled.bytes ?? 0))", role: .destructive) {
                Task { await storage.removeUnused() }
            }
        } message: {
            Text("Local model exports may need to compile again.")
        }
        .task { await storage.refresh() }
    }

    // What deleting a model removes, for its confirmation.
    static func removal(of model: StorageReport.Model) -> String {
        var parts = model.bundles.map { "the \($0.revision) bundle (\(formatBytes($0.bytes)))" }
        let compiled = model.bundles.map(\.compiled.bytes).reduce(0, +)
        if compiled > 0 { parts.append("its compiled copies (\(formatBytes(compiled)))") }
        if let checkpoint = model.checkpoint {
            parts.append("the \(checkpoint.repo) checkpoint (\(formatBytes(checkpoint.bytes)))")
        }
        return "Deletes \(ListFormatter.localizedString(byJoining: parts)). Using this model again requires a download."
    }

    private func row(_ model: StorageReport.Model) -> some View {
        let kept = demo.keeps(model.id)
        let removing = demo.storage.removing.contains(model.id)
        let outdated = model.bundles.filter { !$0.current }
        return HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .firstTextBaseline) {
                    Text(model.name).font(.headline)
                    Text(formatBytes(model.bytes)).foregroundStyle(.secondary).monospacedDigit()
                }
                ForEach(model.bundles, id: \.self) { bundle in
                    location("Core AI bundle \(bundle.revision)\(bundle.current ? "" : " · outdated")",
                             paths: [bundle.path], bytes: bundle.bytes)
                    if !bundle.compiled.paths.isEmpty {
                        location("Compiled for this Mac", paths: bundle.compiled.paths, bytes: bundle.compiled.bytes)
                    }
                }
                if let checkpoint = model.checkpoint {
                    location("Upstream checkpoint \(checkpoint.repo)", paths: [checkpoint.path], bytes: checkpoint.bytes)
                }
            }
            Spacer()
            if removing {
                ProgressView().controlSize(.small)
            } else {
                if let progress = demo.downloads.progress[model.id] {
                    Text("Downloading \(Int(progress * 100))%").font(.caption).foregroundStyle(.secondary).monospacedDigit()
                } else if kept {
                    let runtime = demo.runtime.state
                    Text(model.id != runtime.model ? "Selected" : runtime.loading ? "Loading" : "In use")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if !outdated.isEmpty && model.bundles.contains(where: \.current) {
                    Button("Remove Outdated") { Task { await demo.remove(model.id, outdated: true) } }
                        .disabled(!demo.storage.removing.isEmpty)
                        .help("Delete \(formatBytes(outdated.map { $0.bytes + $0.compiled.bytes }.reduce(0, +))) "
                              + "of bundles this version no longer loads")
                }
                if !kept {
                    Button(role: .destructive) { pending = model } label: { Image(systemName: "trash") }
                        .buttonStyle(.borderless)
                        .help(demo.downloads.contains(model.id) ? "Stop the download and delete its files"
                              : "Delete model files and cached copies")
                        .disabled(!demo.storage.removing.isEmpty)
                }
            }
        }
    }

    private func location(_ title: String, paths: [String], bytes: Int64) -> some View {
        HStack(spacing: 6) {
            Text("\(title) · \(formatBytes(bytes))")
            finder(paths)
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }

    private func files(_ title: String, detail: String, paths: [String], bytes: Int64,
                       @ViewBuilder action: () -> some View) -> some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text(title).font(.headline)
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Text(formatBytes(bytes)).foregroundStyle(.secondary).monospacedDigit()
            finder(paths)
            action()
        }
        .padding(.vertical, 6)
    }

    @ViewBuilder private func finder(_ paths: [String]) -> some View {
        if let path = paths.first {
            Button("Show in Finder", systemImage: "folder") {
                NSWorkspace.shared.activateFileViewerSelecting(paths.map { URL(fileURLWithPath: $0) })
            }
            .labelStyle(.iconOnly)
            .buttonStyle(.borderless)
            .help(paths.count == 1 ? path : "\(paths.count) folders, such as \(path)")
        }
    }
}
