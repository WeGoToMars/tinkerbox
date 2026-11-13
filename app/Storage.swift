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

    // An earlier revision counts only what deleting it frees: the current one is measured first.
    struct Bundle: Decodable, Hashable, Sendable {
        let revision: String
        let path: String
        let bytes: Int64
        let current: Bool  // the revision the catalog loads; others are left from earlier versions
        let compiled: Compiled
    }

    struct Model: Decodable, Identifiable, Sendable {
        let id: String
        let name: String
        let bundles: [Bundle]
        let bytes: Int64

        func bytes(_ kind: StorageKind) -> Int64 {
            switch kind {
            case .bundles: bundles.filter(\.current).map(\.bytes).reduce(0, +)
            case .compiled: bundles.filter(\.current).map(\.compiled.bytes).reduce(0, +)
            case .outdated: bundles.filter { !$0.current }.map { $0.bytes + $0.compiled.bytes }.reduce(0, +)
            case .checkpoints, .unused, .python: 0
            }
        }
    }

    // An upstream checkpoint in the Hugging Face cache; exports and earlier versions download it.
    struct Checkpoint: Decodable, Identifiable, Sendable {
        let repo: String
        let path: String
        let bytes: Int64
        let models: [String]  // installed models converted from it
        var id: String { repo }
    }

    let models: [Model]  // models with files on this Mac
    let checkpoints: [Checkpoint]
    // Of deleted bundles, local exports and earlier macOS builds.
    let unusedCompiled: Compiled
    let python: Files
    let packages: Files
    let bytes: Int64

    func bytes(_ kind: StorageKind) -> Int64 {
        switch kind {
        case .bundles, .compiled, .outdated: models.map { $0.bytes(kind) }.reduce(0, +)
        case .checkpoints: checkpoints.map(\.bytes).reduce(0, +)
        case .unused: unusedCompiled.bytes
        case .python: python.bytes + packages.bytes
        }
    }

    // What deleting a model removes, for its confirmation. A checkpoint goes only with its last model.
    func removal(of model: Model) -> String {
        var parts = model.bundles.map { "the \($0.revision) bundle (\(formatBytes($0.bytes)))" }
        let compiled = model.bundles.map(\.compiled.bytes).reduce(0, +)
        if compiled > 0 { parts.append("its compiled copies (\(formatBytes(compiled)))") }
        if let checkpoint = checkpoints.first(where: { $0.models == [model.id] }) {
            parts.append("the \(checkpoint.repo) checkpoint (\(formatBytes(checkpoint.bytes)))")
        }
        return "Deletes \(ListFormatter.localizedString(byJoining: parts))."
    }
}

// Where storage goes, in the order the page draws it, each in its own series color.
enum StorageKind: CaseIterable, Identifiable {
    case bundles, compiled, outdated, checkpoints, unused, python

    var id: Self { self }

    var title: String {
        switch self {
        case .bundles: "Model bundles"
        case .compiled: "Compiled for this Mac"
        case .outdated: "Earlier revisions"
        case .checkpoints: "Upstream checkpoints"
        case .unused: "Unused compiled copies"
        case .python: "Python and packages"
        }
    }

    var detail: String {
        switch self {
        case .bundles: "The models you downloaded"
        case .compiled: "Core AI's Neural Engine copies, beyond the blocks they share with bundles"
        case .outdated: "Left by updates; this version no longer loads them"
        case .checkpoints: "Original weights in the Hugging Face cache, only needed for exports"
        case .unused: "Of deleted bundles, local exports and earlier macOS builds"
        case .python: "Shared by all models"
        }
    }

    var color: Color { Color.series[Self.allCases.firstIndex(of: self)!] }
}

extension Color {
    // A categorical palette validated for color-vision deficiency in both appearances, in the order
    // series take it; neighbors stay distinguishable.
    static let series = [
        Color(light: 0x2a78d6, dark: 0x3987e5), Color(light: 0xeb6834, dark: 0xd95926),
        Color(light: 0x1baf7a, dark: 0x199e70), Color(light: 0xeda100, dark: 0xc98500),
        Color(light: 0xe87ba4, dark: 0xd55181), Color(light: 0x008300, dark: 0x008300),
        Color(light: 0x6250d6, dark: 0x9085e9), Color(light: 0xe34948, dark: 0xe66767),
    ]

    init(light: UInt32, dark: UInt32) {
        self.init(
            nsColor: NSColor(name: nil) { appearance in
                let hex = appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua ? dark : light
                return NSColor(
                    srgbRed: CGFloat(hex >> 16 & 0xff) / 255, green: CGFloat(hex >> 8 & 0xff) / 255,
                    blue: CGFloat(hex & 0xff) / 255, alpha: 1)
            })
    }
}

func formatBytes(_ bytes: Int64) -> String {
    ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
}

@MainActor @Observable final class Storage {
    private(set) var report: StorageReport?
    private(set) var removing: Set<String> = []  // model IDs, checkpoint repos, or the keys below
    private(set) var measuring = true  // until the first report, and while a newer one is on its way
    var error: String?
    private var generation = 0

    nonisolated static let unused = "unused compiled copies"
    nonisolated static let outdated = "earlier revisions"
    nonisolated static let checkpoints = "upstream checkpoints"

    func refresh() async {
        guard removing.isEmpty else { return }
        await update([])
    }

    func removeUnused() async { await remove(Self.unused, ["--remove-unused"]) }

    // Of every model.
    func removeOutdated() async { await remove(Self.outdated, ["--outdated"]) }

    func removeCheckpoint(_ repo: String) async { await remove(repo, ["--remove-checkpoint", repo]) }

    func removeCheckpoints() async {
        await remove(Self.checkpoints, ["--remove-checkpoint"] + (report?.checkpoints.map(\.repo) ?? []))
    }

    // Deletes a model's bundles, their compiled copies and a checkpoint no other model uses; with `outdated`,
    // only bundles of revisions the catalog no longer loads.
    func remove(_ model: String, outdated: Bool = false) async {
        await remove(model, ["--remove", model] + (outdated ? ["--outdated"] : []))
    }

    private func remove(_ key: String, _ arguments: [String]) async {
        guard removing.isEmpty else { return }
        removing.insert(key)
        defer { removing.remove(key) }
        await update(arguments)
    }

    // The latest call's report wins, so a slow scan never replaces a newer one.
    private func update(_ arguments: [String]) async {
        generation += 1
        let current = generation
        measuring = true
        defer { if current == generation { measuring = false } }
        do {
            let report = try await Self.run(arguments)
            if current == generation {
                withAnimation { self.report = report }
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
}

// One whole split into parts, each as wide as its share; hovering a part highlights its kind everywhere.
struct StorageBar: View {
    let parts: [(kind: StorageKind, bytes: Int64)]
    @Binding var highlighted: StorageKind?

    var body: some View {
        let parts = parts.filter { $0.bytes > 0 }
        let total = Double(max(1, parts.map(\.bytes).reduce(0, +)))
        GeometryReader { geometry in
            let width = geometry.size.width - CGFloat(max(0, parts.count - 1)) * 2
            HStack(spacing: 2) {
                ForEach(parts, id: \.kind) { part in
                    Capsule()
                        .fill(part.kind.color)
                        .frame(width: max(geometry.size.height, width * Double(part.bytes) / total))  // a small part stays a dot
                        .opacity(highlighted == nil || highlighted == part.kind ? 1 : 0.25)
                        .onHover { highlight(part.kind, $0) }
                        .help(
                            "\(part.kind.title): \(formatBytes(part.bytes)), "
                                + (Double(part.bytes) / total).formatted(.percent.precision(.fractionLength(0))))
                }
            }
        }
        .animation(.easeOut(duration: 0.15), value: highlighted)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(parts.map { "\($0.kind.title), \(formatBytes($0.bytes))" }.joined(separator: "; "))
    }

    // Leaving one part can arrive after entering the next, so only a part's own highlight ends with it.
    private func highlight(_ kind: StorageKind, _ hovering: Bool) {
        if hovering { highlighted = kind } else if highlighted == kind { highlighted = nil }
    }
}

struct StorageView: View {
    let demo: Demo
    let showModels: () -> Void
    @State private var highlighted: StorageKind?
    @State private var pending: Removal?
    private var storage: Storage { demo.storage }

    private enum Removal: Identifiable {
        case model(StorageReport.Model), checkpoint(StorageReport.Checkpoint), checkpoints, outdated, unused

        var id: String {
            switch self {
            case .model(let model): "model \(model.id)"
            case .checkpoint(let checkpoint): "checkpoint \(checkpoint.repo)"
            case .checkpoints: Storage.checkpoints
            case .outdated: Storage.outdated
            case .unused: Storage.unused
            }
        }
    }

    var body: some View {
        Group {
            if let report = storage.report {
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        overview(report)
                        models(report)
                        if !report.checkpoints.isEmpty { checkpoints(report) }
                    }
                    .padding(24)
                    .frame(maxWidth: 780)
                    .frame(maxWidth: .infinity)
                }
            } else if storage.measuring {
                ProgressView()
            } else {
                ContentUnavailableView {
                    Label("Storage Unavailable", systemImage: "internaldrive")
                } description: {
                    Text("Tinkerbox could not measure its files.")
                } actions: {
                    Button("Try Again") { Task { await storage.refresh() } }
                }
            }
        }
        .frame(minWidth: 560, minHeight: 500)
        .navigationTitle("Storage")
        .navigationSubtitle(available)
        .toolbar {
            if !storage.removing.isEmpty { ProgressView().controlSize(.small) }
            Button("Refresh", systemImage: "arrow.clockwise") { Task { await storage.refresh() } }
                .disabled(!storage.removing.isEmpty)
        }
        .confirmationDialog(
            title(of: pending), isPresented: Binding(get: { pending != nil }, set: { if !$0 { pending = nil } }),
            presenting: pending
        ) { removal in
            Button("Delete", role: .destructive) { Task { await remove(removal) } }
        } message: { removal in
            Text(message(of: removal))
        }
        .task { await storage.refresh() }
    }

    private var available: String {
        let values = try? URL.homeDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage.map { "\(formatBytes($0)) available on this Mac" } ?? ""
    }

    private func overview(_ report: StorageReport) -> some View {
        let kinds = StorageKind.allCases.filter { report.bytes($0) > 0 }
        return VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(formatBytes(report.bytes)).font(.system(size: 48, weight: .semibold))
                    .contentTransition(.numericText())
                Text("used by Tinkerbox").foregroundStyle(.secondary)
            }
            StorageBar(parts: kinds.map { ($0, report.bytes($0)) }, highlighted: $highlighted)
                .frame(height: 8)
            TaskSection {
                ForEach(kinds) { kind in
                    if kind != kinds[0] { Divider() }
                    kindRow(kind, bytes: report.bytes(kind))
                }
            }
        }
    }

    private func kindRow(_ kind: StorageKind, bytes: Int64) -> some View {
        HStack(spacing: 10) {
            Circle().fill(kind.color).frame(width: 10, height: 10)
            VStack(alignment: .leading, spacing: 2) {
                Text(kind.title)
                Text(kind.detail).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            switch kind {
            case .outdated: removeButton(.outdated, key: Storage.outdated, title: "Remove")
            case .checkpoints: removeButton(.checkpoints, key: Storage.checkpoints, title: "Delete All")
            case .unused: removeButton(.unused, key: Storage.unused, title: "Clean Up")
            default: EmptyView()
            }
            Text(formatBytes(bytes)).monospacedDigit().contentTransition(.numericText())
                .frame(minWidth: 72, alignment: .trailing)
        }
        .contentShape(Rectangle())
        .onHover { highlight(kind, $0) }
        .background(highlighted == kind ? Color.primary.opacity(0.05) : .clear, in: .rect(cornerRadius: 6))
    }

    private func highlight(_ kind: StorageKind, _ hovering: Bool) {
        if hovering { highlighted = kind } else if highlighted == kind { highlighted = nil }
    }

    private func models(_ report: StorageReport) -> some View {
        let models = report.models.sorted { $0.bytes > $1.bytes }
        let largest = Double(max(1, models.first?.bytes ?? 1))
        return TaskSection("Models") {
            if models.isEmpty {
                ContentUnavailableView {
                    Label("No Models Downloaded", systemImage: "square.stack.3d.up")
                } actions: {
                    Button("Browse Models", action: showModels)
                }
            }
            ForEach(models) { model in
                if model.id != models[0].id { Divider() }
                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(model.name)
                        GeometryReader { geometry in
                            StorageBar(
                                parts: [StorageKind.bundles, .compiled, .outdated].map { ($0, model.bytes($0)) },
                                highlighted: $highlighted
                            )
                            .frame(width: geometry.size.width * Double(model.bytes) / largest)
                        }
                        .frame(height: 6)
                    }
                    if storage.removing.contains(model.id) { ProgressView().controlSize(.small) }
                    Text(formatBytes(model.bytes)).monospacedDigit().contentTransition(.numericText())
                        .frame(minWidth: 72, alignment: .trailing)
                    Menu("Actions for \(model.name)", systemImage: "ellipsis.circle") { actions(for: model) }
                        .labelStyle(.iconOnly)
                        .menuStyle(.button)
                        .menuIndicator(.hidden)
                        .buttonStyle(.borderless)
                        .fixedSize()
                }
                .contentShape(Rectangle())
                .contextMenu { actions(for: model) }
            }
        }
    }

    @ViewBuilder private func actions(for model: StorageReport.Model) -> some View {
        Button("Show in Finder") {
            NSWorkspace.shared.activateFileViewerSelecting(model.bundles.map { URL(fileURLWithPath: $0.path) })
        }
        if model.bytes(.outdated) > 0 {
            Button("Remove Earlier Revisions (\(formatBytes(model.bytes(.outdated))))") {
                Task { await demo.remove(model.id, outdated: true) }
            }
            .disabled(!storage.removing.isEmpty)
        }
        Divider()
        Button("Delete…", role: .destructive) { pending = .model(model) }
            .disabled(demo.keeps(model.id) || !storage.removing.isEmpty)
    }

    private func checkpoints(_ report: StorageReport) -> some View {
        let names = Dictionary(report.models.map { ($0.id, $0.name) }, uniquingKeysWith: { first, _ in first })
        return TaskSection(StorageKind.checkpoints.title) {
            ForEach(report.checkpoints) { checkpoint in
                if checkpoint.id != report.checkpoints[0].id { Divider() }
                HStack(spacing: 10) {
                    Circle().fill(StorageKind.checkpoints.color).frame(width: 10, height: 10)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(checkpoint.repo)
                        Text(
                            checkpoint.models.isEmpty
                                ? "No installed model"
                                : ListFormatter.localizedString(byJoining: checkpoint.models.map { names[$0] ?? $0 })
                        )
                        .font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Show in Finder", systemImage: "folder") {
                        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: checkpoint.path)])
                    }
                    .labelStyle(.iconOnly).buttonStyle(.borderless)
                    removeButton(.checkpoint(checkpoint), key: checkpoint.repo, title: "Delete")
                    Text(formatBytes(checkpoint.bytes)).monospacedDigit().frame(minWidth: 72, alignment: .trailing)
                }
            }
        }
    }

    @ViewBuilder private func removeButton(_ removal: Removal, key: String, title: String) -> some View {
        if storage.removing.contains(key) {
            ProgressView().controlSize(.small)
        } else {
            Button(title) { pending = removal }.disabled(!storage.removing.isEmpty)
        }
    }

    private func title(of removal: Removal?) -> String {
        switch removal {
        case .model(let model): "Delete \(model.name)?"
        case .checkpoint(let checkpoint): "Delete the \(checkpoint.repo) checkpoint?"
        case .checkpoints: "Delete all upstream checkpoints?"
        case .outdated: "Delete earlier revisions?"
        case .unused: "Delete unused compiled copies?"
        case nil: ""
        }
    }

    private func message(of removal: Removal) -> String {
        switch removal {
        case .model(let model): storage.report?.removal(of: model) ?? ""
        case .checkpoint: "Exporting this model downloads it again."
        case .checkpoints: "Tinkerbox never loads them; exporting a model downloads its checkpoint again."
        case .outdated: "Deletes the revisions this version no longer loads, for every model."
        case .unused: "Local exports may need to compile again."
        }
    }

    private func remove(_ removal: Removal) async {
        switch removal {
        case .model(let model): await demo.remove(model.id)
        case .checkpoint(let checkpoint): await storage.removeCheckpoint(checkpoint.repo)
        case .checkpoints: await storage.removeCheckpoints()
        case .outdated:
            await storage.removeOutdated()
            demo.catalog.scanDownloads()
        case .unused: await storage.removeUnused()
        }
    }
}
