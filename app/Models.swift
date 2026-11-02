import AppKit
import SwiftUI

private struct ModelFamily: Identifiable {
    let id: String
    let name: String
    var models: [CatalogModel]

    func variant(size: String, quantization: String?) -> CatalogModel? {
        let candidates = models.filter { ($0.size ?? "Default") == size }
        return candidates.first { $0.quantization == quantization } ?? candidates.first
    }
}

// Families are grouped by task; selectors browse variants before downloading or activating one.
// Each downloaded model shows what it keeps on this Mac.
struct ModelsView: View {
    let demo: Demo
    @State private var removalError: String?
    @State private var pending: CatalogModel?
    @State private var confirmingCleanup = false
    @State private var choosingBundle: CatalogModel?
    @State private var variants: [String: String] = [:]
    @Environment(\.openURL) private var openURL

    // In catalog order; the remote catalog can list tasks this app does not run yet.
    private var tasks: [String] {
        demo.catalog.models.map(\.task).reduce(into: []) { tasks, task in
            if !tasks.contains(task) { tasks.append(task) }
        }
    }

    private func families(for task: String) -> [ModelFamily] {
        var families: [ModelFamily] = []
        for model in demo.catalog.models where model.task == task {
            let id = model.family.map { "\(task)/\(model.publisher)/\($0)" } ?? "\(task)/\(model.id)"
            if let index = families.firstIndex(where: { $0.id == id }) { families[index].models.append(model) }
            else { families.append(ModelFamily(id: id, name: model.family ?? model.name, models: [model])) }
        }
        return families
    }

    private func selected(in family: ModelFamily) -> CatalogModel {
        family.models.first { $0.id == variants[family.id] }
            ?? family.models.first { $0.id == demo.model }
            ?? family.models.first { demo.isDownloaded($0.id) }
            ?? family.models[0]
    }

    private func familyRow(_ family: ModelFamily) -> some View {
        let model = selected(in: family)
        let sizes = family.models.map { $0.size ?? "Default" }.reduce(into: [String]()) {
            if !$0.contains($1) { $0.append($1) }
        }
        let weights = family.models.filter { $0.size == model.size }
        let quantizations = weights.map { $0.quantization ?? "Default" }.reduce(into: [String]()) {
            if !$0.contains($1) { $0.append($1) }
        }
        let installed = family.models.filter { demo.isDownloaded($0.id) }.count
        let downloading = family.models.filter { demo.downloads.contains($0.id) }.count
        return VStack(alignment: .leading, spacing: 12) {
            row(model, title: family.name)
            if family.models.count > 1 {
                HStack(spacing: 16) {
                    if sizes.count > 1 {
                        Picker("Size", selection: Binding(get: { model.size ?? "Default" }, set: { size in
                            variants[family.id] = family.variant(size: size, quantization: model.quantization)?.id
                        })) {
                            ForEach(sizes, id: \.self) { size in
                                let variant = family.variant(size: size, quantization: model.quantization)
                                Text([size, variant?.fileSize].compactMap { $0 }.joined(separator: " · ")).tag(size)
                            }
                        }.fixedSize().accessibilityLabel("\(family.name) size")
                    }
                    if quantizations.count > 1 {
                        Picker("Quantization", selection: Binding(get: { model.quantization ?? "Default" }, set: { quantization in
                            variants[family.id] = weights.first { ($0.quantization ?? "Default") == quantization }?.id
                        })) {
                            ForEach(quantizations, id: \.self) { quantization in
                                let variant = weights.first { ($0.quantization ?? "Default") == quantization }
                                Text([quantization, variant?.fileSize].compactMap { $0 }.joined(separator: " · ")).tag(quantization)
                            }
                        }.fixedSize().accessibilityLabel("\(family.name) quantization")
                    }
                    Spacer()
                    Text("\(installed) of \(family.models.count) installed\(downloading > 0 ? " · \(downloading) downloading" : "")")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        }.padding(.vertical, 8)
    }

    var body: some View {
        List {
            ForEach(tasks, id: \.self) { task in
                Section(ModelTask(rawValue: task)?.heading ?? task.capitalized) {
                    ForEach(families(for: task)) { family in familyRow(family) }
                }
            }
        }
        .frame(minWidth: 650, minHeight: 650)
        .navigationTitle("Models")
        .navigationSubtitle(subtitle)
        .toolbar {
            if let unused = demo.storage.report?.unusedCompiled, !unused.paths.isEmpty {
                Button("Clean Up \(formatBytes(unused.bytes))", systemImage: "trash") { confirmingCleanup = true }
                    .disabled(!demo.storage.removing.isEmpty)
                    .help("Delete compiled copies that no downloaded model uses")
            }
            Button("Refresh", systemImage: "arrow.clockwise") {
                Task {
                    await demo.catalog.refresh()
                    await demo.storage.refresh()
                }
            }
        }
        .alert("Model files", isPresented: Binding(get: { removalError != nil }, set: { if !$0 { removalError = nil } })) {
        } message: { Text(removalError ?? "") }
        .alert("Could not update storage", isPresented: Binding(get: { demo.storage.error != nil },
                                                                set: { if !$0 { demo.storage.error = nil } })) {
        } message: { Text(demo.storage.error ?? "") }
        .confirmationDialog("Delete \(pending?.name ?? "")?", isPresented: Binding(
            get: { pending != nil }, set: { if !$0 { pending = nil } }), presenting: pending) { model in
            Button("Delete", role: .destructive) { Task { await demo.remove(model.id) } }
        } message: { model in
            if let stored = demo.storage.report?.models.first(where: { $0.id == model.id }) {
                Text(Storage.removal(of: stored))
            }
        }
        .confirmationDialog("Delete unused compiled copies?", isPresented: $confirmingCleanup) {
            Button("Delete", role: .destructive) { Task { await demo.storage.removeUnused() } }
        } message: {
            Text("Core AI copies of deleted bundles and earlier macOS builds. Local exports may need to compile again.")
        }
        .fileImporter(isPresented: Binding { choosingBundle != nil } set: { if !$0 { choosingBundle = nil } },
                      allowedContentTypes: [.folder]) { result in
            guard let model = choosingBundle, case .success(let url) = result else { return }
            if let issue = Catalog.bundleIssue(at: url) { removalError = issue }
            else { Task { await demo.choose(model, asset: url) } }
        }
        .task {
            await demo.catalog.refresh()
            await demo.storage.refresh()
        }
    }

    // Disk use, and where the catalog came from when it is the built-in copy.
    private var subtitle: String {
        let used = demo.storage.report.map { report in
            "\(formatBytes(report.bytes)) on disk, including \(formatBytes(report.python.bytes + report.packages.bytes)) of Python"
        }
        return [used, demo.catalog.isRemote ? nil : "Offline catalog"].compactMap { $0 }.joined(separator: " · ")
    }

    private func row(_ model: CatalogModel, title: String) -> some View {
        let downloaded = demo.isDownloaded(model.id)
        let downloading = demo.downloads.contains(model.id)
        let issue = Catalog.availabilityIssue(for: model)
        let files = demo.storage.report?.models.first { $0.id == model.id }
        let outdated = files?.bundles.filter { !$0.current } ?? []
        // Partial downloads count, so an interrupted one can be deleted.
        let stored = demo.catalog.downloaded.contains(model.id) || downloading || files != nil
        let inputs = model.inputs?.map { $0 == "image" ? "Images" : $0.capitalized }.joined(separator: " + ")
        let details = [inputs, model.context.map { "\($0 / 1024)K-token context" },
                       model.fileSize,
                       model.license].compactMap { $0 }
        return HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .firstTextBaseline) {
                    Text(title).font(.headline)
                    Text(model.publisher).foregroundStyle(.secondary)
                }
                Text(model.summary)
                Text(details.joined(separator: " · ")).font(.caption).foregroundStyle(.secondary)
                if let files {
                    HStack(spacing: 6) {
                        Text("\(formatBytes(files.bytes)) on this Mac")
                        Button("Show in Finder", systemImage: "folder") {
                            NSWorkspace.shared.activateFileViewerSelecting(files.bundles.map { URL(fileURLWithPath: $0.path) })
                        }
                        .labelStyle(.iconOnly)
                        .buttonStyle(.borderless)
                        .disabled(files.bundles.isEmpty)
                    }
                    .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                }
                if issue == nil, let update = demo.catalog.update(for: model) {
                    Text("Update available · \(update.revision)")
                        .font(.caption).foregroundStyle(.tint)
                }
                if model.isSupported, let issue {
                    Text(issue).font(.caption).foregroundStyle(.secondary)
                }
            }
            Spacer()
            Menu("Hugging Face", systemImage: "arrow.up.right.square") {
                let repositories = [("Base model", model.baseModel), ("Original model", model.originalModel),
                                    ("Converted model", model.repo)]
                ForEach(repositories, id: \.0) { title, repository in
                    if let repository {
                        Button("\(title) · \(repository)") {
                            openURL(URL(string: "https://huggingface.co")!.appending(path: repository))
                        }
                    }
                }
            }
            .labelStyle(.iconOnly)
            .menuIndicator(.hidden)
            .menuStyle(.button)
            .buttonStyle(.borderless)
            .fixedSize()
            .help("Open on Hugging Face")
            if !model.isSupported {
                Text(Catalog.isNewer(model.minVersion, than: Catalog.appVersion)
                     ? "Requires Tinkerbox \(model.minVersion)" : "Update Tinkerbox to use this model")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                if issue != nil {
                    Button("Choose bundle…") { choosingBundle = model }
                        .disabled(demo.busy || demo.storage.removing.contains(model.id))
                } else if model.id == demo.model && demo.loading {
                    VStack(alignment: .trailing, spacing: 4) {
                        Text(demo.status).font(.caption).foregroundStyle(.secondary)
                        ProgressView(value: demo.progress).frame(width: 120)
                    }
                } else if downloading {
                    VStack(alignment: .trailing, spacing: 4) {
                        Text("Downloading…").font(.caption).foregroundStyle(.secondary)
                        ProgressView(value: demo.downloads.progress[model.id]).frame(width: 120)
                    }
                } else if model.id == demo.model {
                    Label("Selected", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(demo.ready ? .green : .secondary)
                        .help("Selected for \(model.kind?.title ?? model.task)")
                }
                if issue == nil, !downloading, demo.catalog.update(for: model) != nil {
                    Button("Update") { Task { await demo.update(model.id) } }
                        .disabled(demo.busy || demo.updating.contains(model.id) || demo.storage.removing.contains(model.id))
                } else if !outdated.isEmpty && files?.bundles.contains(where: \.current) == true {
                    Button("Remove Outdated") { Task { await demo.remove(model.id, outdated: true) } }
                        .disabled(!demo.storage.removing.isEmpty)
                        .help("Delete \(formatBytes(outdated.map { $0.bytes + $0.compiled.bytes }.reduce(0, +))) "
                              + "of revisions this version no longer loads")
                }
                if demo.storage.removing.contains(model.id) {
                    ProgressView().controlSize(.small)
                } else if stored && !demo.keeps(model.id) {
                    Button(role: .destructive) { pending = model } label: { Image(systemName: "trash") }
                        .buttonStyle(.borderless)
                        .help(downloading ? "Stop the download and delete its files" : "Delete model files and cached copies")
                        .disabled(!demo.storage.removing.isEmpty)
                }
                if model.id != demo.model && issue == nil {
                    Button(downloaded ? "Select" : "Download & Select") { Task { await demo.choose(model) } }
                    .disabled(demo.busy || demo.storage.removing.contains(model.id))
                }
            }
        }
    }
}

