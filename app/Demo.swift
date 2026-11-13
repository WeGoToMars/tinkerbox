import AppKit
import SwiftUI

// The catalog and the one model the worker has loaded, shared by every task page.
@MainActor @Observable final class Demo {
    var catalog = Catalog()
    let storage = Storage()
    let downloads = Downloads()
    var model = Runtime.selectedModel(for: .decision)
    // The selected model's worker description, once it has loaded.
    var loaded: ReadyReply? { runtime.state.reply?.modelId == model ? runtime.state.reply : nil }
    var ready: Bool { loaded != nil }
    // The selected model's download, then the runtime loading it.
    var loading: Bool { downloads.contains(model) || (runtime.state.loading && runtime.state.model == model) }
    var status: String { downloads.contains(model) ? "Downloading model…" : runtime.state.status }
    var progress: Double? { downloads.contains(model) ? downloads.progress[model] : runtime.state.progress }
    var busy = false  // a request from the app is running
    private(set) var updating: Set<String> = []
    var unloading = false
    var error: String?
    var settings: [ModelSetting] { loaded?.settings ?? [] }
    var speechCapabilities: SpeechCapabilities? { Catalog.speechCapabilities[model] }
    // Only values that differ from the defaults are kept.
    var settingValues: [String: Int] = [:] {
        didSet { Runtime.store(settings: settingValues, for: model) }
    }
    let runtime = Runtime.shared
    private var generation = 0

    var current: CatalogModel? { catalog.models.first { $0.id == model } }
    var compatibilityIssue: String? {
        current.flatMap { Catalog.availabilityIssue(for: $0, asset: Catalog.overrideAsset) }
    }

    func isDownloaded(_ id: String) -> Bool { catalog.downloaded.contains(id) && !downloads.contains(id) }

    // The models a task can switch to without a download; the Models library lists the rest.
    func models(for task: ModelTask) -> [CatalogModel] {
        catalog.models.filter { model in
            model.kind == task && model.isSupported && (isDownloaded(model.id) || model.id == self.model)
        }
    }

    // Downloads run beside the runtime, so they never hold up requests or model switches.
    // Local bundles, chosen in the app or given at launch, need none.
    private func fetch(_ id: String) async throws {
        guard !isDownloaded(id), Catalog.localAsset(for: id) == nil, Catalog.overrideAsset == nil else { return }
        defer {
            catalog.scanDownloads()
            Task { await storage.refresh() }
        }
        try await downloads.fetch(id)
    }

    // Downloads a model's newer revision beside the installed one, switches to it and removes the outdated files.
    func update(_ id: String) async {
        guard !updating.contains(id), !storage.removing.contains(id) else { return }
        updating.insert(id)
        defer { updating.remove(id) }
        do { try await downloads.fetch(id, entry: catalog.offers[id]?.raw) } catch {
            if !(error is CancellationError) { self.error = error.localizedDescription }
            return
        }
        catalog.adopt(id)
        // A worker running the old revision restarts on the new one before its files are removed.
        if runtime.state.model == id {
            await runtime.unload()
            await load()
        }
        await remove(id, outdated: true)
    }

    func load() async {
        // A model switch supersedes an earlier load; ignore that load's outcome.
        generation += 1
        let current = generation
        error = nil
        settingValues = Runtime.settings(for: model)
        defer {
            if current == generation {
                catalog.scanDownloads()
                Task { await storage.refresh() }  // the load may have downloaded files
            }
        }
        do {
            try await fetch(model)
            guard current == generation else { return }
            _ = try await runtime.start(model: model)
        } catch is CancellationError {
            // Its download was stopped to remove the model.
        } catch {
            if current == generation { self.error = error.localizedDescription }
        }
    }

    // The previous model keeps serving Shortcuts until the chosen one has downloaded.
    func choose(_ choice: CatalogModel, asset: URL? = nil) async {
        guard let task = choice.kind, choice.isSupported, !storage.removing.contains(choice.id),
            choice.id != model || !ready || asset != nil
        else { return }
        generation += 1
        let current = generation
        model = choice.id  // reserve its files before yielding to the runtime
        error = nil
        settingValues = Runtime.settings(for: model)
        do { if asset == nil { try await fetch(choice.id) } } catch {
            if current == generation && !(error is CancellationError) { self.error = error.localizedDescription }
            return
        }
        guard current == generation else { return }
        await runtime.select(model: choice.id, for: task, asset: asset?.path)
        guard current == generation else { return }
        await load()
    }

    func unload() async {
        busy = true
        unloading = true
        defer { busy = false; unloading = false }
        await runtime.unload()
    }

    // The model the runtime has loaded or is loading, and the selected one once downloaded, keep their files.
    func keeps(_ id: String) -> Bool { id == runtime.state.model || (id == model && isDownloaded(id)) }

    // Removing a model stops its download first. Outdated revisions are never loaded, so any model can lose them.
    func remove(_ id: String, outdated: Bool = false) async {
        if !outdated {
            guard !keeps(id) else { return }
            await downloads.stop(id)
            if Catalog.localAsset(for: id) != nil {
                Catalog.setLocalAsset(nil, for: id)
                catalog.scanDownloads()
                return
            }
        }
        await storage.remove(id, outdated: outdated)
        catalog.scanDownloads()
    }
}

func duration(_ milliseconds: Double?) -> String {
    guard let milliseconds else { return "—" }
    return milliseconds < 1000
        ? String(format: "%.1f ms", milliseconds)
        : String(format: "%.2f s", milliseconds / 1000)
}

enum Destination: Hashable {
    case home, task(ModelTask), models, storage
}

struct AppView: View {
    let demo: Demo
    @State private var decisions: DecisionPlayground
    @State private var completions: CompletionPlayground
    @State private var speech: SpeechPlayground
    @State private var destination: Destination? = .home

    init(demo: Demo) {
        self.demo = demo
        _decisions = State(initialValue: DecisionPlayground(demo: demo))
        _completions = State(initialValue: CompletionPlayground(demo: demo))
        _speech = State(initialValue: SpeechPlayground(demo: demo))
    }

    var body: some View {
        NavigationSplitView {
            List(selection: $destination) {
                Label("Home", systemImage: "house").tag(Destination.home)
                Section("Tasks") {
                    ForEach(ModelTask.allCases, id: \.self) { task in
                        Label(task.title, systemImage: task.symbol).tag(Destination.task(task))
                    }
                }
                Section("Library") {
                    Label("Models", systemImage: "square.stack.3d.up")
                        .symbolEffect(.bounce, value: demo.downloads.completed)
                        .tag(Destination.models)
                    Label("Storage", systemImage: "internaldrive").tag(Destination.storage)
                }
            }
            .navigationSplitViewColumnWidth(min: 160, ideal: 180)
            .disabled(demo.busy)
        } detail: {
            Group {
                switch destination ?? .home {
                case .home: HomeView { destination = .task($0) }
                case .task(.decision): DecisionView(playground: decisions) { destination = .models }
                case .task(.speech): SpeechView(playground: speech) { destination = .models }
                case .task(.completion): CompletionView(playground: completions) { destination = .models }
                case .models: ModelsView(demo: demo)
                case .storage: StorageView(demo: demo) { destination = .models }
                }
            }
            .safeAreaInset(edge: .bottom, spacing: 0) { SystemFooter() }
        }
        .task(id: destination) {
            if case .task(let task) = destination {
                let selected = Runtime.selectedModel(for: task)
                if demo.model == selected {
                    if !demo.ready { await demo.load() }
                } else if let model = demo.catalog.models.first(where: { $0.id == selected }) {
                    await demo.choose(model)
                }
            }
        }
        .task { await demo.catalog.refresh() }
        .task { await Updates.shared.check() }
        .onChange(of: demo.model) {
            decisions.modelChanged()
            completions.modelChanged()
        }
        .alert(
            "Could not update storage",
            isPresented: Binding(
                get: { demo.storage.error != nil },
                set: { if !$0 { demo.storage.error = nil } })
        ) {
        } message: {
            Text(demo.storage.error ?? "")
        }
    }
}

// The open task page's main action, for the Model menu.
struct TaskAction {
    let title: String
    let enabled: Bool
    let perform: () -> Void
}

extension FocusedValues {
    @Entry var taskAction: TaskAction?
}

struct ModelCommands: Commands {
    let demo: Demo
    @FocusedValue(\.taskAction) private var action

    var body: some Commands {
        CommandMenu("Model") {
            Button(action?.title ?? "Run") { action?.perform() }
                .keyboardShortcut(.return)
                .disabled(action?.enabled != true)
            Button(demo.ready ? "Unload Model" : "Load Model") {
                Task { if demo.ready { await demo.unload() } else { await demo.load() } }
            }
            .keyboardShortcut("l")
            .disabled(demo.busy || demo.loading || demo.compatibilityIssue != nil)
        }
    }
}

// The loaded model's status, and a menu to switch between this task's downloaded models.
struct ModelToolbar: ToolbarContent {
    let demo: Demo
    let task: ModelTask
    @Binding var showingInspector: Bool
    let showModels: () -> Void

    var body: some ToolbarContent {
        ToolbarItem {
            if demo.unloading {
                Text("Unloading model…").foregroundStyle(.secondary)
            } else if demo.loading {
                HStack(spacing: 8) {
                    if let progress = demo.progress {
                        ProgressView(value: progress).frame(width: 80)
                        Text("\(demo.status) \(Int(progress * 100))%").monospacedDigit()
                            .contentTransition(.numericText())
                            .animation(.default, value: Int(progress * 100))
                    } else {
                        ProgressView().controlSize(.small)
                        Text(demo.status)
                    }
                }
                .foregroundStyle(.secondary)
                .fixedSize()
            } else if demo.ready {
                Label("Model ready", systemImage: "checkmark.circle.fill")
                    .labelStyle(.titleAndIcon)
                    .foregroundStyle(.green)
                    .fixedSize()
                    .help("Unloads after 5 minutes without inference.")
            } else if let issue = demo.compatibilityIssue {
                Label("Model unavailable", systemImage: "exclamationmark.circle")
                    .foregroundStyle(.secondary)
                    .help(issue)
            } else {
                Label("Model unloaded", systemImage: "moon")
                    .foregroundStyle(.secondary)
                    .help("Loads automatically on the next run. Unloads after 5 minutes without inference.")
            }
        }
        // Status text, not a control, so it sits outside the glass capsule.
        .sharedBackgroundVisibility(.hidden)
        ToolbarItem {
            Menu {
                Picker(
                    "Model for \(task.title)",
                    selection: Binding(
                        get: { demo.model },
                        set: { id in
                            guard let model = demo.catalog.models.first(where: { $0.id == id }) else { return }
                            Task { await demo.choose(model) }
                        })
                ) {
                    ForEach(demo.models(for: task)) { model in
                        Text([model.name, model.fileSize].compactMap { $0 }.joined(separator: " · ")).tag(model.id)
                    }
                }
                .pickerStyle(.inline)
                Divider()
                Button("Manage Models…", action: showModels)
            } label: {
                Label(demo.current.map { "Model: \($0.name)" } ?? "Select Model", systemImage: "square.stack.3d.up")
                    .labelStyle(.titleAndIcon)
                    .symbolEffect(.pulse, isActive: demo.loading)
            }
            .disabled(demo.busy)
            .help("Select a model for this task, or open Manage Models to download one.")
        }
        ToolbarItem {
            Button("Unload", systemImage: "eject") { Task { await demo.unload() } }
                .labelStyle(.titleAndIcon)
                .disabled(!demo.ready || demo.loading || demo.busy)
                .help("Unload the model from memory. The next run loads it again.")
        }
        ToolbarItem {
            Toggle("Model Settings", systemImage: "gearshape", isOn: $showingInspector)
        }
    }
}

// The loaded model's settings and fixed properties, beside a task page.
struct ModelInspector: View {
    let demo: Demo

    var body: some View {
        Form {
            Section("Settings") {
                if demo.settings.isEmpty {
                    Text(
                        demo.loading
                            ? "Loading settings…"
                            : demo.ready ? "No model settings." : "Load the model to view settings."
                    )
                    .foregroundStyle(.secondary)
                }
                ForEach(demo.settings) { setting in
                    let value = Binding {
                        demo.settingValues[setting.key] ?? setting.defaultValue
                    } set: {
                        demo.settingValues[setting.key] = $0 == setting.defaultValue ? nil : $0
                    }
                    Stepper(value: value, in: setting.min...setting.max, step: setting.step ?? 1) {
                        LabeledContent(setting.title, value: "\(value.wrappedValue)")
                    }
                    .help(setting.help)
                }
                if !demo.settingValues.isEmpty {
                    Button("Reset to Defaults") { demo.settingValues = [:] }
                }
            }
            if let info = demo.loaded?.info, !info.isEmpty {
                Section("Model") {
                    ForEach(info, id: \.label) { LabeledContent($0.label, value: $0.value) }
                }
            }
        }
        .formStyle(.grouped)
        .disabled(demo.busy)
        .inspectorColumnWidth(min: 240, ideal: 280, max: 360)
    }
}

struct ModelAvailability: View {
    let demo: Demo
    let showModels: () -> Void

    var body: some View {
        if let issue = demo.compatibilityIssue {
            HStack(alignment: .top, spacing: 12) {
                Label(issue, systemImage: "info.circle")
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Manage Models…", action: showModels)
            }
        }
    }
}

private struct TokenBackground: Hashable, Sendable {
    let color: Color?
}

private enum TokenBackgroundKey: AttributedStringKey {
    typealias Value = TokenBackground
    static let name = "tinkerbox.tokenBackground"
}

extension AttributedString {
    mutating func highlightTokens(_ spans: [TokenSpan] = [], selection: inout AttributedTextSelection) {
        transform(updating: &selection) { text in
            for run in text.runs {
                if let saved = run[TokenBackgroundKey.self] {
                    text[run.range].backgroundColor = saved.color
                    text[run.range][TokenBackgroundKey.self] = nil
                }
            }
            let plain = String(text.characters)
            let colors: [Color] = [.teal, .orange, .purple, .pink]
            for (index, span) in spans.enumerated() {
                guard
                    let range = Range(
                        NSRange(location: span.startUtf16, length: span.endUtf16 - span.startUtf16), in: plain),
                    let start = AttributedString.Index(range.lowerBound, within: text),
                    let end = AttributedString.Index(range.upperBound, within: text)
                else { continue }
                for run in text[start..<end].runs {
                    text[run.range][TokenBackgroundKey.self] = TokenBackground(color: run.backgroundColor)
                    text[run.range].backgroundColor = colors[index % colors.count].opacity(0.2)
                }
            }
        }
        selection.resetTokenTyping(in: self)
    }
}

extension AttributedTextSelection {
    mutating func resetTokenTyping(in text: AttributedString) {
        guard case .insertionPoint(let index) = indices(in: text) else { return }
        var attributes = typingAttributes(in: text)
        guard let saved = attributes[TokenBackgroundKey.self] else { return }
        attributes.backgroundColor = saved.color ?? .clear
        attributes[TokenBackgroundKey.self] = nil
        self = AttributedTextSelection(insertionPoint: index, typingAttributes: attributes)
    }
}

struct TaskTextEditor: View {
    @Binding var text: AttributedString
    @Binding var selection: AttributedTextSelection
    let demo: Demo
    var focus: FocusState<Bool>.Binding?
    @FocusState private var ownFocus: Bool
    @AppStorage("editor.highlightTokens") private var highlightTokens = false
    @State private var tokenError: String?

    private struct PreviewInput: Hashable {
        let text: String
        let model: String
        let ready: Bool
        let busy: Bool
        let enabled: Bool
    }

    private var previewInput: PreviewInput {
        PreviewInput(
            text: String(text.characters), model: demo.model,
            ready: demo.ready, busy: demo.busy, enabled: highlightTokens)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle("Highlight tokens", systemImage: "highlighter", isOn: $highlightTokens)
                .toggleStyle(.button).disabled(demo.busy)
                .help(
                    demo.ready
                        ? "Color the text by token with the model's tokenizer."
                        : "Colors the text by token once the model is loaded.")
            TextEditor(text: $text, selection: $selection)
                .focused(focus ?? $ownFocus)
                .font(.body)
                .scrollContentBackground(.hidden)
                .padding(10)
                .background(.background, in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                .overlay { if demo.busy { EditorFollower() } }
                .frame(height: 260)
                .onChange(of: selection) { selection.resetTokenTyping(in: text) }
            if let tokenError {
                Text(tokenError).font(.caption).foregroundStyle(.red)
            }
        }
        .task(id: previewInput) { await refreshTokens() }
    }

    private func refreshTokens() async {
        tokenError = nil
        text.highlightTokens(selection: &selection)
        let input = previewInput
        guard input.enabled, input.ready, !input.busy else { return }
        do {
            try await Task.sleep(for: .milliseconds(250))  // wait for typing to pause
            let spans = try await demo.runtime.tokenSpans(input.text, model: input.model)
            guard input == previewInput, !Task.isCancelled else { return }
            text.highlightTokens(spans, selection: &selection)
        } catch is CancellationError {} catch {
            if input == previewInput, !Task.isCancelled { tokenError = error.localizedDescription }
        }
    }
}

/// Laid over a text editor that is disabled while text streams into it: keeps it scrolled to the end
/// as the text grows until the user scrolls away, and passes the user's scrolling through. SwiftUI's
/// TextEditor has no scroll API, and a disabled one ignores scrolling.
private struct EditorFollower: NSViewRepresentable {
    func makeNSView(context: Context) -> Follower { Follower() }
    func updateNSView(_ view: Follower, context: Context) {}

    final class Follower: NSView {
        private weak var scrollView: NSScrollView?
        private var following = true
        private var observers: [NSObjectProtocol] = []

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            observers.forEach(NotificationCenter.default.removeObserver)
            observers = []
            DispatchQueue.main.async { [weak self] in self?.attach() }  // after the editor's layout
        }

        override func scrollWheel(with event: NSEvent) {
            scrollView?.scrollWheel(with: event)
        }

        private func attach() {
            guard let root = window?.contentView,
                let textView = Self.textView(in: root, within: convert(bounds, to: nil)),
                let scrollView = textView.enclosingScrollView
            else { return }
            self.scrollView = scrollView
            let clip = scrollView.contentView
            clip.postsBoundsChangedNotifications = true
            textView.postsFrameChangedNotifications = true
            let center = NotificationCenter.default
            observers = [
                center.addObserver(forName: NSView.boundsDidChangeNotification, object: clip, queue: .main) {
                    [weak self] _ in
                    MainActor.assumeIsolated { self?.following = clip.bounds.maxY >= textView.frame.height - 4 }
                },
                center.addObserver(forName: NSView.frameDidChangeNotification, object: textView, queue: .main) {
                    [weak self] _ in
                    MainActor.assumeIsolated { if self?.following == true { textView.scrollToEndOfDocument(nil) } }
                },
            ]
        }

        private static func textView(in node: NSView, within frame: CGRect) -> NSTextView? {
            if let text = node as? NSTextView, let scroll = text.enclosingScrollView,
                frame.contains(scroll.convert(scroll.bounds, to: nil))
            {
                return text
            }
            return node.subviews.lazy.compactMap { textView(in: $0, within: frame) }.first
        }
    }
}

// Settings sit beside the input on wide panes and below it otherwise. One AnyLayout keeps the same
// column views across the switch, so editors, focus and playback retain their state.
struct TaskPage<Primary: View, Settings: View>: View {
    @ViewBuilder var primary: Primary
    @ViewBuilder var settings: Settings
    @State private var wide = false

    var body: some View {
        let layout =
            wide
            ? AnyLayout(HStackLayout(alignment: .top, spacing: 24))
            : AnyLayout(VStackLayout(alignment: .leading, spacing: 24))
        ScrollView {
            layout {
                VStack(alignment: .leading, spacing: 22) { primary }
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                VStack(alignment: .leading, spacing: 22) { settings }
                    .frame(width: wide ? 380 : nil, alignment: .topLeading)
                    .frame(maxWidth: wide ? nil : .infinity, alignment: .topLeading)
            }
            .padding(24)
        }
        .onGeometryChange(for: Bool.self) {
            $0.size.width >= 1100
        } action: {
            wide = $0
        }
        .frame(minWidth: 650, minHeight: 650)
        .background(.background.secondary)
    }
}

struct TaskSection<Content: View>: View {
    var title: String?
    @ViewBuilder var content: Content

    init(_ title: String? = nil, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) { content }
                .frame(maxWidth: .infinity, alignment: .leading)
                .textFieldStyle(.roundedBorder)
                .padding(8)
        } label: {
            if let title { Text(title).font(.headline) }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// The newest published release, checked once per launch; nil until known, or offline.
@MainActor @Observable final class Updates {
    static let shared = Updates()
    static let repository = URL(string: "https://github.com/WeGoToMars/tinkerbox")!
    static let releases = repository.appending(path: "releases/latest")
    private(set) var latest: String?

    var available: String? { latest.flatMap { Catalog.isNewer($0, than: Catalog.appVersion) ? $0 : nil } }

    func check() async {
        guard latest == nil else { return }
        // Drafts and prereleases are not "latest".
        var request = URLRequest(url: URL(string: "https://api.github.com/repos/WeGoToMars/tinkerbox/releases/latest")!)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        struct Release: Decodable { let tagName: String }
        guard let (data, response) = try? await URLSession.shared.data(for: request),
            (response as? HTTPURLResponse)?.statusCode == 200,
            let release = try? Runtime.decoder.decode(Release.self, from: data)
        else { return }
        latest = release.tagName.hasPrefix("v") ? String(release.tagName.dropFirst()) : release.tagName
    }
}

struct SystemFooter: View {
    @State private var thermalState = ProcessInfo.processInfo.thermalState
    @State private var lowPowerMode = ProcessInfo.processInfo.isLowPowerModeEnabled
    @Environment(\.openURL) private var openURL

    // GitHub's mark (Primer Octicons, MIT), drawn in the text color.
    private static let githubMark: NSImage = {
        let svg = """
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 \
            1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 \
            0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 \
            .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 \
            3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 \
            1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 \
            8-8Z"/></svg>
            """
        let image = NSImage(data: Data(svg.utf8)) ?? NSImage()
        image.isTemplate = true
        return image
    }()

    private var thermalDescription: String {
        switch thermalState {
        case .nominal: "Nominal"
        case .fair: "Fair"
        case .serious: "Serious"
        case .critical: "Critical"
        @unknown default: "Unknown"
        }
    }

    var body: some View {
        let system = ProcessInfo.processInfo
        let version = system.operatingSystemVersion
        let memory = ByteCountFormatter.string(fromByteCount: Int64(system.physicalMemory), countStyle: .memory)
        VStack(spacing: 0) {
            Divider()
            HStack(spacing: 8) {
                Text(
                    "\(memory) memory · macOS \(version.majorVersion).\(version.minorVersion).\(version.patchVersion)"
                        + " · Thermal: \(thermalDescription) · Low Power Mode: \(lowPowerMode ? "On" : "Off")")
                Spacer()
                if let newer = Updates.shared.available {
                    Button("Update to \(newer)") { openURL(Updates.releases) }
                        .buttonStyle(.plain)
                        .bold()
                        .foregroundStyle(.primary)
                        .help("Download the new version from GitHub")
                }
                Text("Tinkerbox \(Catalog.appVersion)")
                Button {
                    openURL(Updates.repository)
                } label: {
                    Image(nsImage: Self.githubMark).resizable().frame(width: 12, height: 12)
                }
                .buttonStyle(.plain)
                .help("Tinkerbox on GitHub")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(.horizontal, 28)
            .padding(.vertical, 12)
        }
        .background(.background)
        .onReceive(NotificationCenter.default.publisher(for: ProcessInfo.thermalStateDidChangeNotification)) { _ in
            thermalState = system.thermalState
        }
        .onReceive(NotificationCenter.default.publisher(for: .NSProcessInfoPowerStateDidChange)) { _ in
            lowPowerMode = system.isLowPowerModeEnabled
        }
    }
}

@MainActor final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    func applicationWillTerminate(_ notification: Notification) {
        Runtime.stop(Runtime.workerProcess)
        Downloads.processes.values.forEach { Runtime.stop($0) }
    }
}

// The model the worker holds, whichever page or Shortcut loaded it, with a way to free its memory.
struct RuntimeMenu: View {
    let demo: Demo
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        let state = demo.runtime.state
        VStack(alignment: .leading, spacing: 12) {
            if let reply = state.reply {
                VStack(alignment: .leading, spacing: 4) {
                    Text(reply.modelName).font(.headline)
                    Text(
                        [Catalog.current(reply.modelId)?.kind?.title, "Unloads after 5 minutes without inference"]
                            .compactMap { $0 }.joined(separator: " · ")
                    )
                    .font(.caption).foregroundStyle(.secondary)
                }
            } else if state.loading {
                VStack(alignment: .leading, spacing: 6) {
                    Text(state.status)
                    ProgressView(value: state.progress)
                }
            } else {
                Text("No model loaded").foregroundStyle(.secondary)
            }
            HStack {
                Button(demo.unloading ? "Unloading…" : "Unload", systemImage: "eject") { Task { await demo.unload() } }
                    .disabled(state.reply == nil || demo.busy)
                Spacer()
                Button("Open Tinkerbox") {
                    openWindow(id: "demo")
                    NSApp.activate()
                }
            }
            Divider()
            Button("Quit Tinkerbox") { NSApp.terminate(nil) }
                .buttonStyle(.plain)
                .keyboardShortcut("q")
        }
        .padding(14)
        .frame(width: 280)
    }
}

@main struct TinkerboxDemoApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @State private var demo = Demo()

    var body: some Scene {
        Window("Tinkerbox", id: "demo") {
            AppView(demo: demo)
        }
        .defaultSize(width: 940, height: 820)
        .commands { ModelCommands(demo: demo) }
        MenuBarExtra {
            RuntimeMenu(demo: demo)
        } label: {
            Label {
                Text("Tinkerbox")
            } icon: {
                Image(nsImage: LogoGeometry.menuBarIcon)
            }
        }
        .menuBarExtraStyle(.window)
    }
}
