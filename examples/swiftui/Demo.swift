import AppKit
import SwiftUI

@MainActor @Observable final class Demo {
    var state = "I was charged twice this month. Please refund the duplicate payment."
    var question = "Is this a refund request?"
    var type = QuestionType.noul
    var choices = "Refund request\nTechnical support\nOther"
    var levels = "Not urgent\nSomewhat urgent\nVery urgent"
    var ready = false
    var loading = true
    var status = "Preparing runtime…"
    var busy = false
    var error: String?
    var result: Reply?
    var setupMs: Double?
    var modelName: String?
    var cpuModel: String?
    var architecture: String?
    var times: [Double] = []
    let runtime = Runtime.shared

    var criteria: [String] {
        guard type != .noul else { return [] }
        return (type == .choice ? choices : levels).components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }

    var validation: String? {
        if type == .choice && (criteria.count < 2 || Set(criteria).count != criteria.count) {
            return "Add at least two unique options."
        }
        if type == .score && !(2...10).contains(criteria.count) {
            return "Add 2–10 levels, ordered from lowest to highest."
        }
        return nil
    }

    func load() async {
        loading = true
        error = nil
        status = "Preparing runtime…"
        modelName = nil
        cpuModel = nil
        architecture = nil
        defer { loading = false }
        do {
            let reply = try await runtime.start { [weak self] message in self?.status = message }
            setupMs = reply.setupMs
            modelName = reply.modelName
            cpuModel = reply.cpuModel
            architecture = reply.architecture
            ready = reply.ready == true
        } catch {
            self.error = error.localizedDescription
        }
    }

    func run() async {
        busy = true
        error = nil
        defer { busy = false }
        do {
            let reply = try await runtime.decide(Request(state: state, type: type,
                                                         instructions: question, criteria: criteria))
            result = reply
            if let elapsed = reply.elapsedMs { times.append(elapsed) }
        } catch {
            self.error = error.localizedDescription
            if await !runtime.isConnected { ready = false }
        }
    }

    func percentile(_ fraction: Double) -> Double? {
        guard !times.isEmpty else { return nil }
        let sorted = times.sorted()
        let index = Double(sorted.count - 1) * fraction
        let lower = Int(index)
        let upper = min(lower + 1, sorted.count - 1)
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - Double(lower))
    }
}

func duration(_ milliseconds: Double?) -> String {
    guard let milliseconds else { return "—" }
    return milliseconds < 1000
        ? String(format: "%.1f ms", milliseconds)
        : String(format: "%.2f s", milliseconds / 1000)
}

struct ContentView: View {
    @State private var demo = Demo()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                HStack(spacing: 12) {
                    Image(systemName: "cpu").font(.largeTitle).foregroundStyle(.tint)
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Tinkerbox").font(.title2.bold())
                        Text([demo.modelName, demo.cpuModel, demo.architecture].compactMap { $0 }.joined(separator: " · "))
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    if demo.loading {
                        ProgressView().controlSize(.small)
                        Text(demo.status).foregroundStyle(.secondary)
                    } else if demo.ready {
                        Label("Model ready", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                    }
                }

                if !demo.ready {
                    HStack {
                        Text("First launch downloads Python, packages and model files.")
                            .font(.caption).foregroundStyle(.secondary)
                        Spacer()
                        Button("Setup log") { NSWorkspace.shared.open(Runtime.logURL) }
                        if !demo.loading {
                            Button("Retry") { Task { await demo.load() } }
                        }
                    }
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("Input text").font(.headline)
                    TextEditor(text: $demo.state)
                        .font(.body)
                        .scrollContentBackground(.hidden)
                        .padding(10)
                        .background(.background, in: RoundedRectangle(cornerRadius: 8))
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                        .frame(height: 110)
                        .accessibilityLabel("Input text")
                    Picker("Question type", selection: $demo.type) {
                        ForEach(QuestionType.allCases, id: \.self) { type in
                            Text(type.rawValue).tag(type)
                        }
                    }
                    .pickerStyle(.segmented)
                    .padding(.vertical, 6)
                    .onChange(of: demo.type) { _, type in
                        demo.question = type.example
                        demo.result = nil
                        demo.error = nil
                    }
                    Text("Question / instructions").font(.headline)
                    TextField(demo.type.example, text: $demo.question)
                        .textFieldStyle(.roundedBorder)
                        .accessibilityLabel("Question instructions")
                    if demo.type != .noul {
                        Text(demo.type == .choice ? "Options · one per line" : "Score levels · lowest to highest")
                            .font(.headline).padding(.top, 6)
                        TextEditor(text: demo.type == .choice ? $demo.choices : $demo.levels)
                            .font(.body)
                            .scrollContentBackground(.hidden)
                            .padding(10)
                            .background(.background, in: RoundedRectangle(cornerRadius: 8))
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                            .frame(height: 90)
                            .accessibilityLabel(demo.type == .choice ? "Choice options" : "Score levels")
                        Text(demo.validation ?? (demo.type == .choice
                             ? "Each line is a named choice." : "Levels are numbered from 0; the answer is the expected score."))
                            .font(.caption).foregroundStyle(demo.validation == nil ? Color.secondary : .red)
                    }
                }
                .disabled(demo.busy)

                HStack {
                    Button {
                        Task { await demo.run() }
                    } label: {
                        Label(demo.busy ? "Running…" : "Run decision", systemImage: "play.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(!demo.ready || demo.busy || demo.validation != nil
                              || demo.state.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                              || demo.question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    if demo.busy { ProgressView().controlSize(.small) }
                    Spacer()
                    if let setup = demo.setupMs {
                        Text("Model setup \(duration(setup))").font(.caption).foregroundStyle(.secondary)
                    }
                }

                if let error = demo.error {
                    Label(error, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.red).textSelection(.enabled)
                }

                GroupBox {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text(demo.result?.answer?.type == .score ? "Expected score" : "Decision").font(.headline)
                            Spacer()
                            if let score = demo.result?.answer?.score {
                                Text(String(format: "%.2f / %d", score, (demo.result?.probabilities?.count ?? 1) - 1))
                                    .font(.title2.bold())
                            } else if let choice = demo.result?.answer?.choice {
                                Text(choice).font(.title2.bold()).textSelection(.enabled)
                            } else if let probability = demo.result?.answer?.noul {
                                Text(probability >= 0.5 ? "Yes" : "No").font(.title2.bold())
                            } else {
                                Text("Ready for your first question").foregroundStyle(.secondary)
                            }
                        }
                        if let probabilities = demo.result?.probabilities {
                            ForEach(probabilities.indices, id: \.self) { index in
                                probabilityRow(probabilities[index])
                            }
                        }
                        if let tokens = demo.result?.tokens {
                            Text("\(tokens) rendered input tokens").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .padding(10)
                }

                HStack(spacing: 40) {
                    metric("Last decision", value: demo.result?.elapsedMs)
                    metric("Median", value: demo.percentile(0.5))
                    metric("p95", value: demo.percentile(0.95))
                    Spacer()
                    VStack(alignment: .trailing, spacing: 5) {
                        Text("Decisions").font(.caption).foregroundStyle(.secondary)
                        Text("\(demo.times.count)").font(.title3.monospacedDigit())
                    }
                }
            }
            .padding(28)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(minWidth: 650, idealWidth: 760, minHeight: 650)
        .background(.background.secondary)
        .safeAreaInset(edge: .bottom, spacing: 0) { SystemFooter() }
        .task { await demo.load() }
    }

    private func probabilityRow(_ probability: Probability) -> some View {
        HStack(spacing: 12) {
            Text(probability.label).frame(width: 180, alignment: .leading)
            ProgressView(value: probability.value)
            Text(String(format: "%.1f%%", probability.value * 100))
                .monospacedDigit().frame(width: 65, alignment: .trailing)
        }
    }

    private func metric(_ title: String, value: Double?) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(duration(value)).font(.title3.monospacedDigit())
        }
    }
}

struct SystemFooter: View {
    @State private var thermalState = ProcessInfo.processInfo.thermalState
    @State private var lowPowerMode = ProcessInfo.processInfo.isLowPowerModeEnabled

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
            Text("\(memory) memory · macOS \(version.majorVersion).\(version.minorVersion).\(version.patchVersion)"
                 + " · Thermal: \(thermalDescription) · Low Power Mode: \(lowPowerMode ? "On" : "Off")")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
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
        if let process = Runtime.setupProcess, process.isRunning { process.terminate() }
    }
}

@main struct TinkerboxDemoApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        Window("Tinkerbox", id: "demo") {
            ContentView()
        }
        .defaultSize(width: 760, height: 820)
    }
}
