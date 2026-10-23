import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct SpeechView: View {
    let demo: Demo
    let showModels: () -> Void
    @State private var text = "Hello! Welcome to Tinkerbox."
    @State private var mode = "Auto"
    @State private var language = ""
    @State private var instruction = ""
    @State private var reference: URL?
    @State private var prompt: URL?
    @State private var transcript = ""
    @State private var speed = 1.0
    @State private var fixedDuration = false
    @State private var duration = 5.0
    @State private var seed = ""
    @State private var values: [String: SpeechValue] = [:]
    @State private var result: SpeechAudio?
    @State private var elapsed: Double?
    @State private var aneMs: Double?
    @State private var resultSeed: Int64?
    @State private var error: String?
    @State private var showingInspector = false

    var body: some View {
        Form {
            Section("Text") {
                TextEditor(text: $text).frame(minHeight: 120)
                TextField("Language name or code (automatic if empty)", text: $language)
                Text("Pronunciation and non-verbal tags can be included directly in the text.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Voice") {
                Picker("Mode", selection: $mode) {
                    ForEach(["Auto", "Design", "Clone"], id: \.self) { Text($0) }
                }.pickerStyle(.segmented)
                if mode == "Design" {
                    TextField("Voice description", text: $instruction, prompt: Text("female, low pitch, british accent"))
                }
                if mode == "Clone" {
                    HStack {
                        Button("Reference audio…") {
                            if let url = pick(types: [.audio]) { reference = url; prompt = nil; transcript = "" }
                        }
                        Text(reference?.lastPathComponent ?? "Choose a short voice recording").foregroundStyle(.secondary)
                    }
                    if prompt == nil {
                        TextField("Reference transcript (automatic transcription if empty)", text: $transcript, axis: .vertical)
                    }
                    HStack {
                        Button("Load saved voice…") {
                            if let url = pick(types: [.data]) { prompt = url; reference = nil; transcript = "" }
                        }
                        Text(prompt?.lastPathComponent ?? "Reusable voice prompt").foregroundStyle(.secondary)
                        if reference != nil {
                            Button("Save voice…") { Task { await saveVoice() } }
                                .disabled(!demo.ready)
                        }
                    }
                }
            }
            Section("Duration and sampling") {
                Toggle("Fixed duration", isOn: $fixedDuration)
                if fixedDuration {
                    TextField("Seconds", value: $duration, format: .number)
                } else {
                    HStack {
                        Text("Speed")
                        Slider(value: $speed, in: 0.5...2.0, step: 0.05)
                        Text(speed.formatted(.number.precision(.fractionLength(2))) + "×").monospacedDigit()
                    }
                }
                TextField("Random seed (optional)", text: $seed)
                DisclosureGroup("Advanced generation settings") {
                    ForEach(demo.speechParameters) { parameter in control(parameter) }
                }
            }
            Section {
                HStack {
                    Button(demo.busy ? "Generating…" : "Generate speech", systemImage: "waveform") {
                        Task { await generate() }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(demo.busy || !demo.ready || demo.current?.kind != .speech || isBlank(text))
                    if demo.busy { ProgressView().controlSize(.small) }
                    if let result {
                        Button("Save WAV…", systemImage: "square.and.arrow.down") { save(result.wav) }
                    }
                }
                if let result {
                    AudioPlayback(data: result.wav)
                        .padding(.vertical, 8)
                    Text("\(result.duration.formatted(.number.precision(.fractionLength(2)))) seconds · \(result.sampleRate) Hz")
                    if let elapsed, let aneMs {
                        Text("Generated in \((elapsed / 1000).formatted(.number.precision(.fractionLength(2)))) s · ANE calls \((aneMs / 1000).formatted(.number.precision(.fractionLength(2)))) s")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    if let resultSeed {
                        Text("Seed \(String(resultSeed))").font(.caption).textSelection(.enabled)
                    }
                }
                if let message = error ?? demo.error { Text(message).foregroundStyle(.red).textSelection(.enabled) }
            }
        }
        .formStyle(.grouped)
        .disabled(demo.busy)
        .frame(minWidth: 650, minHeight: 650)
        .navigationTitle(ModelTask.speech.title)
        .toolbar { ModelToolbar(demo: demo, task: .speech, showingInspector: $showingInspector, showModels: showModels) }
        .inspector(isPresented: $showingInspector) { ModelInspector(demo: demo) }
        .onChange(of: mode) {
            result = nil
            elapsed = nil
            aneMs = nil
            resultSeed = nil
            error = nil
        }
    }

    @ViewBuilder private func control(_ parameter: SpeechParameter) -> some View {
        let value = values[parameter.key] ?? parameter.defaultValue
        switch value {
        case .boolean(let enabled):
            Toggle(parameter.title, isOn: Binding(get: { enabled }, set: { values[parameter.key] = .boolean($0) }))
        case .number(let number):
            HStack {
                Text(parameter.title)
                Spacer()
                TextField(parameter.title, value: Binding(get: { number }, set: {
                    values[parameter.key] = .number(parameter.type == "int" ? $0.rounded() : $0)
                }), format: .number).labelsHidden().frame(width: 90).multilineTextAlignment(.trailing)
            }
        }
    }

    private func generate() async {
        guard seed.isEmpty || Int64(seed) != nil else { error = "Seed must be an integer."; return }
        guard mode != "Clone" || reference != nil || prompt != nil else {
            error = "Choose reference audio or a saved voice prompt."; return
        }
        demo.busy = true
        error = nil
        defer { demo.busy = false }
        do {
            let item = SpeechItem(text: text, language: isBlank(language) ? nil : language,
                                  instruct: mode == "Design" ? instruction : nil,
                                  refAudio: mode == "Clone" && prompt == nil ? reference?.path : nil,
                                  refText: mode == "Clone" && prompt == nil && !isBlank(transcript) ? transcript : nil,
                                  voiceClonePrompt: mode == "Clone" ? prompt?.path : nil,
                                  duration: fixedDuration ? duration : nil, speed: fixedDuration ? nil : speed)
            let reply = try await demo.runtime.speak(SpeechRequest(items: [item], parameters: values, seed: Int64(seed)))
            result = reply.audio?.first
            elapsed = reply.elapsedMs
            aneMs = reply.aneMs
            resultSeed = reply.seed
        } catch { self.error = error.localizedDescription }
    }

    private func pick(types: [UTType]) -> URL? {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = types
        panel.canChooseDirectories = false
        return panel.runModal() == .OK ? panel.url : nil
    }

    private func save(_ data: Data) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.wav]
        panel.nameFieldStringValue = "speech.wav"
        if panel.runModal() == .OK, let url = panel.url {
            do { try data.write(to: url) } catch { self.error = error.localizedDescription }
        }
    }

    private func saveVoice() async {
        guard let referenceURL = reference else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "voice.pt"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        demo.busy = true
        defer { demo.busy = false }
        do {
            let preprocess = values["preprocess_prompt"] ?? .boolean(true)
            let data = try await demo.runtime.createVoice(SpeechVoiceRequest(
                refAudio: referenceURL.path, refText: isBlank(transcript) ? nil : transcript,
                preprocessPrompt: preprocess == .boolean(true)))
            try data.write(to: url)
            prompt = url
            reference = nil
            transcript = ""
        } catch { self.error = error.localizedDescription }
    }
}
