import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct SpeechView: View {
    let demo: Demo
    let showModels: () -> Void
    @State private var text = "Hello! Welcome to Tinkerbox."
    @State private var textSelection: TextSelection?
    @FocusState private var textFocused: Bool
    @State private var mode = ""
    @State private var useReference = false
    @State private var language = ""
    @State private var instruction = ""
    @State private var reference: URL?
    @State private var referenceData: Data?
    @State private var recorder = VoiceRecording()
    @FocusState private var recordFocused: Bool
    @State private var referencePlaybackFocused = false
    @Environment(\.scenePhase) private var scenePhase
    @State private var prompt: URL?
    @State private var transcript = ""
    @State private var referenceLocale = ""
    @State private var referenceLocales: [Locale] = []
    @State private var transcriptionStatus: String?
    @State private var transcriptionError: String?
    @State private var transcriptionID: UUID?
    @State private var transcriptionAttempt = 0
    @State private var transcriptionCancelled = false
    @State private var seed = ""
    @State private var values: [String: SpeechValue] = [:]
    @State private var controls: [String: SpeechValue] = [:]
    @State private var result: SpeechAudio?
    @State private var elapsed: Double?
    @State private var cloneMs: Double?
    @State private var aneMs: Double?
    @State private var resultSeed: Int64?
    @State private var error: String?
    @State private var showingInspector = false

    private var activeMode: SpeechMode? {
        demo.speechCapabilities?.modes.first { $0.id == mode } ?? demo.speechCapabilities?.modes.first
    }
    private var usesReference: Bool {
        activeMode?.reference == "required" || (activeMode?.reference == "optional" && useReference)
    }
    private var nativeControls: [SpeechParameter] {
        (demo.speechCapabilities?.controls ?? []) + (activeMode?.controls ?? [])
    }

    var body: some View {
        Form {
            if demo.compatibilityIssue != nil {
                Section { ModelAvailability(demo: demo, showModels: showModels) }
            }
            Section("Text") {
                TextEditor(text: $text, selection: $textSelection)
                    .focused($textFocused).frame(minHeight: 120)
                if activeMode?.language != false {
                    TextField("Language name or code (automatic if empty)", text: $language)
                }
                Text("Pronunciation and non-verbal tags can be included directly in the text.")
                    .font(.caption).foregroundStyle(.secondary)
                if let tags = demo.current?.nonVerbalTags, !tags.isEmpty {
                    DisclosureGroup("Non-verbal tags") {
                        Text("Click a tag to insert it at the cursor.")
                            .font(.caption).foregroundStyle(.secondary)
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 210), alignment: .leading)],
                                  alignment: .leading, spacing: 8) {
                            ForEach(tags, id: \.self) { tag in
                                Button { insertTag(tag) } label: {
                                    Text(tag).font(.system(.callout, design: .monospaced))
                                }.buttonStyle(.bordered).help("Insert \(tag)")
                            }
                        }.padding(.vertical, 4)
                    }
                }
            }
            Section("Voice") {
                Picker("Mode", selection: $mode) {
                    ForEach(demo.speechCapabilities?.modes ?? []) { option in
                        Text(option.title).tag(option.id)
                    }
                }
                if let title = activeMode?.instructionTitle {
                    TextField(title, text: $instruction, prompt: Text(activeMode?.instructionHint ?? ""), axis: .vertical)
                }
                ForEach(nativeControls) { parameter in control(parameter, values: $controls) }
                if activeMode?.reference == "optional" {
                    Toggle("Use reference voice", isOn: $useReference)
                }
                if usesReference {
                    HStack {
                        Button("Reference audio…") {
                            if let url = pick(types: [.audio]) {
                                useReference(url)
                            }
                        }.disabled(recorder.active)
                        Text(reference?.lastPathComponent ?? "Choose a clear 3–10 second voice recording").foregroundStyle(.secondary)
                    }
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Button(recorder.preparing ? "Cancel setup" : recorder.armed ? "Disarm microphone" : "Arm microphone", systemImage: recorder.armed ? "mic.badge.xmark" : "mic.fill") {
                                if recorder.armed || recorder.preparing {
                                    recorder.disarm()
                                    recordFocused = false
                                } else {
                                    recordFocused = true
                                    recorder.arm()
                                }
                            }
                            .focusable()
                            .focused($recordFocused)
                            .onKeyPress(.space, phases: [.down, .repeat, .up]) { key in
                                guard recorder.armed else { return .ignored }
                                if key.phase == .down { startRecording() }
                                if key.phase == .up { recorder.stop() }
                                return .handled
                            }
                            if recorder.active {
                                Text("Recording · \(recorder.duration.formatted(.number.precision(.fractionLength(1)))) s")
                                    .foregroundStyle(.secondary).monospacedDigit()
                            } else {
                                Text(recorder.preparing ? "Preparing microphone…" : recorder.armed ?
                                     "Armed · Hold Space to record; release to finish." :
                                     "Arm the microphone, then hold Space to record.")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                        if recorder.active {
                            AudioWaveform(peaks: recorder.peaks).frame(height: 88)
                                .accessibilityLabel("Recording waveform")
                            Text("Release Space to finish. Aim for 3–10 seconds; recording stops at 20 seconds.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        if let error = recorder.error { Text(error).font(.caption).foregroundStyle(.red) }
                    }
                    if !recorder.active, let referenceData {
                        AudioPlayback(data: referenceData, spacebarEnabled: !recordFocused && (referencePlaybackFocused || result == nil),
                                      fileExtension: reference?.pathExtension ?? "wav",
                                      suspended: demo.busy || (!referencePlaybackFocused && result != nil)) {
                            recordFocused = false
                            referencePlaybackFocused = true
                        }
                            .padding(.vertical, 8)
                    }
                    if prompt == nil {
                        Picker("Reference language", selection: $referenceLocale) {
                            Text("System language").tag("")
                            ForEach(referenceLocales, id: \.identifier) { locale in
                                Text(Locale.current.localizedString(forIdentifier: locale.identifier) ?? locale.identifier)
                                    .tag(locale.identifier)
                            }
                        }.disabled(recorder.active)
                        HStack {
                            TextField("Reference transcript", text: $transcript, axis: .vertical)
                                .disabled(transcriptionStatus != nil || recorder.active)
                            Button("Transcribe") {
                                transcript = ""
                                transcriptionCancelled = false
                                transcriptionAttempt += 1
                            }.disabled(reference == nil || transcriptionStatus != nil || recorder.active)
                        }.disabled(recorder.active)
                        if let transcriptionStatus {
                            HStack {
                                ProgressView().controlSize(.small)
                                Text(transcriptionStatus)
                                Spacer()
                                Button("Cancel") {
                                    transcriptionCancelled = true
                                    transcriptionAttempt += 1
                                }
                            }.font(.caption).foregroundStyle(.secondary)
                        } else if let transcriptionError {
                            Text(transcriptionError).font(.caption).foregroundStyle(.red)
                        } else {
                            Text(reference == nil ? "Choose reference audio to transcribe it on this Mac." :
                                 "Review and edit the words to match your recording, or enter them manually.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    HStack {
                        Button("Load saved voice…") {
                            if let url = pick(types: [.data]) { prompt = url; reference = nil; transcript = "" }
                        }.disabled(recorder.active)
                        Text(prompt?.lastPathComponent ?? "Reusable voice prompt").foregroundStyle(.secondary)
                        if reference != nil {
                            Button("Save voice…") { Task { await saveVoice() } }
                                .disabled(demo.loading || demo.compatibilityIssue != nil || demo.current?.kind != .speech || recorder.active || transcriptionStatus != nil || isBlank(transcript))
                        }
                    }
                }
            }
            Section("Sampling") {
                TextField("Random seed (optional)", text: $seed)
                DisclosureGroup("Advanced generation settings") {
                    ForEach(demo.speechParameters) { parameter in control(parameter, values: $values) }
                }
            }
            Section {
                HStack {
                    Button(demo.busy ? "Generating…" : "Generate speech", systemImage: "waveform") {
                        Task { await generate() }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(demo.busy || demo.loading || demo.compatibilityIssue != nil || recorder.active || demo.current?.kind != .speech || isBlank(text) ||
                              (usesReference && prompt == nil && (transcriptionStatus != nil || isBlank(transcript))) ||
                              (activeMode?.instructionRequired == true && isBlank(instruction)))
                    if demo.busy { ProgressView().controlSize(.small) }
                    if let result {
                        Menu("Save audio…", systemImage: "square.and.arrow.down") {
                            Button("WAV (.wav)…") { save(result.wav, type: .wav) }
                            Button("M4A (.m4a)…") { save(result.wav, type: .mpeg4Audio) }
                        }.menuStyle(.borderlessButton).fixedSize()
                    }
                }
                if let result {
                    AudioPlayback(data: result.wav, spacebarEnabled: !recordFocused && !referencePlaybackFocused,
                                  suspended: recorder.active || demo.busy || referencePlaybackFocused) {
                        recordFocused = false
                        referencePlaybackFocused = false
                    }
                        .disabled(recorder.active)
                        .padding(.vertical, 8)
                    Text("\(result.duration.formatted(.number.precision(.fractionLength(2)))) seconds · \(result.sampleRate) Hz")
                    if let elapsed, let aneMs {
                        if let cloneMs, cloneMs > 0 {
                            Text("Voice prepared in \((cloneMs / 1000).formatted(.number.precision(.fractionLength(2)))) s")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Text("Speech generated in \((elapsed / 1000).formatted(.number.precision(.fractionLength(2)))) s · ANE calls \((aneMs / 1000).formatted(.number.precision(.fractionLength(2)))) s")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    if let resultSeed {
                        Text("Seed \(String(resultSeed))").font(.caption).textSelection(.enabled)
                    }
                }
                if let message = error ?? demo.error, demo.compatibilityIssue == nil {
                    Text(message).foregroundStyle(.red).textSelection(.enabled)
                }
            }
        }
        .formStyle(.grouped)
        .disabled(demo.busy)
        .frame(minWidth: 650, minHeight: 650)
        .navigationTitle(ModelTask.speech.title)
        .toolbar { ModelToolbar(demo: demo, task: .speech, showingInspector: $showingInspector, showModels: showModels) }
        .inspector(isPresented: $showingInspector) { ModelInspector(demo: demo) }
        .task { referenceLocales = await ReferenceTranscription.locales }
        .task(id: reference) {
            referenceData = nil
            guard let reference else { return }
            do {
                let data = try await Task.detached { try Data(contentsOf: reference) }.value
                try Task.checkCancellation()
                referenceData = data
            } catch is CancellationError { }
            catch { self.error = error.localizedDescription }
        }
        .task(id: [String(usesReference), reference?.path ?? "", referenceLocale, String(transcriptionAttempt)]) {
            await transcribeReference()
        }
        .onChange(of: referenceLocale) { transcript = ""; transcriptionCancelled = false }
        .onChange(of: transcript) { if !isBlank(transcript) { transcriptionError = nil } }
        .onChange(of: recordFocused) { if !recordFocused { recorder.disarm() } }
        .onChange(of: scenePhase) { if scenePhase != .active { recorder.stop() } }
        .onDisappear { recorder.disarm(discard: true) }
        .onChange(of: demo.model) {
            values = [:]
            controls = [:]
            mode = ""
            useReference = false
            instruction = ""
            prompt = nil
            result = nil
            elapsed = nil
            cloneMs = nil
            aneMs = nil
            resultSeed = nil
            error = nil
        }
        .onChange(of: demo.speechCapabilities?.modes.map(\.id), initial: true) {
            guard let modes = demo.speechCapabilities?.modes else { return }
            if !modes.contains(where: { $0.id == mode }) {
                mode = modes.first?.id ?? ""
            }
        }
        .onChange(of: mode) {
            controls = [:]
            useReference = false
            recorder.disarm(discard: true)
            recordFocused = false
            result = nil
            elapsed = nil
            cloneMs = nil
            aneMs = nil
            resultSeed = nil
            error = nil
        }
    }

    private func insertTag(_ tag: String) {
        let ranges: [Range<String.Index>]
        switch textSelection?.indices {
        case .selection(let range): ranges = [range]
        case .multiSelection(let selections): ranges = Array(selections.ranges)
        default: ranges = [text.endIndex..<text.endIndex]
        }
        guard let first = ranges.first else { return }
        let cursor = text.distance(from: text.startIndex, to: first.lowerBound) + tag.count
        var updated = text
        for range in ranges.reversed() { updated.replaceSubrange(range, with: tag) }
        text = updated
        textSelection = TextSelection(insertionPoint: updated.index(updated.startIndex, offsetBy: cursor))
        recordFocused = false
        textFocused = true
    }

    private func startRecording() {
        guard !demo.busy else { return }
        recorder.start { url in useReference(url) }
    }

    private func useReference(_ url: URL) {
        reference = url
        prompt = nil
        transcript = ""
        transcriptionCancelled = false
        transcriptionAttempt += 1
    }

    private func transcribeReference() async {
        let id = UUID()
        transcriptionID = id
        transcriptionStatus = nil
        transcriptionError = nil
        guard usesReference, let reference, prompt == nil, isBlank(transcript), !transcriptionCancelled else { return }
        transcriptionStatus = "Preparing speech recognition…"
        defer { if transcriptionID == id { transcriptionStatus = nil } }
        do {
            let locale = referenceLocale.isEmpty ? Locale.current : Locale(identifier: referenceLocale)
            let recognized = try await ReferenceTranscription.transcribe(reference, locale: locale) { status in
                if transcriptionID == id && !Task.isCancelled { transcriptionStatus = status }
            }
            try Task.checkCancellation()
            if transcriptionID == id { transcript = recognized }
        } catch is CancellationError { }
        catch {
            if transcriptionID == id && !Task.isCancelled { transcriptionError = error.localizedDescription }
        }
    }

    @ViewBuilder private func control(_ parameter: SpeechParameter, values: Binding<[String: SpeechValue]>) -> some View {
        let initial = parameter.defaultValue ?? (parameter.type == "bool" ? .boolean(false) :
                      parameter.type == "text" || parameter.type == "enum" ? .text("") : .number(parameter.min ?? 1))
        VStack(alignment: .leading, spacing: 4) {
            if parameter.optional == true {
                Toggle(parameter.title, isOn: Binding(
                    get: { values.wrappedValue[parameter.key] != nil },
                    set: { values.wrappedValue[parameter.key] = $0 ? initial : nil }))
            }
            if parameter.optional != true || values.wrappedValue[parameter.key] != nil {
                controlField(parameter, value: values.wrappedValue[parameter.key] ?? initial, values: values)
            }
            if let help = parameter.help {
                Text(help).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder private func controlField(_ parameter: SpeechParameter, value: SpeechValue,
                                          values: Binding<[String: SpeechValue]>) -> some View {
        switch value {
        case .boolean(let enabled):
            Toggle(parameter.title, isOn: Binding(get: { enabled }, set: { values.wrappedValue[parameter.key] = .boolean($0) }))
        case .text(let text):
            if let choices = parameter.choices {
                Picker(parameter.title, selection: Binding(get: { text }, set: { values.wrappedValue[parameter.key] = .text($0) })) {
                    ForEach(choices) { choice in Text(choice.title).tag(choice.value) }
                }
            } else {
                TextField(parameter.title, text: Binding(get: { text }, set: { values.wrappedValue[parameter.key] = .text($0) }))
            }
        case .number(let number):
            HStack {
                Text(parameter.title)
                Spacer()
                TextField(parameter.title, value: Binding(get: { number }, set: {
                    values.wrappedValue[parameter.key] = .number(parameter.type == "int" ? $0.rounded() : $0)
                }), format: .number).labelsHidden().frame(width: 90).multilineTextAlignment(.trailing)
            }
        }
    }

    private func generate() async {
        guard seed.isEmpty || Int64(seed) != nil else { error = "Seed must be an integer."; return }
        guard !usesReference || reference != nil || prompt != nil else {
            error = "Choose reference audio or a saved voice prompt."; return
        }
        recordFocused = false
        referencePlaybackFocused = false
        demo.busy = true
        error = nil
        defer { demo.busy = false }
        do {
            let item = SpeechItem(text: text, language: activeMode?.language != false && !isBlank(language) ? language : nil,
                                  instruct: activeMode?.instructionTitle != nil && !isBlank(instruction) ? instruction : nil,
                                  refAudio: usesReference && prompt == nil ? reference?.path : nil,
                                  refText: usesReference && prompt == nil && !isBlank(transcript) ? transcript : nil,
                                  voiceClonePrompt: usesReference ? prompt?.path : nil,
                                  mode: activeMode?.id, controls: controls.isEmpty ? nil : controls)
            let reply = try await demo.runtime.speak(SpeechRequest(items: [item], parameters: values, seed: Int64(seed)))
            result = reply.audio?.first
            elapsed = reply.audioMs
            cloneMs = usesReference ? reply.cloneMs : nil
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

    private func save(_ data: Data, type: UTType) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [type]
        panel.nameFieldStringValue = type == .wav ? "speech.wav" : "speech.m4a"
        if panel.runModal() == .OK, let url = panel.url {
            Task {
                do { try await Task.detached { try exportAudio(data, to: url) }.value }
                catch { self.error = error.localizedDescription }
            }
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
