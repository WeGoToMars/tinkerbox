import SwiftUI
import AppKit
import UniformTypeIdentifiers

// The Text to Speech page's text, voice, options and latest result; they outlive visits to other pages.
@MainActor @Observable final class SpeechPlayground {
    let demo: Demo
    var content = AttributedString("Hello! Welcome to Tinkerbox.")
    var selection = AttributedTextSelection()
    var mode = ""
    var useReference = false
    var language = ""
    var instruction = ""
    private(set) var reference: URL?
    private(set) var referenceData: Data?
    let recorder = VoiceRecording()
    var prompt: URL?  // a saved voice
    var transcript = ""
    var referenceLocale = ""  // empty: the system language
    var referenceLocales: [Locale] = []
    private(set) var transcriptionStatus: String?
    private(set) var transcriptionError: String?
    private var transcription: Task<Void, Never>?
    var seed: Int64?
    var values: [String: SpeechValue] = [:]  // request parameters
    var controls: [String: SpeechValue] = [:]  // the mode's per-item controls
    private(set) var result: SpeechAudio?
    private(set) var reply: SpeechReply?
    private(set) var live: StreamingAudio?
    var error: String?
    private var resultsModel: String?

    init(demo: Demo) { self.demo = demo }

    var text: String { String(content.characters) }
    var capabilities: SpeechCapabilities? { demo.speechCapabilities }
    var activeMode: SpeechMode? { capabilities?.modes.first { $0.id == mode } ?? capabilities?.modes.first }
    var usesReference: Bool {
        activeMode?.reference == .required || (activeMode?.reference == .optional && useReference)
    }
    var needsTranscript: Bool { activeMode?.referenceTranscript != Requirement.none }
    var streams: Bool { capabilities?.streaming == true }
    var nativeControls: [SpeechParameter] { (capabilities?.controls ?? []) + (activeMode?.controls ?? []) }

    var canGenerate: Bool {
        guard !demo.busy, !demo.loading, demo.compatibilityIssue == nil, demo.current?.kind == .speech,
            !recorder.active, !isBlank(text)
        else { return false }
        if activeMode?.instructionRequired == true && isBlank(instruction) { return false }
        guard usesReference, prompt == nil else { return true }
        return reference != nil && (!needsTranscript || (transcriptionStatus == nil && !isBlank(transcript)))
    }

    var canSaveVoice: Bool {
        !demo.busy && !demo.loading && demo.compatibilityIssue == nil && demo.current?.kind == .speech
            && !recorder.active && transcriptionStatus == nil && (!needsTranscript || !isBlank(transcript))
    }

    // Values and saved voices belong to one model.
    func modelChanged() {
        guard demo.current?.kind == .speech, demo.model != resultsModel else { return }
        resultsModel = demo.model
        values = [:]
        controls = [:]
        mode = capabilities?.modes.first?.id ?? ""
        useReference = false
        instruction = ""
        prompt = nil
        clearResult()
    }

    func modeChanged() {
        controls = [:]
        useReference = false
        recorder.disarm(discard: true)
        clearResult()
        transcribe(replacing: false)
    }

    func clearResult() {
        result = nil
        reply = nil
        live?.stop()
        live = nil
        error = nil
    }

    func setReference(_ url: URL?) {
        reference = url
        referenceData = nil
        prompt = nil
        transcript = ""
        transcribe()
        guard let url else { return }
        Task {
            do {
                let data = try await Task.detached { try Data(contentsOf: url) }.value
                if reference == url { referenceData = data }
            } catch { self.error = error.localizedDescription }
        }
    }

    func useSavedVoice(_ url: URL) {
        setReference(nil)
        prompt = url
    }

    // Recognizes the reference's words when the mode needs them; a newer attempt replaces an older one.
    // Without `replacing`, words the user entered stay.
    func transcribe(replacing: Bool = true) {
        cancelTranscription()
        transcriptionError = nil
        guard usesReference, needsTranscript, let reference, prompt == nil, replacing || isBlank(transcript) else {
            return
        }
        transcript = ""
        transcriptionStatus = "Preparing speech recognition…"
        let locale = referenceLocale.isEmpty ? Locale.current : Locale(identifier: referenceLocale)
        transcription = Task {
            do {
                let recognized = try await ReferenceTranscription.transcribe(reference, locale: locale) { status in
                    if !Task.isCancelled { self.transcriptionStatus = status }
                }
                try Task.checkCancellation()
                transcript = recognized
            } catch is CancellationError {
                return
            } catch {
                transcriptionError = error.localizedDescription
            }
            transcriptionStatus = nil
        }
    }

    func cancelTranscription() {
        transcription?.cancel()
        transcription = nil
        transcriptionStatus = nil
    }

    func generate(playWhileGenerating: Bool) async {
        demo.busy = true
        error = nil
        defer { demo.busy = false }
        // Streamed pieces play as they arrive; the complete reply then replaces them for replay and saving.
        let audio = streams && playWhileGenerating ? StreamingAudio() : nil
        live?.stop()
        live = audio
        if audio != nil { result = nil }
        defer { audio?.finish() }
        let item = SpeechItem(
            text: text, language: activeMode?.language != false && !isBlank(language) ? language : nil,
            instruct: activeMode?.instructionTitle != nil && !isBlank(instruction) ? instruction : nil,
            refAudio: usesReference && prompt == nil ? reference?.path : nil,
            refText: usesReference && prompt == nil && !isBlank(transcript) ? transcript : nil,
            voiceClonePrompt: usesReference ? prompt?.path : nil,
            mode: activeMode?.id, controls: controls.isEmpty ? nil : controls)
        let onAudio: (@Sendable (SpeechChunk) async -> Void)? = audio.map { audio in
            { @Sendable chunk in await audio.append(chunk) }
        }
        do {
            let reply = try await demo.runtime.speak(
                SpeechRequest(items: [item], parameters: values, seed: seed),
                onAudio: onAudio)
            self.reply = reply
            result = reply.audio.first
        } catch { self.error = error.localizedDescription }
    }

    func saveVoice(to url: URL) async {
        guard let reference else { return }
        demo.busy = true
        defer { demo.busy = false }
        do {
            let data = try await demo.runtime.createVoice(
                SpeechVoiceRequest(
                    refAudio: reference.path, refText: isBlank(transcript) ? nil : transcript,
                    preprocessPrompt: values["preprocess_prompt"].map { $0 == .boolean(true) }, mode: activeMode?.id))
            try data.write(to: url)
            useSavedVoice(url)
        } catch { self.error = error.localizedDescription }
    }

    func insertTag(_ tag: String) {
        let ranges: [Range<AttributedString.Index>]
        switch selection.indices(in: content) {
        case .insertionPoint(let index): ranges = [index..<index]
        case .ranges(let selections): ranges = Array(selections.ranges)
        }
        guard let first = ranges.first else { return }
        let cursor = content.characters.distance(from: content.startIndex, to: first.lowerBound) + tag.count
        for range in ranges.reversed() { content.replaceSubrange(range, with: AttributedString(tag)) }
        selection = AttributedTextSelection(
            insertionPoint: content.characters.index(content.startIndex, offsetBy: cursor))
    }
}

// A generated clip as a WAV file, for dragging and sharing.
struct SpeechFile: Transferable {
    static let name = "Speech.wav"
    let wav: Data

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(exportedContentType: .wav) { file in
            let folder = URL.temporaryDirectory.appending(path: UUID().uuidString)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let url = folder.appending(path: name)
            try file.wav.write(to: url)
            return SentTransferredFile(url)
        }
        .suggestedFileName(name)
    }
}

struct SpeechView: View {
    @Bindable var playground: SpeechPlayground
    let showModels: () -> Void
    @FocusState private var textFocused: Bool
    @FocusState private var recordFocused: Bool
    @State private var referencePlaybackFocused = false
    @State private var importing: [UTType]?
    @Environment(\.scenePhase) private var scenePhase
    @AppStorage("speech.playWhileGenerating") private var playWhileGenerating = true
    @AppStorage("showingInspector") private var showingInspector = false

    private var demo: Demo { playground.demo }
    private var recorder: VoiceRecording { playground.recorder }

    var body: some View {
        TaskPage {
            if demo.compatibilityIssue != nil {
                TaskSection { ModelAvailability(demo: demo, showModels: showModels) }
            }
            TaskSection("Text") {
                TaskTextEditor(
                    text: $playground.content, selection: $playground.selection, demo: demo, focus: $textFocused
                )
                .accessibilityLabel("Text to speak")
                if playground.activeMode?.language != false {
                    LabeledContent("Language") {
                        TextField("Language", text: $playground.language, prompt: Text("Automatic")).labelsHidden()
                    }
                }
                if let tags = demo.current?.nonVerbalTags, !tags.isEmpty {
                    DisclosureGroup("Non-verbal tags") {
                        LazyVGrid(
                            columns: [GridItem(.adaptive(minimum: 210), alignment: .leading)],
                            alignment: .leading, spacing: 8
                        ) {
                            ForEach(tags, id: \.self) { tag in
                                Button {
                                    playground.insertTag(tag)
                                    recordFocused = false
                                    textFocused = true
                                } label: {
                                    Text(tag).font(.system(.callout, design: .monospaced))
                                }.buttonStyle(.bordered).help("Insert \(tag) at the cursor")
                            }
                        }.padding(.vertical, 4)
                    }
                }
            }
            .disabled(demo.busy)
            // Live playback stays usable while the model generates; everything else waits.
            TaskSection {
                if playground.streams {
                    Toggle("Play while generating", isOn: $playWhileGenerating).disabled(demo.busy)
                }
                HStack {
                    Button(demo.busy ? "Generating…" : "Generate speech", systemImage: "waveform", action: generate)
                        .symbolEffect(.variableColor.iterative, isActive: demo.busy)
                        .buttonStyle(.borderedProminent)
                        .disabled(!playground.canGenerate)
                    if demo.busy { ProgressView().controlSize(.small) }
                    if let result = playground.result {
                        Menu("Save audio…", systemImage: "square.and.arrow.down") {
                            Button("WAV (.wav)…") { save(result.wav, type: .wav) }
                            Button("M4A (.m4a)…") { save(result.wav, type: .mpeg4Audio) }
                        }.menuStyle(.borderlessButton).fixedSize().disabled(demo.busy)
                        ShareLink(item: SpeechFile(wav: result.wav), preview: SharePreview("Speech")) {
                            Label("Share", systemImage: "square.and.arrow.up")
                        }
                        .labelStyle(.iconOnly).buttonStyle(.borderless).disabled(demo.busy)
                        Spacer()
                        Label(SpeechFile.name, systemImage: "waveform")
                            .font(.callout)
                            .padding(.horizontal, 10).padding(.vertical, 4)
                            .background(.quaternary, in: Capsule())
                            .draggable(SpeechFile(wav: result.wav))
                            .help("Drag into Finder, Mail or another app")
                    }
                }
                if let live = playground.live, !live.finished {
                    StreamingPlayback(audio: live).padding(.vertical, 8)
                } else if let result = playground.result {
                    AudioPlayback(
                        data: result.wav, spacebarEnabled: !recordFocused && !referencePlaybackFocused,
                        suspended: recorder.active || demo.busy || referencePlaybackFocused
                    ) {
                        recordFocused = false
                        referencePlaybackFocused = false
                    }
                    .disabled(recorder.active || demo.busy)
                    .padding(.vertical, 8)
                }
                if let result = playground.result, let reply = playground.reply {
                    Text(summary(result, reply)).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                }
                if let message = playground.error ?? demo.error, demo.compatibilityIssue == nil {
                    Text(message).foregroundStyle(.red).textSelection(.enabled)
                }
            }
        } settings: {
            TaskSection("Voice") {
                Picker("Mode", selection: $playground.mode) {
                    ForEach(playground.capabilities?.modes ?? []) { option in
                        Text(option.title).tag(option.id)
                    }
                }
                if let title = playground.activeMode?.instructionTitle {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(title)
                        TextField(
                            title, text: $playground.instruction,
                            prompt: Text(playground.activeMode?.instructionHint ?? ""), axis: .vertical
                        )
                        .labelsHidden()
                    }
                }
                ForEach(playground.nativeControls) { parameter in control(parameter, values: $playground.controls) }
                if playground.activeMode?.reference == .optional {
                    Toggle("Use reference voice", isOn: $playground.useReference)
                }
                if playground.usesReference { referenceControls }
            }
            .disabled(demo.busy)
            TaskSection("Sampling") {
                LabeledContent("Seed") {
                    TextField("Seed", value: $playground.seed, format: .number.grouping(.never), prompt: Text("Random"))
                        .labelsHidden().frame(maxWidth: 180)
                }
                DisclosureGroup("Advanced generation settings") {
                    ForEach(playground.capabilities?.parameters ?? []) { parameter in
                        control(parameter, values: $playground.values)
                    }
                }
            }
            .disabled(demo.busy)
        }
        .navigationTitle(ModelTask.speech.title)
        .toolbar {
            ModelToolbar(demo: demo, task: .speech, showingInspector: $showingInspector, showModels: showModels)
        }
        .inspector(isPresented: $showingInspector) { ModelInspector(demo: demo) }
        .focusedSceneValue(
            \.taskAction, TaskAction(title: "Generate Speech", enabled: playground.canGenerate, perform: generate)
        )
        .fileImporter(
            isPresented: Binding {
                importing != nil
            } set: {
                if !$0 { importing = nil }
            },
            allowedContentTypes: importing ?? []
        ) { result in
            guard case .success(let url) = result else { return }
            if importing == [.audio] { playground.setReference(url) } else { playground.useSavedVoice(url) }
        }
        .task { playground.referenceLocales = await ReferenceTranscription.locales }
        .onChange(of: playground.referenceLocale) { playground.transcribe() }
        .onChange(of: playground.useReference) { playground.transcribe(replacing: false) }
        .onChange(of: recordFocused) { if !recordFocused { recorder.disarm() } }
        .onChange(of: recorder.active) { if recorder.active { playground.live?.stop() } }
        .onChange(of: scenePhase) { if scenePhase != .active { recorder.stop() } }
        .onDisappear {
            recorder.disarm(discard: true); playground.live?.stop()
        }
        .onChange(of: demo.model, initial: true) { playground.modelChanged() }
        .onChange(of: playground.mode) {
            recordFocused = false
            playground.modeChanged()
        }
    }

    private func generate() {
        recordFocused = false
        referencePlaybackFocused = false
        Task { await playground.generate(playWhileGenerating: playWhileGenerating) }
    }

    @ViewBuilder private var referenceControls: some View {
        HStack {
            Button("Reference audio…") { importing = [.audio] }.disabled(recorder.active)
            if let reference = playground.reference {
                Text(reference.lastPathComponent).foregroundStyle(.secondary)
            }
        }
        .help("A clear 3–10 second recording of the voice")
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Button(
                    recorder.preparing ? "Cancel setup" : recorder.armed ? "Disarm microphone" : "Arm microphone",
                    systemImage: recorder.armed ? "mic.badge.xmark" : "mic.fill"
                ) {
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
                    if key.phase == .down && !demo.busy { recorder.start { playground.setReference($0) } }
                    if key.phase == .up { recorder.stop() }
                    return .handled
                }
                .help("Once armed, hold Space to record and release to finish; recording stops at 20 seconds.")
                if recorder.active {
                    HStack(spacing: 6) {
                        Image(systemName: "record.circle").foregroundStyle(.red).symbolEffect(.breathe)
                        Text("Recording · \(recorder.duration.formatted(.number.precision(.fractionLength(1)))) s")
                    }
                    .foregroundStyle(.secondary).monospacedDigit()
                } else if recorder.preparing {
                    ProgressView().controlSize(.small)
                } else if recorder.armed {
                    Text("Hold Space").foregroundStyle(.secondary)
                }
            }
            if recorder.active {
                AudioWaveform(peaks: recorder.peaks).frame(height: 88)
                    .accessibilityLabel("Recording waveform")
            }
            if let error = recorder.error { Text(error).font(.caption).foregroundStyle(.red) }
        }
        if !recorder.active, let data = playground.referenceData, let reference = playground.reference {
            AudioPlayback(
                data: data, spacebarEnabled: !recordFocused && (referencePlaybackFocused || playground.result == nil),
                fileExtension: reference.pathExtension,
                suspended: demo.busy || (!referencePlaybackFocused && playground.result != nil)
            ) {
                recordFocused = false
                referencePlaybackFocused = true
            }
            .padding(.vertical, 8)
        }
        if playground.prompt == nil && playground.needsTranscript {
            Picker("Reference language", selection: $playground.referenceLocale) {
                Text("System language").tag("")
                ForEach(playground.referenceLocales, id: \.identifier) { locale in
                    Text(Locale.current.localizedString(forIdentifier: locale.identifier) ?? locale.identifier)
                        .tag(locale.identifier)
                }
            }.disabled(recorder.active)
            HStack {
                TextField(
                    "Reference transcript", text: $playground.transcript,
                    prompt: Text(
                        playground.reference == nil
                            ? "Words spoken in the reference"
                            : "Review the recognized words, or type them"), axis: .vertical
                )
                .disabled(playground.transcriptionStatus != nil)
                Button("Transcribe") { playground.transcribe() }
                    .disabled(playground.reference == nil || playground.transcriptionStatus != nil)
            }.disabled(recorder.active)
            if let status = playground.transcriptionStatus {
                HStack {
                    ProgressView().controlSize(.small)
                    Text(status)
                    Spacer()
                    Button("Cancel") { playground.cancelTranscription() }
                }.font(.caption).foregroundStyle(.secondary)
            } else if let error = playground.transcriptionError {
                Text(error).font(.caption).foregroundStyle(.red)
            }
        }
        HStack {
            Button("Load saved voice…") { importing = [.data] }.disabled(recorder.active)
            if let prompt = playground.prompt {
                Text(prompt.lastPathComponent).foregroundStyle(.secondary)
            }
            if playground.reference != nil {
                Button("Save voice…") { saveVoice() }.disabled(!playground.canSaveVoice)
            }
        }
    }

    // Audio length, generation time against real time, first audio and the seed for replay.
    private func summary(_ result: SpeechAudio, _ reply: SpeechReply) -> String {
        let seconds = { (milliseconds: Double) in
            (milliseconds / 1000).formatted(.number.precision(.fractionLength(2))) + " s"
        }
        let speed = (result.duration * 1000 / max(reply.audioMs, 1)).formatted(.number.precision(.fractionLength(1)))
        return [
            "\(result.duration.formatted(.number.precision(.fractionLength(2)))) s audio",
            "generated in \(seconds(reply.audioMs)) (\(speed)× real time)",
            reply.firstAudioMs.map { "first audio after \(seconds($0))" },
            reply.cloneMs > 0 ? "voice prepared in \(seconds(reply.cloneMs))" : nil,
            "seed \(reply.seed)",
        ].compactMap { $0 }.joined(separator: " · ")
    }

    @ViewBuilder private func control(_ parameter: SpeechParameter, values: Binding<[String: SpeechValue]>) -> some View
    {
        let initial =
            parameter.defaultValue
            ?? (parameter.type == .bool
                ? .boolean(false)
                : parameter.type == .text || parameter.type == .enum ? .text("") : .number(parameter.min ?? 1))
        Group {
            if parameter.optional == true {
                Toggle(
                    parameter.title,
                    isOn: Binding(
                        get: { values.wrappedValue[parameter.key] != nil },
                        set: { values.wrappedValue[parameter.key] = $0 ? initial : nil }))
            }
            if parameter.optional != true || values.wrappedValue[parameter.key] != nil {
                controlField(parameter, value: values.wrappedValue[parameter.key] ?? initial, values: values)
            }
        }
        .help(parameter.help ?? "")
    }

    @ViewBuilder private func controlField(
        _ parameter: SpeechParameter, value: SpeechValue,
        values: Binding<[String: SpeechValue]>
    ) -> some View {
        switch value {
        case .boolean(let enabled):
            Toggle(
                parameter.title,
                isOn: Binding(get: { enabled }, set: { values.wrappedValue[parameter.key] = .boolean($0) }))
        case .text(let text):
            if let choices = parameter.choices {
                Picker(
                    parameter.title,
                    selection: Binding(get: { text }, set: { values.wrappedValue[parameter.key] = .text($0) })
                ) {
                    ForEach(choices) { choice in Text(choice.title).tag(choice.value) }
                }
            } else {
                LabeledContent(parameter.title) {
                    TextField(
                        parameter.title,
                        text: Binding(get: { text }, set: { values.wrappedValue[parameter.key] = .text($0) })
                    )
                    .labelsHidden()
                }
            }
        case .number(let number):
            LabeledContent(parameter.title) {
                TextField(
                    parameter.title,
                    value: Binding(
                        get: { number },
                        set: {
                            values.wrappedValue[parameter.key] = .number(parameter.type == .int ? $0.rounded() : $0)
                        }), format: .number
                ).labelsHidden().frame(width: 90).multilineTextAlignment(.trailing)
            }
        }
    }

    private func save(_ data: Data, type: UTType) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [type]
        panel.nameFieldStringValue = type == .wav ? "speech.wav" : "speech.m4a"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task {
            do { try await Task.detached { try exportAudio(data, to: url) }.value } catch {
                playground.error = error.localizedDescription
            }
        }
    }

    private func saveVoice() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "voice.pt"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task { await playground.saveVoice(to: url) }
    }
}
