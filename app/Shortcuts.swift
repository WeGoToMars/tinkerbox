import AppIntents
import Foundation
import UniformTypeIdentifiers

// Shortcuts hands over file data; the worker opens images by path.
func decide(text: String?, images: [IntentFile]?, questions: [Question]) async throws -> Decisions {
    let folder = FileManager.default.temporaryDirectory.appending(path: "tinkerbox-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let paths = try (images ?? []).enumerated().map { index, file in
        let url = folder.appending(path: "\(index)")  // Pillow detects the format from the contents.
        try file.data.write(to: url)
        return url.path
    }
    return try await Runtime.shared.decide(Request(items: [Item(state: text ?? "", images: paths)], questions: questions))
}

// One question about each nonblank list item, packed into one batch; blank lines are dropped.
func decide(items: [String], question: Question) async throws -> [(item: String, answer: Answer)] {
    let items = items.filter { !isBlank($0) }
    guard !items.isEmpty else { return [] }
    let decisions = try await Runtime.shared.decide(Request(items: items.map { Item(state: $0) }, questions: [question]))
    return zip(items, decisions.results.map { $0.answers[0] }).map { ($0, $1) }
}

func percent(_ value: Double) -> String { value.formatted(.percent.precision(.fractionLength(0))) }

// Each result reads as text through its title (e.g. "Yes (93%)") and exposes typed properties.
struct DecisionProbability: TransientAppEntity {
    static let typeDisplayRepresentation = TypeDisplayRepresentation(name: "Decision Probability")
    @Property(title: "Label") var label: String
    @Property(title: "Probability") var probability: Double

    var displayRepresentation: DisplayRepresentation {
        DisplayRepresentation(title: "\(label) (\(percent(probability)))")
    }

    init() {}
    init(_ value: Probability) {
        label = value.label
        probability = value.value
    }
}

struct YesNoDecision: TransientAppEntity {
    static let typeDisplayRepresentation = TypeDisplayRepresentation(name: "Yes/No Decision")
    @Property(title: "Question") var question: String
    @Property(title: "Answer") var answer: Bool
    @Property(title: "Yes Probability") var yesProbability: Double
    @Property(title: "Probabilities") var probabilities: [DecisionProbability]
    @Property(title: "Input Tokens") var inputTokens: Int
    @Property(title: "Inference Time (ms)") var inferenceTime: Double

    // The probability of the given answer: "No (88%)" means 12% Yes.
    var displayRepresentation: DisplayRepresentation {
        DisplayRepresentation(title: "\(answer ? "Yes" : "No") (\(percent(answer ? yesProbability : 1 - yesProbability)))",
                              subtitle: "\(question)")
    }

    init() {}
    init(question: String, threshold: Double, answer value: Answer, tokens: Int, elapsed: Double) throws {
        guard value.type == .noul, let probability = value.noul else { throw DecisionError.invalidReply }
        self.question = question
        answer = probability >= threshold
        yesProbability = probability
        probabilities = value.probabilities.map(DecisionProbability.init)
        inputTokens = tokens
        inferenceTime = elapsed
    }
}

struct ChoiceDecision: TransientAppEntity {
    static let typeDisplayRepresentation = TypeDisplayRepresentation(name: "Choice Decision")
    @Property(title: "Chosen Option") var chosenOption: String?
    @Property(title: "Most Likely Option") var mostLikelyOption: String
    @Property(title: "Confidence") var confidence: Double
    @Property(title: "Probabilities") var probabilities: [DecisionProbability]
    @Property(title: "Input Tokens") var inputTokens: Int
    @Property(title: "Inference Time (ms)") var inferenceTime: Double

    var displayRepresentation: DisplayRepresentation {
        let likely = "\(mostLikelyOption) (\(percent(confidence)))"
        return DisplayRepresentation(title: "\(chosenOption == nil ? "Unsure: \(likely)" : likely)")
    }

    init() {}
    init(_ decisions: Decisions, minimumConfidence: Double) throws {
        let value = decisions.results[0].answers[0]
        guard value.type == .choice, let option = value.choice, let confidence = value.confidence else {
            throw DecisionError.invalidReply
        }
        chosenOption = confidence >= minimumConfidence ? option : nil
        mostLikelyOption = option
        self.confidence = confidence
        probabilities = value.probabilities.map(DecisionProbability.init)
        inputTokens = decisions.results[0].tokens
        inferenceTime = decisions.elapsedMs
    }
}

struct ScoreDecision: TransientAppEntity {
    static let typeDisplayRepresentation = TypeDisplayRepresentation(name: "Score Decision")
    @Property(title: "Score") var score: Double
    @Property(title: "Maximum Score") var maximumScore: Int
    @Property(title: "Most Likely Level") var level: String
    @Property(title: "Probabilities") var probabilities: [DecisionProbability]
    @Property(title: "Input Tokens") var inputTokens: Int
    @Property(title: "Inference Time (ms)") var inferenceTime: Double

    var displayRepresentation: DisplayRepresentation {
        DisplayRepresentation(title: "\(level) (\(score.formatted(.number.precision(.fractionLength(1)))) / \(maximumScore))")
    }

    init() {}
    init(_ decisions: Decisions, levels: [String]) throws {
        let value = decisions.results[0].answers[0]
        (score, level) = try scored(value, levels: levels)
        maximumScore = levels.count - 1
        probabilities = value.probabilities.map(DecisionProbability.init)
        inputTokens = decisions.results[0].tokens
        inferenceTime = decisions.elapsedMs
    }
}

// The expected score and the most likely level of a score answer.
func scored(_ value: Answer, levels: [String]) throws -> (Double, String) {
    let values = value.probabilities.map(\.value)
    guard value.type == .score, let score = value.score, values.count == levels.count,
          let best = values.indices.max(by: { values[$0] < values[$1] }) else { throw DecisionError.invalidReply }
    return (score, levels[best])
}

struct RankedItem: TransientAppEntity {
    static let typeDisplayRepresentation = TypeDisplayRepresentation(name: "Ranked Item")
    @Property(title: "Item") var item: String
    @Property(title: "Score") var score: Double
    @Property(title: "Maximum Score") var maximumScore: Int
    @Property(title: "Most Likely Level") var level: String

    // Reads as the original item, so a sorted list can be used as text directly.
    var displayRepresentation: DisplayRepresentation {
        DisplayRepresentation(title: "\(item)",
                              subtitle: "\(level) (\(score.formatted(.number.precision(.fractionLength(1)))) / \(maximumScore))")
    }

    init() {}
    init(item: String, answer: Answer, levels: [String]) throws {
        self.item = item
        (score, level) = try scored(answer, levels: levels)
        maximumScore = levels.count - 1
    }
}

struct YesNoDecisionIntent: AppIntent {
    static let title: LocalizedStringResource = "Yes/No Decision"
    static let description = IntentDescription(
        "Answer a yes/no question about text or images, with a confidence score. Uses the model selected on Tinkerbox’s Decision screen."
    )
    static var supportedModes: IntentModes { .background }

    @Parameter(title: "Input Text") var text: String?
    @Parameter(title: "Question", description: "For example: Is this a refund request?") var question: String
    @Parameter(title: "Images", description: "Images to evaluate. Requires a model that accepts images.",
               supportedContentTypes: [.image]) var images: [IntentFile]?
    @Parameter(title: "Yes Threshold", description: "Answer Yes when the Yes Probability is at least this value (0–1).",
               default: 0.5, inclusiveRange: (0, 1)) var threshold: Double

    static var parameterSummary: some ParameterSummary {
        Summary("Decide \(\.$question) about \(\.$text)") {
            \.$images
            \.$threshold
        }
    }

    func perform() async throws -> some IntentResult & ReturnsValue<YesNoDecision> {
        let decisions = try await decide(text: text, images: images,
                                         questions: [Question(type: .noul, instructions: question)])
        return .result(value: try YesNoDecision(question: question, threshold: threshold,
                                                answer: decisions.results[0].answers[0],
                                                tokens: decisions.results[0].tokens, elapsed: decisions.elapsedMs))
    }
}

struct ChooseOptionIntent: AppIntent {
    static let title: LocalizedStringResource = "Choose Option"
    static let description = IntentDescription(
        "Choose the best option for text or images. Returns no choice below Minimum Confidence. Uses the model selected on Tinkerbox’s Decision screen."
    )
    static var supportedModes: IntentModes { .background }

    @Parameter(title: "Input Text") var text: String?
    @Parameter(title: "Question", description: "For example: What is the topic of this message?") var question: String
    @Parameter(title: "Options", description: "At least two distinct, nonempty option names.") var options: [String]
    @Parameter(title: "Images", description: "Images to evaluate. Requires a model that accepts images.",
               supportedContentTypes: [.image]) var images: [IntentFile]?
    @Parameter(title: "Minimum Confidence",
               description: "Leave Chosen Option empty when the most likely option's probability is below this value (0–1).",
               default: 0, inclusiveRange: (0, 1)) var minimumConfidence: Double

    static var parameterSummary: some ParameterSummary {
        Summary("Choose from \(\.$options) for \(\.$text)") {
            \.$question
            \.$images
            \.$minimumConfidence
        }
    }

    func perform() async throws -> some IntentResult & ReturnsValue<ChoiceDecision> {
        let decisions = try await decide(text: text, images: images,
                                         questions: [Question(type: .choice, instructions: question, criteria: options)])
        return .result(value: try ChoiceDecision(decisions, minimumConfidence: minimumConfidence))
    }
}

struct ScoreTextIntent: AppIntent {
    static let title: LocalizedStringResource = "Score Text"
    static let description = IntentDescription(
        "Rate text or images using your ordered levels. Uses the model selected on Tinkerbox’s Decision screen."
    )
    static var supportedModes: IntentModes { .background }

    @Parameter(title: "Input Text") var text: String?
    @Parameter(title: "Question", description: "For example: How urgent is this request?") var question: String
    @Parameter(title: "Levels", description: "2–10 nonempty levels, ordered from lowest to highest.") var levels: [String]
    @Parameter(title: "Images", description: "Images to evaluate. Requires a model that accepts images.",
               supportedContentTypes: [.image]) var images: [IntentFile]?

    static var parameterSummary: some ParameterSummary {
        Summary("Score \(\.$text) using \(\.$levels)") {
            \.$question
            \.$images
        }
    }

    func perform() async throws -> some IntentResult & ReturnsValue<ScoreDecision> {
        let decisions = try await decide(text: text, images: images,
                                         questions: [Question(type: .score, instructions: question, criteria: levels)])
        return .result(value: try ScoreDecision(decisions, levels: levels))
    }
}

struct AskQuestionsIntent: AppIntent {
    static let title: LocalizedStringResource = "Ask Several Questions"
    static let description = IntentDescription(
        "Answer several yes/no questions about the same text or images. Returns answers in question order. Uses the model selected on Tinkerbox’s Decision screen."
    )
    static var supportedModes: IntentModes { .background }

    @Parameter(title: "Input Text") var text: String?
    @Parameter(title: "Questions", description: "A list of nonempty yes/no questions.") var questions: [String]
    @Parameter(title: "Images", description: "Images to evaluate. Requires a model that accepts images.",
               supportedContentTypes: [.image]) var images: [IntentFile]?
    @Parameter(title: "Yes Threshold", description: "Answer Yes when the Yes Probability is at least this value (0–1).",
               default: 0.5, inclusiveRange: (0, 1)) var threshold: Double

    static var parameterSummary: some ParameterSummary {
        Summary("Ask \(\.$questions) about \(\.$text)") {
            \.$images
            \.$threshold
        }
    }

    func perform() async throws -> some IntentResult & ReturnsValue<[YesNoDecision]> {
        let decisions = try await decide(text: text, images: images,
                                         questions: questions.map { Question(type: .noul, instructions: $0) })
        let result = decisions.results[0]
        return .result(value: try zip(questions, result.answers).map { question, answer in
            try YesNoDecision(question: question, threshold: threshold, answer: answer,
                              tokens: result.tokens, elapsed: decisions.elapsedMs)
        })
    }
}

struct FilterListIntent: AppIntent {
    static let title: LocalizedStringResource = "Filter List by Question"
    static let description = IntentDescription(
        "Keep list items that answer Yes to your question, in their original order. Skips blank items. Uses the model selected on Tinkerbox’s Decision screen."
    )
    static var supportedModes: IntentModes { .background }

    @Parameter(title: "List", description: "Text items to test, for example from Split Text.") var items: [String]
    @Parameter(title: "Question", description: "Asked about each item. For example: Is this a refund request?") var question: String
    @Parameter(title: "Yes Threshold", description: "Keep an item when its Yes Probability is at least this value (0–1).",
               default: 0.5, inclusiveRange: (0, 1)) var threshold: Double

    static var parameterSummary: some ParameterSummary {
        Summary("Keep items in \(\.$items) where \(\.$question)") {
            \.$threshold
        }
    }

    func perform() async throws -> some IntentResult & ReturnsValue<[String]> {
        let answers = try await decide(items: items, question: Question(type: .noul, instructions: question))
        return .result(value: answers.filter { ($0.answer.noul ?? 0) >= threshold }.map(\.item))
    }
}

struct SortListIntent: AppIntent {
    static let title: LocalizedStringResource = "Sort List by Score"
    static let description = IntentDescription(
        "Rank list items using your question and ordered levels. Skips blank items and keeps ties in their original order. Uses the model selected on Tinkerbox’s Decision screen."
    )
    static var supportedModes: IntentModes { .background }

    @Parameter(title: "List", description: "Text items to rank, for example from Split Text.") var items: [String]
    @Parameter(title: "Question", description: "Asked about each item. For example: How urgent is this request?") var question: String
    @Parameter(title: "Levels", description: "2–10 nonempty levels, ordered from lowest to highest.") var levels: [String]
    @Parameter(title: "Highest First", default: true) var highestFirst: Bool

    static var parameterSummary: some ParameterSummary {
        Summary("Sort \(\.$items) by \(\.$question)") {
            \.$levels
            \.$highestFirst
        }
    }

    func perform() async throws -> some IntentResult & ReturnsValue<[RankedItem]> {
        let answers = try await decide(items: items, question: Question(type: .score, instructions: question, criteria: levels))
        let ranked = try answers.map { try RankedItem(item: $0.item, answer: $0.answer, levels: levels) }
        // Ties keep the list's order.
        let order = ranked.indices.sorted { a, b in
            ranked[a].score == ranked[b].score ? a < b : (ranked[a].score > ranked[b].score) == highestFirst
        }
        return .result(value: order.map { ranked[$0] })
    }
}

struct SpeechModelEntity: AppEntity {
    static let typeDisplayRepresentation = TypeDisplayRepresentation(name: "Speech Model")
    static var defaultQuery = SpeechModelQuery()
    let id: String
    let name: String
    var displayRepresentation: DisplayRepresentation { DisplayRepresentation(title: "\(name)") }
}

struct SpeechModelQuery: EntityQuery {
    func suggestedEntities() async throws -> [SpeechModelEntity] {
        Catalog.bundled.filter { $0.kind == .speech && Catalog.speechCapabilities[$0.id] != nil }
            .map { SpeechModelEntity(id: $0.id, name: $0.name) }
    }

    func entities(for identifiers: [SpeechModelEntity.ID]) async throws -> [SpeechModelEntity] {
        let available = try await suggestedEntities()
        return identifiers.compactMap { id in available.first { $0.id == id } }
    }
}

// Persist the model and mode IDs; app selection changes never change a saved Shortcut.
struct SpeechModeEntity: AppEntity {
    static let typeDisplayRepresentation = TypeDisplayRepresentation(name: "Speech Mode")
    static var defaultQuery = SpeechModeQuery()

    let modelID: String
    let modeID: String
    let title: String
    var id: String { "\(modelID)/\(modeID)" }
    var displayRepresentation: DisplayRepresentation { DisplayRepresentation(title: "\(title)") }

    init(model: String, mode: SpeechMode) {
        modelID = model
        modeID = mode.id
        title = mode.title
    }
}

struct SpeechModeQuery: EntityQuery {
    func suggestedEntities() async throws -> [SpeechModeEntity] {
        let models = try await SpeechModelQuery().suggestedEntities()
        return models.flatMap { model in
            (Catalog.speechCapabilities[model.id]?.modes ?? []).map { SpeechModeEntity(model: model.id, mode: $0) }
        }
    }

    func entities(for identifiers: [SpeechModeEntity.ID]) async throws -> [SpeechModeEntity] {
        let available = try await suggestedEntities()
        return identifiers.compactMap { id in available.first { $0.id == id } }
    }
}

struct SpeechGenerationModeOptions: DynamicOptionsProvider {
    @IntentParameterDependency<GenerateSpeechIntent>(\.$model) var intent

    func results() async throws -> [SpeechModeEntity] {
        guard let model = intent?.model else { return [] }
        return (Catalog.speechCapabilities[model.id]?.modes ?? []).map { SpeechModeEntity(model: model.id, mode: $0) }
    }
}

struct SpeechVoiceModeOptions: DynamicOptionsProvider {
    @IntentParameterDependency<CreateVoicePromptIntent>(\.$model) var intent

    func results() async throws -> [SpeechModeEntity] {
        guard let model = intent?.model else { return [] }
        return (Catalog.speechCapabilities[model.id]?.modes ?? [])
            .filter { $0.reference == "required" || $0.reference == "optional" }
            .map { SpeechModeEntity(model: model.id, mode: $0) }
    }
}

// The backend validates the mode itself; a saved Shortcut can still pair a mode with another model.
func speechMode(_ selection: SpeechModeEntity?, model: SpeechModelEntity) throws -> String? {
    guard let selection else { return nil }
    guard selection.modelID == model.id else {
        throw DecisionError.invalid("\(model.name) does not support the selected mode ‘\(selection.title)’. Choose a mode for this model.")
    }
    return selection.modeID
}

struct GenerateSpeechIntent: AppIntent {
    static let title: LocalizedStringResource = "Generate Speech"
    static let description = IntentDescription(
        "Turn text into WAV audio files, ready to play or save. The model is saved with this action."
    )
    static var supportedModes: IntentModes { .background }

    @Parameter(title: "Text", description: "One text or a list. Returns one audio file per item in the same order.") var texts: [String]
    @Parameter(title: "Model") var model: SpeechModelEntity
    @Parameter(title: "Mode", optionsProvider: SpeechGenerationModeOptions()) var mode: SpeechModeEntity?
    @Parameter(title: "Voice Description", description: "A voice or delivery instruction supported by the selected mode.") var instruction: String?
    @Parameter(title: "Reference Audio", description: "Audio to clone or continue. Use this or Saved Voice, not both.",
               supportedContentTypes: [.audio]) var reference: IntentFile?
    @Parameter(title: "Reference Transcript", description: "Words spoken in the reference. Transcribed on-device when the mode needs it and this is empty.") var transcript: String?
    @Parameter(title: "Saved Voice", description: "A voice.pt file from Create Voice Prompt or Tinkerbox.",
               supportedContentTypes: [.data]) var voice: IntentFile?
    @Parameter(title: "Language", description: "Optional language code, when supported by the model.") var language: String?
    @Parameter(title: "Target Duration", description: "Seconds per text, when the model supports a native target duration.") var duration: Double?
    @Parameter(title: "Speaking Speed", description: "Native speed factor, when supported. Above 1 is faster; below 1 is slower.") var speed: Double?
    @Parameter(title: "Seed", description: "A nonnegative integer for replay. Leave empty for a random seed.") var seed: Int?
    @Parameter(title: "Diffusion Steps", description: "Optional generation steps; leave empty for the model default.") var steps: Int?
    @Parameter(title: "Guidance Scale", description: "Optional native guidance scale; leave empty for the model default.") var guidance: Double?

    static var parameterSummary: some ParameterSummary {
        Summary("Generate speech for \(\.$texts) with \(\.$model)") {
            \.$mode
            \.$instruction
            \.$reference
            \.$transcript
            \.$voice
            \.$language
            \.$duration
            \.$speed
            \.$seed
            \.$steps
            \.$guidance
        }
    }

    func perform() async throws -> some IntentResult & ReturnsValue<[IntentFile]> {
        let folder = FileManager.default.temporaryDirectory.appending(path: "tinkerbox-speech-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        var referencePath: String?
        var voicePath: String?
        if let reference {
            let path = folder.appending(path: "reference.\(URL(fileURLWithPath: reference.filename).pathExtension)")
            try reference.data.write(to: path)
            referencePath = path.path
        }
        if let voice {
            let path = folder.appending(path: "voice.pt")
            try voice.data.write(to: path)
            voicePath = path.path
        }
        let selectedMode = try speechMode(mode, model: model)
        let items = texts.map { SpeechItem(text: $0, language: language.flatMap { isBlank($0) ? nil : $0 },
            instruct: instruction.flatMap { isBlank($0) ? nil : $0 }, refAudio: referencePath,
            refText: transcript.flatMap { isBlank($0) ? nil : $0 }, voiceClonePrompt: voicePath,
            duration: duration, speed: speed, mode: selectedMode) }
        var parameters: [String: SpeechValue] = [:]
        if let steps { parameters["num_step"] = .number(Double(steps)) }
        if let guidance { parameters["guidance_scale"] = .number(guidance) }
        let reply = try await Runtime.shared.speak(SpeechRequest(items: items, parameters: parameters,
                                                                 seed: seed.map(Int64.init)), model: model.id)
        guard let audio = reply.audio, let requestSeed = reply.seed else { throw DecisionError.invalidReply }
        return .result(value: audio.enumerated().map { index, clip in
            IntentFile(data: clip.wav, filename: "speech-\(requestSeed)-\(index + 1).wav", type: .wav)
        })
    }
}

struct CreateVoicePromptIntent: AppIntent {
    static let title: LocalizedStringResource = "Create Voice Prompt"
    static let description = IntentDescription(
        "Create a reusable voice file from a recording. Use it with the same model and mode in Generate Speech. The model is saved with this action."
    )
    static var supportedModes: IntentModes { .background }

    @Parameter(title: "Reference Audio", supportedContentTypes: [.audio]) var reference: IntentFile
    @Parameter(title: "Model") var model: SpeechModelEntity
    @Parameter(title: "Reference Transcript", description: "Optional words spoken in the reference audio.") var transcript: String?
    @Parameter(title: "Mode", description: "Leave empty for the model’s default cloning mode.",
               optionsProvider: SpeechVoiceModeOptions()) var mode: SpeechModeEntity?
    @Parameter(title: "Preprocess Reference", description: "Use the model’s native reference preprocessing. Leave empty for its default.") var preprocess: Bool?

    static var parameterSummary: some ParameterSummary {
        Summary("Create voice prompt from \(\.$reference) with \(\.$model)") {
            \.$transcript
            \.$mode
            \.$preprocess
        }
    }

    func perform() async throws -> some IntentResult & ReturnsValue<IntentFile> {
        let folder = FileManager.default.temporaryDirectory.appending(path: "tinkerbox-voice-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let path = folder.appending(path: "reference.\(URL(fileURLWithPath: reference.filename).pathExtension)")
        try reference.data.write(to: path)
        let prompt = try await Runtime.shared.createVoice(SpeechVoiceRequest(refAudio: path.path,
            refText: transcript.flatMap { isBlank($0) ? nil : $0 }, preprocessPrompt: preprocess,
            mode: try speechMode(mode, model: model)), model: model.id)
        return .result(value: IntentFile(data: prompt, filename: "voice.pt", type: .data))
    }
}

struct TinkerboxShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(intent: YesNoDecisionIntent(), phrases: ["Make a decision with \(.applicationName)"],
                    shortTitle: "Yes/No Decision", systemImageName: "checkmark.circle")
        AppShortcut(intent: ChooseOptionIntent(), phrases: ["Choose an option with \(.applicationName)"],
                    shortTitle: "Choose Option", systemImageName: "list.bullet")
        AppShortcut(intent: ScoreTextIntent(), phrases: ["Score text with \(.applicationName)"],
                    shortTitle: "Score Text", systemImageName: "slider.horizontal.3")
        AppShortcut(intent: AskQuestionsIntent(), phrases: ["Ask several questions with \(.applicationName)"],
                    shortTitle: "Ask Several Questions", systemImageName: "questionmark.bubble")
        AppShortcut(intent: FilterListIntent(), phrases: ["Filter a list with \(.applicationName)"],
                    shortTitle: "Filter List", systemImageName: "line.3.horizontal.decrease.circle")
        AppShortcut(intent: SortListIntent(), phrases: ["Sort a list with \(.applicationName)"],
                    shortTitle: "Sort List", systemImageName: "arrow.up.arrow.down")
        AppShortcut(intent: GenerateSpeechIntent(), phrases: ["Generate speech with \(.applicationName)"],
                    shortTitle: "Generate Speech", systemImageName: "waveform")
        AppShortcut(intent: CreateVoicePromptIntent(), phrases: ["Create a voice with \(.applicationName)"],
                    shortTitle: "Create Voice Prompt", systemImageName: "person.wave.2")
    }
}
