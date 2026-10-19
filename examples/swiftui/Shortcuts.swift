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
        "Ask a yes/no question about text, images, or both, using the model selected in Tinkerbox. As text, the result reads like “Yes (93%)”. Its properties are the Boolean Answer, the Yes Probability (0–1), all probabilities, input tokens, and inference time. First run may download the runtime and model; open Tinkerbox once to finish setup."
    )
    static var supportedModes: IntentModes { .background }

    @Parameter(title: "Input Text") var text: String?
    @Parameter(title: "Question", description: "For example: Is this a refund request?") var question: String
    @Parameter(title: "Images", description: "Optional pictures to decide about, if the model selected in Tinkerbox accepts images.",
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
        "Choose among at least two unique options for text, images, or both, using the model selected in Tinkerbox. Supply a List of option names. As text, the result reads like “Refund request (81%)”, or “Unsure: …” below the Minimum Confidence, when Chosen Option has no value. Its properties are the Chosen Option, the Most Likely Option, the Confidence (0–1), all probabilities, input tokens, and inference time. First run may download the runtime and model; open Tinkerbox once to finish setup."
    )
    static var supportedModes: IntentModes { .background }

    @Parameter(title: "Input Text") var text: String?
    @Parameter(title: "Question", description: "For example: What is the topic of this message?") var question: String
    @Parameter(title: "Options", description: "A list of at least two unique, nonempty option names.") var options: [String]
    @Parameter(title: "Images", description: "Optional pictures to decide about, if the model selected in Tinkerbox accepts images.",
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
        "Evaluate text, images, or both across 2–10 ordered levels, lowest to highest, using the model selected in Tinkerbox. Supply a List of levels. As text, the result reads like “Friendly (1.8 / 2)”. Its properties are the expected Score from 0 to the number of levels minus one (may be fractional), the Most Likely Level, all probabilities, input tokens, and inference time. First run may download the runtime and model; open Tinkerbox once to finish setup."
    )
    static var supportedModes: IntentModes { .background }

    @Parameter(title: "Input Text") var text: String?
    @Parameter(title: "Question", description: "For example: How urgent is this request?") var question: String
    @Parameter(title: "Levels", description: "A list of 2–10 nonempty levels, ordered from lowest to highest.") var levels: [String]
    @Parameter(title: "Images", description: "Optional pictures to decide about, if the model selected in Tinkerbox accepts images.",
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
        "Ask several yes/no questions about the same text, images, or both, using the model selected in Tinkerbox. The model reads the input once and answers every question in one pass, so extra questions cost little. Returns one Yes/No Decision per question, in order. Each answer is independent, so several can be Yes: use it to tag an input with every label that applies."
    )
    static var supportedModes: IntentModes { .background }

    @Parameter(title: "Input Text") var text: String?
    @Parameter(title: "Questions", description: "A list of nonempty yes/no questions.") var questions: [String]
    @Parameter(title: "Images", description: "Optional pictures to decide about, if the model selected in Tinkerbox accepts images.",
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
        "Keep the items of a list for which the model selected in Tinkerbox answers Yes, in their original order. All items are evaluated together in one batch; blank items are dropped."
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
        "Score every item of a list across 2–10 ordered levels with the model selected in Tinkerbox, and sort by the expected score. Each ranked item reads as its original text and has Score, Maximum Score, and Most Likely Level properties. All items are evaluated together in one batch; blank items are dropped."
    )
    static var supportedModes: IntentModes { .background }

    @Parameter(title: "List", description: "Text items to rank, for example from Split Text.") var items: [String]
    @Parameter(title: "Question", description: "Asked about each item. For example: How urgent is this request?") var question: String
    @Parameter(title: "Levels", description: "A list of 2–10 nonempty levels, ordered from lowest to highest.") var levels: [String]
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
    }
}
