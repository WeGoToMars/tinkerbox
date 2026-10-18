import AppIntents
import Foundation

struct DecisionProbability: TransientAppEntity {
    static let typeDisplayRepresentation = TypeDisplayRepresentation(name: "Decision Probability")
    @Property(title: "Label") var label: String
    @Property(title: "Probability") var probability: Double

    var displayRepresentation: DisplayRepresentation {
        DisplayRepresentation(title: "\(label)", subtitle: "\(probability)")
    }

    init() {}
    init(_ value: Probability) {
        label = value.label
        probability = value.value
    }
}

struct YesNoDecision: TransientAppEntity {
    static let typeDisplayRepresentation = TypeDisplayRepresentation(name: "Yes/No Decision")
    @Property(title: "Answer") var answer: Bool
    @Property(title: "Yes Probability") var yesProbability: Double
    @Property(title: "Probabilities") var probabilities: [DecisionProbability]
    @Property(title: "Input Tokens") var inputTokens: Int
    @Property(title: "Inference Time (ms)") var inferenceTime: Double

    var displayRepresentation: DisplayRepresentation {
        DisplayRepresentation(title: "\(answer ? "Yes" : "No")")
    }

    init() {}
    init(_ reply: Reply) throws {
        guard reply.answer?.type == .noul, let value = reply.answer?.noul,
              let probabilities = reply.probabilities, let tokens = reply.tokens,
              let elapsed = reply.elapsedMs else { throw DecisionError.invalidReply }
        answer = value >= 0.5
        yesProbability = value
        self.probabilities = probabilities.map(DecisionProbability.init)
        inputTokens = tokens
        inferenceTime = elapsed
    }
}

struct ChoiceDecision: TransientAppEntity {
    static let typeDisplayRepresentation = TypeDisplayRepresentation(name: "Choice Decision")
    @Property(title: "Chosen Option") var chosenOption: String
    @Property(title: "Confidence") var confidence: Double
    @Property(title: "Probabilities") var probabilities: [DecisionProbability]
    @Property(title: "Input Tokens") var inputTokens: Int
    @Property(title: "Inference Time (ms)") var inferenceTime: Double

    var displayRepresentation: DisplayRepresentation {
        DisplayRepresentation(title: "\(chosenOption)")
    }

    init() {}
    init(_ reply: Reply) throws {
        guard reply.answer?.type == .choice, let value = reply.answer?.choice,
              let confidence = reply.answer?.confidence, let probabilities = reply.probabilities,
              let tokens = reply.tokens, let elapsed = reply.elapsedMs else { throw DecisionError.invalidReply }
        chosenOption = value
        self.confidence = confidence
        self.probabilities = probabilities.map(DecisionProbability.init)
        inputTokens = tokens
        inferenceTime = elapsed
    }
}

struct ScoreDecision: TransientAppEntity {
    static let typeDisplayRepresentation = TypeDisplayRepresentation(name: "Score Decision")
    @Property(title: "Score") var score: Double
    @Property(title: "Maximum Score") var maximumScore: Int
    @Property(title: "Probabilities") var probabilities: [DecisionProbability]
    @Property(title: "Input Tokens") var inputTokens: Int
    @Property(title: "Inference Time (ms)") var inferenceTime: Double

    var displayRepresentation: DisplayRepresentation {
        DisplayRepresentation(title: "\(score) / \(maximumScore)")
    }

    init() {}
    init(_ reply: Reply) throws {
        guard reply.answer?.type == .score, let value = reply.answer?.score,
              let probabilities = reply.probabilities, let tokens = reply.tokens,
              let elapsed = reply.elapsedMs else { throw DecisionError.invalidReply }
        score = value
        maximumScore = probabilities.count - 1
        self.probabilities = probabilities.map(DecisionProbability.init)
        inputTokens = tokens
        inferenceTime = elapsed
    }
}

struct YesNoDecisionIntent: AppIntent {
    static let title: LocalizedStringResource = "Yes/No Decision"
    static let description = IntentDescription(
        "Ask a yes/no question about text using the on-device decision model. Returns a Boolean answer, the probability of Yes (0–1), all probabilities, input tokens, and inference time. First run may download the runtime and model; open Tinkerbox once to finish setup."
    )
    static var supportedModes: IntentModes { .background }

    @Parameter(title: "Input Text") var text: String
    @Parameter(title: "Question", description: "For example: Is this a refund request?") var question: String

    static var parameterSummary: some ParameterSummary {
        Summary("Decide \(\.$question) about \(\.$text)")
    }

    func perform() async throws -> some IntentResult & ReturnsValue<YesNoDecision> {
        let reply = try await Runtime.shared.decide(
            Request(state: text, type: .noul, instructions: question, criteria: [])
        )
        return .result(value: try YesNoDecision(reply))
    }
}

struct ChooseOptionIntent: AppIntent {
    static let title: LocalizedStringResource = "Choose Option"
    static let description = IntentDescription(
        "Choose among at least two unique options using the on-device decision model. Supply a List of option names. Returns the chosen option, confidence (0–1), all probabilities, input tokens, and inference time. First run may download the runtime and model; open Tinkerbox once to finish setup."
    )
    static var supportedModes: IntentModes { .background }

    @Parameter(title: "Input Text") var text: String
    @Parameter(title: "Question", description: "For example: What is the topic of this message?") var question: String
    @Parameter(title: "Options", description: "A list of at least two unique, nonempty option names.") var options: [String]

    static var parameterSummary: some ParameterSummary {
        Summary("Choose from \(\.$options) for \(\.$text)") { \.$question }
    }

    func perform() async throws -> some IntentResult & ReturnsValue<ChoiceDecision> {
        let reply = try await Runtime.shared.decide(
            Request(state: text, type: .choice, instructions: question, criteria: options)
        )
        return .result(value: try ChoiceDecision(reply))
    }
}

struct ScoreTextIntent: AppIntent {
    static let title: LocalizedStringResource = "Score Text"
    static let description = IntentDescription(
        "Evaluate text across 2–10 ordered levels, lowest to highest, using the on-device decision model. Supply a List of levels. Returns the expected score from 0 to the number of levels minus one (may be fractional), all probabilities, input tokens, and inference time. First run may download the runtime and model; open Tinkerbox once to finish setup."
    )
    static var supportedModes: IntentModes { .background }

    @Parameter(title: "Input Text") var text: String
    @Parameter(title: "Question", description: "For example: How urgent is this request?") var question: String
    @Parameter(title: "Levels", description: "A list of 2–10 nonempty levels, ordered from lowest to highest.") var levels: [String]

    static var parameterSummary: some ParameterSummary {
        Summary("Score \(\.$text) using \(\.$levels)") { \.$question }
    }

    func perform() async throws -> some IntentResult & ReturnsValue<ScoreDecision> {
        let reply = try await Runtime.shared.decide(
            Request(state: text, type: .score, instructions: question, criteria: levels)
        )
        return .result(value: try ScoreDecision(reply))
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
    }
}
