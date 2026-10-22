import AppIntents
import AppKit
import Foundation

func solidImage(_ color: NSColor) -> IntentFile {
    let image = NSImage(size: NSSize(width: 100, height: 100), flipped: false) { rect in
        color.setFill()
        rect.fill()
        return true
    }
    let png = NSBitmapImageRep(data: image.tiffRepresentation!)!.representation(using: .png, properties: [:])!
    return IntentFile(data: png, filename: "solid.png", type: .png)
}

func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw NSError(domain: "ShortcutsSmoke", code: 1, userInfo: [NSLocalizedDescriptionKey: message]) }
}

func checkProbabilities(_ values: [DecisionProbability], labels: [String]) throws {
    try require(values.map(\.label) == labels, "Probability labels or ordering changed")
    try require(values.allSatisfy { $0.probability.isFinite && (0...1).contains($0.probability) },
                "Probabilities must be finite and between zero and one")
    try require(abs(values.reduce(0) { $0 + $1.probability } - 1) < 0.00001, "Probabilities must sum to one")
}

@main struct ShortcutsSmoke {
    static func main() async throws {
        let yes = YesNoDecisionIntent()
        yes.text = "Hello! Nice to meet you."
        yes.question = "Is this a greeting?"
        let no = YesNoDecisionIntent()
        no.text = "This invoice is overdue. Please pay the balance."
        no.question = yes.question
        let choice = ChooseOptionIntent()
        choice.text = yes.text
        choice.question = "What is the topic?"
        choice.options = ["Greeting", "Billing", "Technical support"]
        let score = ScoreTextIntent()
        score.text = yes.text
        score.question = "How friendly is this message?"
        score.levels = ["Unfriendly", "Neutral", "Friendly"]

        let invalid = ChooseOptionIntent()
        invalid.text = choice.text
        invalid.question = choice.question
        invalid.options = ["Same", "Same"]
        do {
            _ = try await invalid.perform()
            throw DecisionError.invalid("Duplicate choices were accepted")
        } catch DecisionError.invalid(let message) {
            try require(message.contains("unique"), "Expected a useful duplicate-options error")
        }
        let invalidScore = ScoreTextIntent()
        invalidScore.text = score.text
        invalidScore.question = score.question
        invalidScore.levels = Array(repeating: "Level", count: 11)
        do {
            _ = try await invalidScore.perform()
            throw DecisionError.invalid("Too many levels were accepted")
        } catch DecisionError.invalid(let message) {
            try require(message.contains("2–10"), "Expected a useful score-level error")
        }
        let blank = YesNoDecisionIntent()
        blank.text = " \n "
        blank.question = yes.question
        do {
            _ = try await blank.perform()
            throw DecisionError.invalid("Blank text was accepted")
        } catch DecisionError.invalid(let message) {
            try require(message.contains("empty"), "Expected a useful empty-text error")
        }

        // Cold startup, multiple Shortcuts, and the UI's runtime entry point overlap.
        async let yesCall = yes.perform()
        async let noCall = no.perform()
        async let choiceCall = choice.perform()
        async let scoreCall = score.perform()
        async let uiCall = Runtime.shared.decide(
            Request(items: [Item(state: yes.text!)], questions: [Question(type: .noul, instructions: yes.question)])
        )
        async let setupCall = Runtime.shared.start(model: Runtime.selectedModel(for: .decision))
        let (yesReply, noReply, choiceReply, scoreReply, uiReply, setupReply) =
            try await (yesCall, noCall, choiceCall, scoreCall, uiCall, setupCall)
        let affirmative = yesReply.value!
        let negative = noReply.value!
        let chosen = choiceReply.value!
        let scored = scoreReply.value!
        try require(setupReply.ready == true, "Concurrent UI setup did not finish")
        try require(affirmative.answer && affirmative.yesProbability > 0.9, "Expected a greeting")
        try require(!negative.answer && negative.yesProbability < 0.5, "Expected a non-greeting")
        try require(chosen.chosenOption == "Greeting", "Expected Greeting option")
        try require((0...1).contains(chosen.confidence), "Invalid choice confidence")
        try require(scored.maximumScore == 2 && (0...2).contains(scored.score), "Invalid expected score")
        try checkProbabilities(affirmative.probabilities, labels: ["Yes", "No"])
        try checkProbabilities(chosen.probabilities, labels: choice.options)
        try checkProbabilities(scored.probabilities, labels: ["0 · Unfriendly", "1 · Neutral", "2 · Friendly"])
        try require([affirmative.inputTokens, negative.inputTokens, chosen.inputTokens, scored.inputTokens].allSatisfy { $0 > 0 },
                    "Missing input-token usage")
        try require([affirmative.inferenceTime, negative.inferenceTime, chosen.inferenceTime, scored.inferenceTime]
            .allSatisfy { $0.isFinite && $0 >= 0 }, "Missing inference timing")
        try require(abs(uiReply.results[0].answers[0].noul! - affirmative.yesProbability) < 0.01, "UI and Shortcut answers differ")
        try require(String(localized: affirmative.displayRepresentation.title).hasPrefix("Yes ("),
                    "The text form should state the answer and its probability")

        // Image-only decisions: Shortcuts file data reaches the vision tower and changes the answer.
        let blue = YesNoDecisionIntent()
        blue.question = "Is this image blue?"
        blue.images = [solidImage(.blue)]
        let red = YesNoDecisionIntent()
        red.question = blue.question
        red.images = [solidImage(.red)]
        let blueAnswer = try await blue.perform().value!
        let redAnswer = try await red.perform().value!
        try require(blueAnswer.answer && blueAnswer.yesProbability > 0.9, "Expected a blue image")
        try require(!redAnswer.answer && redAnswer.yesProbability < 0.1, "Expected a non-blue image")

        // Several questions over one input keep their order and answer independently.
        let several = AskQuestionsIntent()
        several.text = "Hi! I was charged twice, please refund me."
        several.questions = ["Is this a greeting?", "Is this a refund request?", "Is this about the weather?"]
        let tags = try await several.perform().value!
        try require(tags.map(\.question) == several.questions, "Questions lost their order")
        try require(tags.map(\.answer) == [true, true, false], "Expected greeting and refund tags only")

        // List actions: one question over every item, dropping blank lines.
        let messages = ["Hello there, friend!", "", "Your invoice is overdue.", "Good morning! Hope you are well."]
        let filter = FilterListIntent()
        filter.items = messages
        filter.question = yes.question
        try require(try await filter.perform().value! == [messages[0], messages[3]], "Expected only the greetings")
        let sort = SortListIntent()
        sort.items = ["Pay now or we will sue.", "", "Thanks so much, you are wonderful!"]
        sort.question = score.question
        sort.levels = score.levels
        let ranked = try await sort.perform().value!
        try require(ranked.map(\.item) == [sort.items[2], sort.items[0]], "Expected the friendliest message first")
        try require(ranked[0].score > ranked[1].score && ranked[0].level == "Friendly", "Expected descending scores")

        let oversized = YesNoDecisionIntent()
        oversized.text = String(repeating: "hello ", count: 40_000)
        oversized.question = yes.question
        do {
            _ = try await oversized.perform()
            throw DecisionError.invalid("Oversized input was accepted")
        } catch {
            try require(error.localizedDescription.contains("context limit"), "Expected a context-limit error")
        }
        let repeated = try await yes.perform().value!
        try require(repeated.answer && abs(repeated.yesProbability - affirmative.yesProbability) < 0.01,
                    "A failed request affected the next decision")
        // Unloading preserves the selection, and the next App Intent reloads it automatically.
        let selected = Runtime.selectedModel(for: .decision)
        let settings = Runtime.settings(for: selected)
        await Runtime.shared.unload()
        try require(await !Runtime.shared.isConnected, "Manual unload kept the model loaded")
        try require(Runtime.selectedModel(for: .decision) == selected, "Unload changed the selected model")
        let afterUnload = try await yes.perform().value!
        try require(afterUnload.answer && abs(afterUnload.yesProbability - affirmative.yesProbability) < 0.01,
                    "The App Intent did not recover after unloading")
        try require(Runtime.settings(for: selected) == settings, "Reload changed the saved settings")
        await Runtime.shared.unload()

        // A short idle policy exercises the same lifecycle without waiting five minutes.
        let idle = Runtime(idleTimeout: .seconds(2))
        let request = Request(items: [Item(state: yes.text!)], questions: [Question(type: .noul, instructions: yes.question)])
        _ = try await idle.decide(request)
        try await Task.sleep(for: .milliseconds(1200))
        _ = try await idle.decide(request)
        try await Task.sleep(for: .milliseconds(1200))
        try require(await idle.isConnected, "New inference did not renew the idle timeout")
        let deadline = ContinuousClock.now.advanced(by: .seconds(8))
        var expired = false
        while ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(200))
            do { _ = try await idle.count(request) }
            catch DecisionError.invalid { expired = true; break }
        }
        let disconnected = await !idle.isConnected
        try require(expired && disconnected, "Token previews kept an idle model loaded")
        let reloaded = try await idle.decide(request)
        try require(reloaded.results[0].answers[0].noul! > 0.9, "Inference did not reload the idle model")

        // Unload waits for an active request rather than cancelling it.
        let long = Request(items: [Item(state: String(repeating: "Hello, friend! ", count: 1200))], questions: request.questions)
        async let active = idle.decide(long)
        try await Task.sleep(for: .milliseconds(100))
        await idle.unload()
        let completed = try await active
        try require(completed.results[0].answers.count == 1, "Unload interrupted inference")
        try require(await !idle.isConnected, "Queued unload did not release the model")
        print("PASS: typed answers, probabilities, usage, images, several questions, list filter and sort, cold startup, concurrent UI/Shortcuts calls, validation, error recovery, manual/idle unload, preview inactivity, automatic reload and queued unload")
    }
}
