import Foundation

actor CompletionCapture {
    var text: [String]
    var tokens: [Int]
    var deliveries = 0
    var firstDelivery = 0
    init(count: Int) {
        text = Array(repeating: "", count: count)
        tokens = Array(repeating: 0, count: count)
    }
    func receive(_ deltas: [CompletionDelta]) {
        for delta in deltas {
            text[delta.index] += delta.text
            tokens[delta.index] += delta.tokens
        }
        if deltas.contains(where: { !$0.text.isEmpty }) {
            deliveries += 1
            if deliveries == 1 { firstDelivery = text.reduce(0) { $0 + $1.count } }
        }
    }
    func snapshot() -> ([String], Int) { (text, deliveries) }
    func tokenCounts() -> [Int] { tokens }
    func firstTextCount() -> Int { firstDelivery }
}

@main struct CompletionSmoke {
    static func main() async throws {
        let runtime = Runtime.shared
        let model = ProcessInfo.processInfo.environment["TINKERBOX_COMPLETION_TEST_MODEL"] ?? "gpt2"
        precondition(Catalog.current(model)?.kind == .completion)
        await runtime.select(model: model, for: .completion)
        let ready = try await runtime.start(model: model)
        precondition(Catalog.current(model)?.context == 1024 && ready.batchSizes == [1, 4, 8])
        let prompts = ["The capital of France is", "Once upon a time,", "The capital of France is"]
        let request = CompletionRequest(prompt: prompts, maxTokens: 4, temperature: 0, batchSize: 4)
        let counts = try await runtime.count(request)
        precondition(counts.map(\.tokens) == [5, 5, 5])
        let spans = try await runtime.tokenSpans("Hello world!", model: model)
        precondition(spans.map(\.startUtf16) == [0, 5, 11] && spans.map(\.endUtf16) == [5, 11, 12])
        let small = ["gpt2", "gpt2-lut8"].contains(model)  // greedy ids pinned for GPT-2 Small
        precondition(spans.flatMap(\.tokenIds) == [15496, 995, 0])
        let unicode = try await runtime.tokenSpans("Hi 👋🏽!", model: model)
        precondition(unicode.last!.endUtf16 == "Hi 👋🏽!".utf16.count)
        let noText = try await runtime.tokenSpans("", model: model)
        precondition(noText == [])
        let first = try await runtime.complete(request)
        precondition(first.choices.count == 3 && first.metrics.elapsedMs > 0)
        precondition(!small || first.choices[0].tokenIds == [262, 3139, 286, 262])
        precondition(first.choices[2].tokenIds == first.choices[0].tokenIds)
        precondition(
            first.choices.allSatisfy {
                !$0.text.isEmpty && $0.metrics.inputTokens == 5 && $0.metrics.outputTokens == $0.tokenIds.count
                    && $0.metrics.batchSize == 4 && $0.finishReason == "length"
            })
        let capture = CompletionCapture(count: prompts.count)
        let streamed = try await runtime.complete(request) { deltas in await capture.receive(deltas) }
        let (text, deliveries) = await capture.snapshot()
        precondition(deliveries > 0 && text == first.choices.map(\.text))
        let streamedTokens = await capture.tokenCounts()
        precondition(streamedTokens == first.choices.map(\.metrics.outputTokens))
        precondition(streamed.choices.map(\.tokenIds) == first.choices.map(\.tokenIds))
        precondition(streamed.choices.allSatisfy { $0.metrics.batchSize == 4 })
        var longer = request
        longer.maxTokens = 128
        let live = CompletionCapture(count: prompts.count)
        let finished = try await runtime.complete(longer) { deltas in await live.receive(deltas) }
        let (liveText, updates) = await live.snapshot()
        let firstText = await live.firstTextCount()
        precondition(updates > 1 && firstText > 0 && firstText < liveText.reduce(0) { $0 + $1.count })
        precondition(liveText == finished.choices.map(\.text))
        // A token whose text arrives with a later one, or never, is not counted.
        let liveTokens = await live.tokenCounts()
        precondition(zip(liveTokens, finished.choices).allSatisfy { $0 > 0 && $0 <= $1.metrics.outputTokens })
        precondition(finished.choices.allSatisfy { $0.metrics.outputTokens == $0.tokenIds.count })
        let sampled = CompletionRequest(
            prompt: prompts, maxTokens: 12,
            temperature: 0.8, topK: 40, topP: 0.95, seed: 42)
        let replay = try await runtime.complete(sampled)
        for (index, prompt) in prompts.enumerated() {
            let alone = try await runtime.complete(
                CompletionRequest(
                    prompt: [prompt], maxTokens: 12,
                    temperature: 0.8, topK: 40, topP: 0.95, seed: 42))
            precondition(alone.choices[0].tokenIds == replay.choices[index].tokenIds)
        }
        let empty = try await runtime.complete(CompletionRequest(prompt: [""], maxTokens: 0))
        precondition(
            empty.choices[0].text.isEmpty && empty.choices[0].metrics.inputTokens == 1
                && empty.choices[0].tokenIds.isEmpty)
        do {
            _ = try await runtime.complete(
                CompletionRequest(
                    prompt: [prompts[0], String(repeating: " hello", count: 1024)], maxTokens: 1))
            preconditionFailure("oversized batch accepted")
        } catch { precondition(error.localizedDescription.contains("context limit")) }
        let concurrent = CompletionCapture(count: prompts.count)
        async let a = runtime.complete(request, onText: { deltas in await concurrent.receive(deltas) })
        async let b = runtime.complete(CompletionRequest(prompt: [prompts[1]], maxTokens: 4, temperature: 0))
        let (batch, single) = try await (a, b)
        precondition(batch.choices.map(\.tokenIds) == first.choices.map(\.tokenIds))
        precondition(single.choices[0].tokenIds == first.choices[1].tokenIds)
        let (isolatedText, _) = await concurrent.snapshot()
        precondition(isolatedText == first.choices.map(\.text))
        await runtime.unload()
        do {
            _ = try await runtime.count(request)
            preconditionFailure("token preview reloaded an unloaded model")
        } catch { precondition(error.localizedDescription.contains("Load the model")) }
        let reloaded = try await runtime.complete(request)
        precondition(reloaded.choices.map(\.tokenIds) == first.choices.map(\.tokenIds))
        await runtime.unload()
        print(
            "PASS: completion worker, streaming, prompt ordering, usage, token preview, seeded replay, BOS, batch limits, error recovery, concurrent request isolation and reload"
        )
    }
}
