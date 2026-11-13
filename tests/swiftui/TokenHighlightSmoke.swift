import Foundation

@main struct TokenHighlightSmoke {
    static func main() async throws {
        let runtime = Runtime.shared
        let environment = ProcessInfo.processInfo.environment
        let models = (environment["TINKERBOX_TOKEN_TEST_MODELS"] ?? "gpt2,d1-3b,d3-nano,d3-mini,omnivoice,voxcpm2")
            .split(separator: ",").map(String.init)
        let assets =
            environment["TINKERBOX_TOKEN_TEST_ASSETS"].flatMap {
                try? JSONDecoder().decode([String: String].self, from: Data($0.utf8))
            } ?? [:]
        let unicode = "Hello world! 👋🏽 日本語 cafe\u{301}"
        var vocabularies = [[Int]]()
        for model in models {
            FileHandle.standardOutput.write(Data("START: \(model)\n".utf8))
            let task = Catalog.current(model)!.kind!
            await runtime.select(model: model, for: task, asset: assets[model])
            _ = try await runtime.start(model: model)
            let first = try await runtime.tokenSpans("Hello world!", model: model)
            precondition(!first.isEmpty && first.first!.startUtf16 == 0 && first.last!.endUtf16 == 12)
            vocabularies.append(first.flatMap(\.tokenIds))
            if task == .completion { precondition(first.flatMap(\.tokenIds) == [15496, 995, 0]) }
            async let a = runtime.tokenSpans("Hello world!", model: model)
            async let b = runtime.tokenSpans(unicode, model: model)
            let (replay, spans) = try await (a, b)
            precondition(replay == first && !spans.isEmpty && spans.last!.endUtf16 == unicode.utf16.count)
            var end = 0
            for span in spans {
                precondition(span.startUtf16 >= end && span.endUtf16 > span.startUtf16 && !span.tokenIds.isEmpty)
                precondition(
                    Range(NSRange(location: span.startUtf16, length: span.endUtf16 - span.startUtf16), in: unicode)
                        != nil)
                end = span.endUtf16
            }
            let empty = try await runtime.tokenSpans("", model: model)
            precondition(empty.isEmpty)
            do {
                _ = try await runtime.tokenSpans("Hello", model: "another-model")
                preconditionFailure("tokenizer from a different model was accepted")
            } catch { precondition(error.localizedDescription.contains("Load the model")) }
            switch task {
            case .completion:
                let reply = try await runtime.complete(
                    CompletionRequest(prompt: ["The capital of France is"], maxTokens: 4, temperature: 0))
                precondition(
                    !["gpt2", "gpt2-lut8"].contains(model) || reply.choices[0].tokenIds == [262, 3139, 286, 262])
            case .decision:
                let reply = try await runtime.decide(
                    Request(
                        items: [Item(state: "Hello! Welcome.")],
                        questions: [Question(type: .noul, instructions: "Is this a greeting?")]))
                precondition(reply.results[0].answers[0].noul! > 0.5)
            case .speech:
                let capabilities = Catalog.speechCapabilities[model]!
                let controls: [String: SpeechValue]? =
                    capabilities.controls.contains { $0.key == "duration" } ? ["duration": .number(0.5)] : nil
                var parameters: [String: SpeechValue] = ["num_step": .number(4)]
                if capabilities.parameters.contains(where: { $0.key == "max_decode_steps" }) {
                    parameters["max_decode_steps"] = .number(64)
                }
                let reply = try await runtime.speak(
                    SpeechRequest(
                        items: [SpeechItem(text: "Hello.", mode: "auto", controls: controls)], parameters: parameters,
                        seed: 42), model: model)
                precondition(reply.audio[0].duration > 0)
            }
            let after = try await runtime.tokenSpans("Hello world!", model: model)
            precondition(after == first)
            await runtime.unload()
            do {
                _ = try await runtime.tokenSpans("Hello", model: model)
                preconditionFailure("highlighting reloaded an unloaded model")
            } catch { precondition(error.localizedDescription.contains("Load the model")) }
            FileHandle.standardOutput.write(
                Data("PASS: \(model) token spans, Unicode, isolation, task execution, and unload\n".utf8))
        }
        if vocabularies.count > 1 { precondition(vocabularies.contains { $0 != vocabularies[0] }) }
    }
}
