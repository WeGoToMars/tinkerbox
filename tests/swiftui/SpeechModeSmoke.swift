import AppIntents
import Foundation

// Picker choices remain available with no downloaded model and offline Python setup.
@main struct SpeechModeSmoke {
    static func main() async throws {
        for model in ["omnivoice", "voxcpm2"] {
            let savedModel = try await SpeechModelQuery().entities(for: [model])
            guard savedModel.count == 1 else { throw DecisionError.invalid("Missing saved speech model.") }
            let otherModel = model == "omnivoice" ? "voxcpm2" : "omnivoice"
            await Runtime.shared.select(model: otherModel, for: .speech)
            let modes = try await SpeechModeQuery().suggestedEntities().filter { $0.modelID == model }
            let titles = modes.map(\.title)
            guard titles.contains("Automatic voice"), titles.contains("Voice design"), titles.contains("Voice clone") else {
                throw DecisionError.invalid("Missing \(model) speech choices before setup.")
            }
            let selected = modes.first { $0.title == "Voice clone" }!
            let restored = try await SpeechModeQuery().entities(for: [selected.id])
            guard restored.count == 1, restored[0].title == "Voice clone", restored[0].modelID == savedModel[0].id else {
                throw DecisionError.invalid("Saved speech choice did not resolve before setup.")
            }
        }
        print("PASS: saved speech models and modes resolve offline independently of the app selection")
    }
}
