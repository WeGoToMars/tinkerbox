import Foundation

// What a model does. Each task has its own page in the app and its own selected model.
enum ModelTask: String, CaseIterable, Sendable {
    case decision

    var title: String {
        switch self {
        case .decision: "Decision (System One)"
        }
    }

    var symbol: String {
        switch self {
        case .decision: "arrow.triangle.branch"
        }
    }

    // Its section in the Models library.
    var heading: String {
        switch self {
        case .decision: "Decision"
        }
    }
}

// One entry of tinkerbox_ane/models.json.
struct CatalogModel: Decodable, Identifiable, Hashable, Sendable {
    let id: String
    let name: String
    let publisher: String
    let summary: String
    let task: String  // a ModelTask; the remote catalog may list tasks newer than this app
    // Hugging Face repositories: the foundation it was fine-tuned from, the model that was
    // converted, and the converted Core AI bundles.
    let baseModel: String?
    let originalModel: String?
    let repo: String
    let inputs: [String]
    let context: Int
    let downloadBytes: Int64
    let license: String
    let minVersion: String

    var kind: ModelTask? { ModelTask(rawValue: task) }
    var acceptsImages: Bool { inputs.contains("image") }
    var isSupported: Bool { kind != nil && !Catalog.isNewer(minVersion, than: Catalog.appVersion) }
}

@MainActor @Observable final class Catalog {
    nonisolated static let remoteURL = URL(string: "https://raw.githubusercontent.com/WeGoToMars/tinkerbox/main/src/tinkerbox_ane/models.json")!
    nonisolated static let appVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0"
    // Matches tinkerbox_ane.bundle.CACHE: <model>/<revision>/macos27/<architecture>/model.json.
    nonisolated static let cacheURL = FileManager.default.homeDirectoryForCurrentUser
        .appending(path: "Library/Caches/tinkerbox-ane/models")

    // The bundled catalog lists exactly the models this app's Python package can run.
    nonisolated static let bundled: [CatalogModel] = {
        guard let url = Bundle.main.url(forResource: "models", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return [] }
        return (try? decode(data)) ?? []
    }()

    var models = Catalog.bundled
    var isRemote = false
    var downloaded: Set<String> = []

    nonisolated static func decode(_ data: Data) throws -> [CatalogModel] {
        struct File: Decodable { let models: [CatalogModel] }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(File.self, from: data).models
    }

    nonisolated static func isNewer(_ version: String, than other: String) -> Bool {
        let parts = { (text: String) in text.split(separator: ".").map { Int($0) ?? 0 } }
        let (a, b) = (parts(version), parts(other))
        for index in 0..<max(a.count, b.count) {
            let (x, y) = (index < a.count ? a[index] : 0, index < b.count ? b[index] : 0)
            if x != y { return x > y }
        }
        return false
    }

    func refresh() async {
        scanDownloads()
        do {
            let (data, response) = try await URLSession.shared.data(from: Self.remoteURL)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
            let remote = try Self.decode(data)
            // Keep bundled models the remote catalog no longer lists; this app can still run them.
            models = remote + Self.bundled.filter { model in !remote.contains { $0.id == model.id } }
            isRemote = true
        } catch {
            models = Self.bundled
            isRemote = false
        }
        scanDownloads()
    }

    func scanDownloads() {
        let files = FileManager.default
        downloaded = Set(models.map(\.id).filter { id in
            let revisions = (try? files.contentsOfDirectory(at: Self.cacheURL.appending(path: id),
                                                            includingPropertiesForKeys: nil)) ?? []
            return revisions.contains { revision in
                let bundles = (try? files.contentsOfDirectory(at: revision.appending(path: "macos27"),
                                                              includingPropertiesForKeys: nil)) ?? []
                return bundles.contains { files.fileExists(atPath: $0.appending(path: "model.json").path) }
            }
        })
    }

    func removeDownload(_ model: CatalogModel) throws {
        try FileManager.default.removeItem(at: Self.cacheURL.appending(path: model.id))
        scanDownloads()
    }
}
