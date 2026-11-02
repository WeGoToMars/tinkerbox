import SwiftUI

struct HomeView: View {
    let openTask: (ModelTask) -> Void

    var body: some View {
        ScrollView {
            VStack(spacing: 36) {
                VStack(spacing: 14) {
                    LogoActivity().frame(width: 200, height: 190)
                    Text("Tinkerbox").font(.largeTitle.bold())
                    Text("Models on your Mac's Neural Engine")
                        .font(.title3).foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 260), spacing: 16, alignment: .top)], spacing: 16) {
                    ForEach(ModelTask.allCases, id: \.self) { task in
                        HomeTaskCard(task: task) { openTask(task) }
                    }
                }
            }
            .frame(maxWidth: 1000)
            .padding(32)
            .frame(maxWidth: .infinity)
        }
        .background(.background.secondary)
        .navigationTitle("Home")
    }
}

private struct HomeTaskCard: View {
    let task: ModelTask
    let action: () -> Void
    @State private var hovered = false

    private var color: Color {
        switch task {
        case .decision: .teal
        case .speech: .orange
        case .completion: .purple
        }
    }

    private var description: String {
        switch task {
        case .decision: "Classify text and images, compare options, and score results with clear probabilities."
        case .speech: "Turn text into speech. Design a voice or create one from a recording."
        case .completion: "Continue a prompt, explore ideas, and watch text stream into your draft."
        }
    }

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 16) {
                Image(systemName: task.symbol)
                    .font(.title).foregroundStyle(color)
                    .frame(height: 32)
                Text(task.title).font(.headline)
                Text(description)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(22)
            .frame(maxWidth: .infinity, minHeight: 180, alignment: .topLeading)
            .background(color.opacity(hovered ? 0.16 : 0.08), in: RoundedRectangle(cornerRadius: 18))
            .overlay(RoundedRectangle(cornerRadius: 18).stroke(color.opacity(hovered ? 0.5 : 0.2)))
            .contentShape(RoundedRectangle(cornerRadius: 18))
        }
        .buttonStyle(.plain)
        .onHover { hovered = $0 }
        .animation(.easeOut(duration: 0.15), value: hovered)
        .accessibilityElement(children: .combine)
    }
}
