import SwiftUI

// The Tinkerbox logo, drawn from the geometry in Tinkerbox.icon/Assets (a 1085×1015 view box)
// and set on a dark stage like the icon's.
enum LogoGeometry {
    static let inset: CGFloat = 150
    static let stage = CGSize(width: 1085 + inset * 2, height: 1015 + inset * 2)
    static let nodeRadius: CGFloat = 78.75
    static let ringWidth: CGFloat = 29.17
    static let edgeWidth: CGFloat = 21
    static let lensRadius = nodeRadius + 30
    static let tubeWidth: CGFloat = 46
    static let nodes: [CGPoint] = [
        CGPoint(x: 358.75, y: 936.25), CGPoint(x: 883.75, y: 825.42), CGPoint(x: 78.75, y: 627.08),
        CGPoint(x: 545.42, y: 504.58), CGPoint(x: 1006.25, y: 358.75), CGPoint(x: 189.58, y: 177.92),
        CGPoint(x: 650.42, y: 78.75),
    ].map { CGPoint(x: $0.x + inset, y: $0.y + inset) }
    static let cyan = Color(red: 0, green: 0.753, blue: 0.910)
    static let teal = Color(red: 0, green: 0.765, blue: 0.816)
    // The icon's three edge layers, back to front (Short, Long 1, Long 2), with their gradients
    // from icon.json as unit points of the view box. The nodes sit in front of all of them.
    static let edgeLayers: [(edges: [(Int, Int)], colors: [Color], start: UnitPoint, end: UnitPoint)] = [
        ([(1, 0), (3, 0), (4, 1), (3, 2), (5, 2), (4, 3), (6, 3), (6, 5)], [cyan, teal.opacity(0.74)],
         UnitPoint(x: 0.070, y: 0), UnitPoint(x: 0.911, y: 0.935)),
        ([(1, 2), (4, 5)], [teal, teal.opacity(0.73)], UnitPoint(x: 1, y: 0.661), UnitPoint(x: 0, y: 0.333)),
        ([(0, 5), (1, 6)], [cyan.opacity(0.6), cyan], UnitPoint(x: 0.344, y: 0), UnitPoint(x: 0.601, y: 1)),
    ]
    // Dims the edges against the dark stage; the icon's colors at full strength glare there.
    static let edgeBrightness = 0.75
    // Closed walks along the edges, so every loop ends where it began; none doubles straight back.
    static let routes = [
        [6, 3, 0, 1, 4, 5, 2, 3, 6], [2, 1, 6, 5, 0, 3, 4, 1, 2], [4, 3, 2, 5, 6, 1, 0, 5, 4],
        [0, 3, 6, 1, 2, 5, 4, 3, 0], [5, 6, 3, 2, 1, 4, 3, 0, 5], [1, 0, 3, 4, 5, 6, 3, 2, 1],
    ]
    // A particle darts along an edge, sinks into the node it reached, and rests there before the node
    // charges up and fires it on. Rests vary, and each route starts at a different one, so the network
    // never ticks in step.
    static let dart: TimeInterval = 0.18
    static let rests: [TimeInterval] = [1.6, 2.4, 1.1, 2.9, 1.9, 1.3, 2.6, 2.0]
    static let charge: TimeInterval = 0.45
    static let loop = dart * Double(rests.count) + rests.reduce(0, +)
    // Each particle activates the node it reaches on this hop of its route; every route rests long
    // enough there for the spin.
    static let activatingHop = 3
    // Only the first few routes spin nodes up, so activations stay occasional.
    static let spinningRoutes = 3
    static let spin: TimeInterval = 1.1

    struct Particle: Equatable { var layer: Int; var position: CGPoint; var tail: CGPoint }
    // A droplet that rests inside a node's lens and spirals out under the ring when the node spins up.
    // `size` is 0 at rest; the droplet always exists, since adding glass to a container flashes all of it.
    struct Spinner { var position: CGPoint; var size: Double }

    struct State {
        var particles: [Particle] = []
        var spinners: [Spinner] = []
        var glow = [Double](repeating: 0, count: nodes.count)
    }

    static func state(at phase: Double) -> State {
        var state = State()
        for (index, route) in routes.enumerated() {
            let rest = (0..<rests.count).map { rests[($0 + index * 3) % rests.count] }
            let time = ((phase + Double(index) / Double(routes.count)) * loop).truncatingRemainder(dividingBy: loop)
            var local = time, step = 0
            while local >= dart + rest[step] {
                local -= dart + rest[step]
                step += 1
            }
            let from = route[step], to = route[step + 1]
            if local < dart {
                // Smootherstep: slow out of the node, fastest mid-edge, slow into the next node.
                let progress = local / dart, eased = progress * progress * progress * (progress * (progress * 6 - 15) + 10)
                let (start, end) = segment(from, to)
                func point(_ t: Double) -> CGPoint {
                    CGPoint(x: start.x + (end.x - start.x) * t, y: start.y + (end.y - start.y) * t)
                }
                let layer = edgeLayers.firstIndex { $0.edges.contains { $0 == (from, to) || $0 == (to, from) } }!
                state.particles.append(Particle(layer: layer, position: point(eased), tail: point(max(0, eased - 0.12))))
                state.glow[from] += 1 - progress
            } else {
                let resting = local - dart, remaining = rest[step] - resting
                state.glow[to] += exp(-resting / 0.5)
                if remaining < charge { state.glow[to] += pow(1 - remaining / charge, 2) }
            }

            guard index < spinningRoutes else { continue }
            let node = route[activatingHop + 1]
            let arrival = (0...activatingHop).reduce(0) { $0 + dart + ($1 < activatingHop ? rest[$1] : 0) }
            let t = min(1, (time - arrival + loop).truncatingRemainder(dividingBy: loop) / spin)
            // Accelerate into one and a half turns under the ring, then sink back to the centre.
            let angle = -Double.pi / 2 + t * t * 3 * .pi
            let size = sin(t * .pi), radius = (nodeRadius - ringWidth / 2) * min(1, size * 2)
            state.spinners.append(Spinner(
                position: CGPoint(x: nodes[node].x + cos(angle) * radius, y: nodes[node].y + sin(angle) * radius),
                size: size
            ))
            state.glow[node] += 0.6 * size
        }
        return state
    }

    // An edge cut back from both node centres, by default to the middle of the rings, so nothing
    // shows inside a ring's hole.
    static func segment(_ a: Int, _ b: Int, trim: CGFloat = nodeRadius - ringWidth / 2) -> (CGPoint, CGPoint) {
        let from = nodes[a], to = nodes[b]
        let length = hypot(to.x - from.x, to.y - from.y)
        let dx = (to.x - from.x) / length * trim, dy = (to.y - from.y) / length * trim
        return (CGPoint(x: from.x + dx, y: from.y + dy), CGPoint(x: to.x - dx, y: to.y - dy))
    }

    static func circle(_ center: CGPoint, _ radius: CGFloat) -> Path {
        Path(ellipseIn: CGRect(x: center.x - radius, y: center.y - radius, width: radius * 2, height: radius * 2))
    }

    static func tile(_ context: inout GraphicsContext, phase: Double) {
        let tile = Path(roundedRect: CGRect(origin: .zero, size: stage), cornerRadius: stage.width * 0.22, style: .continuous)
        context.clip(to: tile)
        context.fill(tile, with: .linearGradient(Gradient(colors: [Color(white: 0.11), Color(white: 0.04)]),
                                                 startPoint: .zero, endPoint: CGPoint(x: 0, y: stage.height)))
        // A soft teal light behind the network, breathing four times a loop, gives the glass something to bend.
        var glow = context
        glow.addFilter(.blur(radius: 140))
        let breath = (1 - cos(phase * 4 * 2 * .pi)) / 2
        glow.fill(circle(CGPoint(x: stage.width / 2, y: stage.height / 2), 330 + 70 * breath),
                  with: .color(Color(red: 0.05, green: 0.47, blue: 0.54).opacity(0.45 + 0.3 * breath)))
    }

    // One edge layer under its glass: the edge lines over a wide glow of their own color, and the
    // particles running along them.
    static func edgeLayer(_ context: inout GraphicsContext, _ index: Int, particles: [Particle]) {
        let layer = edgeLayers[index]
        var cores = Path()
        for (a, b) in layer.edges {
            let (from, to) = segment(a, b)
            cores.move(to: from)
            cores.addLine(to: to)
        }
        func point(_ unit: UnitPoint) -> CGPoint {
            CGPoint(x: inset + unit.x * (stage.width - inset * 2), y: inset + unit.y * (stage.height - inset * 2))
        }
        let shading = GraphicsContext.Shading.linearGradient(
            Gradient(colors: layer.colors), startPoint: point(layer.start), endPoint: point(layer.end))
        var under = context
        under.addFilter(.blur(radius: 18))
        under.opacity = edgeBrightness * 0.9
        under.stroke(cores, with: shading, lineWidth: tubeWidth * 1.4)
        var lines = context
        lines.opacity = edgeBrightness
        lines.stroke(cores, with: shading, lineWidth: edgeWidth)

        context.blendMode = .plusLighter
        var bloom = context
        bloom.addFilter(.blur(radius: 24))
        for particle in particles {
            var streak = Path()
            streak.move(to: particle.tail)
            streak.addLine(to: particle.position)
            bloom.stroke(streak, with: .color(.white.opacity(0.5)), style: StrokeStyle(lineWidth: 24, lineCap: .round))
            bloom.fill(circle(particle.position, 34), with: .color(.white))
            context.stroke(streak, with: .linearGradient(
                Gradient(colors: [.white.opacity(0), .white]), startPoint: particle.tail, endPoint: particle.position
            ), style: StrokeStyle(lineWidth: edgeWidth, lineCap: .round))
            context.fill(circle(particle.position, 16), with: .color(.white))
        }
    }

    // The light under the node lenses: flares as particles arrive, and the spinners.
    static func nodeLight(_ context: inout GraphicsContext, state: State) {
        context.blendMode = .plusLighter
        var bloom = context
        bloom.addFilter(.blur(radius: 24))
        for (node, amount) in zip(nodes, state.glow) where amount > 0.01 {
            bloom.fill(circle(node, lensRadius), with: .color(cyan.opacity(min(1, amount))))
        }
        for spinner in state.spinners where spinner.size > 0.01 {
            bloom.fill(circle(spinner.position, 48), with: .color(.white.opacity(spinner.size)))
            context.fill(circle(spinner.position, 26), with: .color(.white.opacity(spinner.size)))
        }
    }
}

// A straight glass tube between two points, in the coordinates of the view it shapes.
struct Tube: Shape {
    var from: CGPoint
    var to: CGPoint
    var width: CGFloat

    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: from)
        path.addLine(to: to)
        return path.strokedPath(StrokeStyle(lineWidth: width, lineCap: .round))
    }
}

// The logo in layered liquid glass: particles dart through the glass tubes between nodes, and now
// and then a node spins up a droplet that runs around its lens. Holds still under Reduce Motion.
struct LogoActivity: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        let stage = LogoGeometry.stage
        GeometryReader { proxy in
            let scale = min(proxy.size.width / stage.width, proxy.size.height / stage.height)
            if reduceMotion {
                LogoScene(scale: scale, phase: nil)
            } else {
                Color.clear.keyframeAnimator(initialValue: 0.0, repeating: true) { _, phase in
                    LogoScene(scale: scale, phase: phase)
                } keyframes: { _ in
                    LinearKeyframe(1.0, duration: LogoGeometry.loop)
                }
            }
        }
        .aspectRatio(stage.width / stage.height, contentMode: .fit)
        .accessibilityHidden(true)
    }
}

// The scene at one point of the loop, or at rest without a phase. The parts that hold still between
// frames are Equatable views, so SwiftUI skips them while the phase changes.
struct LogoScene: View {
    let scale: CGFloat
    let phase: Double?

    var body: some View {
        let state = phase.map(LogoGeometry.state(at:))
        ZStack {
            Canvas { context, _ in
                context.scaleBy(x: scale, y: scale)
                LogoGeometry.tile(&context, phase: phase ?? 0)
            }
            // Each edge layer gets its own glass, so upper edges cross over (and blur) lower ones.
            ForEach(LogoGeometry.edgeLayers.indices, id: \.self) { index in
                EdgeLayer(index: index, scale: scale, particles: state?.particles.filter { $0.layer == index } ?? [])
                    .equatable()
                EdgeGlass(index: index, scale: scale)
            }
            if let state {
                Canvas { context, _ in
                    context.scaleBy(x: scale, y: scale)
                    LogoGeometry.nodeLight(&context, state: state)
                }
            }
            Nodes(scale: scale, spinners: state?.spinners.map(\.position) ?? []).equatable()
        }
        .frame(width: LogoGeometry.stage.width * scale, height: LogoGeometry.stage.height * scale)
    }
}

// One edge layer's lines and the particles on them; it redraws only while a particle crosses it.
struct EdgeLayer: View, Equatable {
    let index: Int
    let scale: CGFloat
    let particles: [LogoGeometry.Particle]

    var body: some View {
        Canvas { context, _ in
            context.scaleBy(x: scale, y: scale)
            LogoGeometry.edgeLayer(&context, index, particles: particles)
        }
    }
}

// One edge layer's glass tubes, cut back to the lens rims. Apart from the lines, so particles
// moving under it don't rebuild the glass.
struct EdgeGlass: View {
    let index: Int
    let scale: CGFloat

    var body: some View {
        GlassEffectContainer(spacing: 28 * scale) {
            ZStack {
                ForEach(LogoGeometry.edgeLayers[index].edges.indices, id: \.self) { edge in
                    let (a, b) = LogoGeometry.edgeLayers[index].edges[edge]
                    let (from, to) = LogoGeometry.segment(a, b, trim: LogoGeometry.nodeRadius)
                    Color.clear.glassEffect(.clear, in: Tube(
                        from: from.scaled(scale), to: to.scaled(scale), width: LogoGeometry.tubeWidth * scale
                    ))
                }
            }
        }
    }
}

// The node lenses with the spinner droplets resting in them, under the white rings. Droplets move
// only while a node spins up, so most frames leave this view alone.
struct Nodes: View, Equatable {
    let scale: CGFloat
    let spinners: [CGPoint]

    var body: some View {
        ZStack {
            // Lenses and droplets blend, so a spinner pushes its lens's rim out.
            GlassEffectContainer(spacing: 28 * scale) {
                ZStack {
                    ForEach(LogoGeometry.nodes.indices, id: \.self) { index in
                        Color.clear
                            .frame(width: LogoGeometry.lensRadius * 2 * scale, height: LogoGeometry.lensRadius * 2 * scale)
                            .glassEffect(Glass.clear.interactive(), in: Circle())
                            .position(LogoGeometry.nodes[index].scaled(scale))
                    }
                    // Wide enough to push the lens rim out as it runs around under the ring.
                    ForEach(spinners.indices, id: \.self) { index in
                        Color.clear
                            .frame(width: 100 * scale, height: 100 * scale)
                            .glassEffect(.clear, in: Circle())
                            .position(spinners[index].scaled(scale))
                    }
                }
            }
            // The rings sit above the glass, so it can't dull them and spinners pass beneath.
            Canvas { context, _ in
                context.scaleBy(x: scale, y: scale)
                var rings = Path()
                for node in LogoGeometry.nodes {
                    rings.addPath(LogoGeometry.circle(node, LogoGeometry.nodeRadius - LogoGeometry.ringWidth / 2))
                }
                context.stroke(rings, with: .color(.white), lineWidth: LogoGeometry.ringWidth)
            }
            .allowsHitTesting(false)
        }
    }
}

private extension CGPoint {
    func scaled(_ scale: CGFloat) -> CGPoint { CGPoint(x: x * scale, y: y * scale) }
}
