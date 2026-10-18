// Swift releases output buffers that the Python binding retains.
// Pipe: R = RoPE tables, S = chain, T = packed tree, V = vision, Q = quit; integers are little-endian.

import CoreAI
import Darwin
import Foundation

let maskMin = Float16(-30000)  // model.MASK_MIN
let input = "hidden_in"
let output = "hidden_out"

struct HostError: Error, CustomStringConvertible {
  let description: String
  init(_ description: String) { self.description = description }
}

func readExact(_ count: Int) -> [UInt8]? {
  var buffer = [UInt8](repeating: 0, count: count)
  var done = 0
  while done < count {
    let n = buffer.withUnsafeMutableBytes { read(0, $0.baseAddress! + done, count - done) }
    if n <= 0 { return nil }
    done += n
  }
  return buffer
}

func readHalves(_ count: Int) -> [Float16]? {
  var values = [Float16](repeating: 0, count: count)
  let complete = values.withUnsafeMutableBytes { raw in
    var done = 0
    while done < raw.count {
      let n = read(0, raw.baseAddress! + done, raw.count - done)
      if n <= 0 { return false }
      done += n
    }
    return true
  }
  return complete ? values : nil
}

// Keep framework stdout logs out of the reply pipe.
nonisolated(unsafe) var replyFD: Int32 = 1

func protectReplies() {
  replyFD = dup(1)
  dup2(2, 1)
}

func writeAll(_ bytes: [UInt8]) {
  var done = 0
  while done < bytes.count {
    let n = bytes.withUnsafeBytes { write(replyFD, $0.baseAddress! + done, bytes.count - done) }
    if n <= 0 { exit(1) }  // Python is gone
    done += n
  }
}

func u32(_ bytes: [UInt8], _ at: Int) -> Int {
  Int(bytes[at]) | Int(bytes[at + 1]) << 8 | Int(bytes[at + 2]) << 16 | Int(bytes[at + 3]) << 24
}

func le32(_ value: Int) -> [UInt8] { (0..<4).map { UInt8(truncatingIfNeeded: value >> (8 * $0)) } }

func halves(_ bytes: [UInt8]) -> [Float16] {
  bytes.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
}

func sendJSON(_ object: [String: Any]) {
  let data = try! JSONSerialization.data(withJSONObject: object)
  writeAll(le32(data.count) + [UInt8](data))
}

func sendError(_ message: String) {
  let text = [UInt8](message.utf8)
  writeAll([1] + le32(text.count) + text)
}

// Core AI pads some axes; use strides for every fp16 array access.
func write4(_ array: inout NDArray, _ body: (UnsafeMutablePointer<Float16>, [Int], [Int]) -> Void) {
  let view = array.mutableView(as: Float16.self)
  view.withUnsafeMutablePointer { pointer, shape, strides in
    body(pointer, (0..<shape.count).map { shape[$0] }, (0..<strides.count).map { strides[$0] })
  }
}

func read4<R>(_ array: NDArray, _ body: (UnsafePointer<Float16>, [Int], [Int]) -> R) -> R {
  array.rawView().withUnsafeBytes { raw, shape, strides in
    body(
      raw.assumingMemoryBound(to: Float16.self),
      (0..<shape.count).map { shape[$0] }, (0..<strides.count).map { strides[$0] })
  }
}

func descriptor(_ value: InferenceValue.Descriptor?, _ name: String) throws -> NDArrayDescriptor {
  guard case .ndArray(let nd)? = value else { throw HostError("\(name) is not an ndarray") }
  return nd
}

func loadAsset(_ url: URL) async throws -> AIModel {
  // Core AI rejects an asset directory that is itself a symbolic link.
  let canonicalURL = url.resolvingSymlinksInPath()
  var options = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
  options.expectFrequentReshapes = false
  // Recover invalid persistent function caches before processing input.
  for attempt in 0..<2 {
    do {
      let model = try await AIModel.specialize(
        contentsOf: canonicalURL, options: options, cachePolicy: .persistent)
      for name in model.functionNames {
        guard try model.loadFunction(named: name) != nil else {
          throw HostError("missing function \(name)")
        }
      }
      return model
    } catch {
      let message = String(reflecting: error)
      if attempt == 1 || !message.contains("nilError") { throw error }
      FileHandle.standardError.write(
        Data(
          "REBUILD_INVALID_SPECIALIZATION asset=\(url.lastPathComponent) error=\(message)\n".utf8))
      try AIModelCache.default.deleteEntry(for: canonicalURL, options: options)
    }
  }
  fatalError("unreachable")
}
@_silgen_name("installStaging") func installStaging()

func column(_ array: NDArray, _ column: Int) -> [Float16] {
  read4(array) { p, shape, s in (0..<shape[1]).map { p[$0 * s[1] + column * s[3]] } }
}

func fillInput(_ array: inout NDArray, _ x: [Float16]) {
  write4(&array) { p, shape, s in
    for c in 0..<shape[1] {
      for t in 0..<shape[3] { p[c * s[1] + t * s[3]] = x[c * shape[3] + t] }
    }
  }
}

struct FusedManifest: Decodable {
  let width: Int, context: Int, hidden: Int
  let kv_heads: Int, head_dim: Int, conv_taps: Int
  let asset: String, tiers: [Int], conv_layers: [Int], attention_layers: [Int]
  let version: Int
  let vision_asset: String?
}

final class Model {
  let manifest: FusedManifest
  let root: URL
  let width: Int, context: Int, hiddenSize: Int
  let tiers: [Int], convCount: Int, cacheCount: Int
  let tailElements: Int, kvHeads: Int, headDim: Int
  var activeModel: AIModel?
  var function: InferenceFunction?
  var packed: NDArray?
  var inputs: [String: NDArray] = [:]
  var visionModel: AIModel?
  var visionFunction: InferenceFunction?
  var visionInputs: [String: NDArray] = [:]
  var cos: [Float16] = [], sin: [Float16] = [], positions = 0
  var tier = 0, visible = 0
  var chainInputsValid = false
  var tailCount: Int { convCount * tailElements }

  init(root: URL) throws {
    self.root = root
    let spec = try JSONDecoder().decode(
      FusedManifest.self, from: Data(contentsOf: root.appendingPathComponent("model.json")))
    guard spec.version == 2 else { throw HostError("unsupported fused bundle manifest") }
    manifest = spec
    width = manifest.width
    context = manifest.context
    hiddenSize = manifest.hidden
    kvHeads = manifest.kv_heads
    headDim = manifest.head_dim
    tiers = spec.tiers.map { $0 - spec.width }
    convCount = manifest.conv_layers.count
    cacheCount = manifest.attention_layers.count * 2
    tailElements = manifest.hidden * (manifest.conv_taps - 1) * 2
  }
  func ensureTier(_ needed: Int, keep: Int) async throws {
    guard let next = tiers.first(where: { $0 >= needed }) else {
      throw HostError("ANE context capacity exceeded")
    }
    guard next != tier else { return }
    let old = packed
    let oldTier = tier
    function = nil
    packed = nil
    malloc_zone_pressure_relief(nil, 0)
    let total = next + width
    if activeModel == nil {
      let model = try await loadAsset(root.appendingPathComponent(manifest.asset))
      guard Set(model.functionNames) == Set(manifest.tiers.map { "c\($0)" }) else {
        throw HostError("fused asset function set does not match manifest")
      }
      activeModel = model
    }
    let model = activeModel!
    let name = "c\(total)"
    guard let f = try model.loadFunction(named: name) else {
      throw HostError("missing fused function \(name)")
    }
    var grown = NDArray(
      descriptor: try descriptor(f.descriptor.inputDescriptor(of: "packed_state"), "packed_state"))
    let count = tailCount + cacheCount * kvHeads * next * headDim
    write4(&grown) { dst, shape, ds in
      precondition(shape == [1, 1, 1, count])
      precondition(ds[3] == 1)
      memset(dst, 0, count * 2)
      if keep > 0, let old {
        read4(old) { src, _, ss in
          precondition(ss[3] == 1 && ds[3] == 1)
          memcpy(dst, src, self.tailCount * 2)
          for cache in 0..<self.cacheCount {
            for h in 0..<self.kvHeads {
              let a = self.tailCount + (cache * self.kvHeads + h) * oldTier * self.headDim
              let b = self.tailCount + (cache * self.kvHeads + h) * next * self.headDim
              memcpy(dst.advanced(by: b), src.advanced(by: a), keep * self.headDim * 2)
            }
          }
        }
      }
    }
    packed = grown
    function = f
    tier = next
    visible = 0
    let names = [input, "cos", "sin", "chunk_mask", "conv_route"]
    for name in names where inputs[name] == nil {
      inputs[name] = NDArray(
        descriptor: try descriptor(f.descriptor.inputDescriptor(of: name), name))
    }
    inputs["past_mask"] = NDArray(
      descriptor: try descriptor(f.descriptor.inputDescriptor(of: "past_mask"), "past_mask"))
    write4(&inputs["past_mask"]!) { p, shape, s in
      for r in 0..<shape[2] { for c in 0..<shape[3] { p[r * s[2] + c * s[3]] = maskMin } }
    }
  }
  func setVisible(_ count: Int) {
    guard count != visible else { return }
    let (lo, hi, value) = count > visible ? (visible, count, Float16(0)) : (count, visible, maskMin)
    write4(&inputs["past_mask"]!) { p, shape, s in
      for r in 0..<shape[2] { for c in lo..<hi { p[r * s[2] + c * s[3]] = value } }
    }
    visible = count
  }
  func setRope(cos: [Float16], sin: [Float16], positions: Int) throws {
    guard positions >= context, cos.count == sin.count, cos.count == positions * headDim else {
      throw HostError("invalid RoPE")
    }
    (self.cos, self.sin, self.positions) = (cos, sin, positions)
  }
  func prepareChain() {
    guard !chainInputsValid else { return }
    let keep = manifest.conv_taps - 1
    // A chain's routing is translation invariant: the previous tail followed
    // by this chunk. Keep the unused trunk snapshot in its separate slots.
    write4(&inputs["conv_route"]!) { p, shape, s in
      for r in 0..<shape[2] {
        for c in 0..<shape[3] { p[r * s[2] + c * s[3]] = 0 }
      }
      for tap in 0..<self.manifest.conv_taps {
        for t in 0..<self.width {
          let ancestor = t + tap - keep
          let source = ancestor < 0 ? keep + ancestor : 2 * keep + ancestor
          p[source * s[2] + (tap * self.width + t) * s[3]] = 1
        }
      }
      let tails = self.width * self.manifest.conv_taps
      for t in 0..<keep {
        p[(2 * keep + self.width - keep + t) * s[2] + (tails + t) * s[3]] = 1
        p[(keep + t) * s[2] + (tails + keep + t) * s[3]] = 1
      }
    }
    write4(&inputs["chunk_mask"]!) { p, shape, s in
      for r in 0..<shape[2] {
        for c in 0..<shape[3] { p[r * s[2] + c * s[3]] = c <= r ? 0 : maskMin }
      }
    }
    // Tree masks vary per query row. Reset them before using the chain's
    // incremental, row-independent visibility updates.
    write4(&inputs["past_mask"]!) { p, shape, s in
      for r in 0..<shape[2] { for c in 0..<shape[3] { p[r * s[2] + c * s[3]] = maskMin } }
    }
    visible = 0
    chainInputsValid = true
  }
  func treeMetadata(offset: Int) throws -> [Int] {
    chainInputsValid = false
    let keep = manifest.conv_taps - 1
    let routes = width * manifest.conv_taps + 2 * keep
    guard let bytes = readExact((routes + width * 5) * 4) else {
      throw HostError("incomplete tree metadata")
    }
    let values = stride(from: 0, to: bytes.count, by: 4).map { u32(bytes, $0) }
    let positions = Array(values[routes..<routes + width])
    guard values[..<routes].allSatisfy({ $0 < width + 2 * keep }),
          positions.allSatisfy({ $0 < self.positions }) else {
      throw HostError("invalid tree routing or positions")
    }
    write4(&inputs["conv_route"]!) { p, shape, s in
      for r in 0..<shape[2] {
        for c in 0..<shape[3] { p[r * s[2] + c * s[3]] = 0 }
      }
      for c in 0..<routes { p[values[c] * s[2] + c * s[3]] = 1 }
    }
    for name in ["past_mask", "chunk_mask"] {
      write4(&inputs[name]!) { p, shape, s in
        for r in 0..<shape[2] {
          let at = routes + width + r * 4
          for c in 0..<shape[3] {
            let key = name == "past_mask" ? c : offset + c
            let valid = name != "past_mask" || c < offset
            let allowed = (values[at]..<values[at + 1]).contains(key)
                       || (values[at + 2]..<values[at + 3]).contains(key)
            p[r * s[2] + c * s[3]] = valid && allowed ? 0 : maskMin
          }
        }
      }
    }
    return positions
  }
  func hidden(tokens: Int, answers: [Int]? = nil) async throws -> [Float16] {
    guard positions > 0, tokens >= 1, tokens <= context
    else { throw HostError("invalid request") }
    let leaves = answers ?? [tokens - 1]
    guard !leaves.isEmpty, leaves == leaves.sorted(), Set(leaves).count == leaves.count,
          leaves.allSatisfy({ $0 >= 0 && $0 < tokens }) else {
      throw HostError("invalid answer slots")
    }
    let answerChunk = (tokens - 1) / width
    let initialNeeded = min(answerChunk * width, 4096 - width)
    try await ensureTier(initialNeeded, keep: 0)
    if answers == nil { prepareChain() }
    write4(&packed!) { p, _, s in
      precondition(s[3] == 1)
      memset(p, 0, self.tailCount * 2)
    }
    var result: [Float16] = []
    for chunk in 0...answerChunk {
      let offset = chunk * width
      let last = chunk == answerChunk
      try await ensureTier(offset + (last ? 0 : width), keep: offset)
      if answers == nil { setVisible(offset) }
      writeAll([0])
      let ropePositions = answers == nil ? Array(offset..<offset + width) : try treeMetadata(offset: offset)
      guard let values = readHalves(width * hiddenSize) else {
        throw HostError("incomplete embedding chunk")
      }
      fillInput(&inputs[input]!, values)
      for (name, table) in [("cos", cos), ("sin", sin)] {
        write4(&inputs[name]!) { p, shape, s in
          for i in 0..<shape[2] {
            for t in 0..<shape[3] {
              p[i * s[2] + t * s[3]] = table[i * self.positions + ropePositions[t]]
            }
          }
        }
      }
      var args = inputs
      args["packed_state"] = packed!
      var out = try await function!.run(inputs: args)
      args.removeAll(keepingCapacity: false)
      guard let hidden = out.remove(output)?.ndArray,
        let state = out.remove("packed_state_out")?.ndArray
      else { throw HostError("missing packed output") }
      for leaf in leaves where leaf / width == chunk {
        result += column(hidden, leaf % width)
      }
      if !last {
        read4(state) { src, shape, ss in
          precondition(
            shape == [
              1, 1, 1,
              self.tailCount + self.cacheCount * self.kvHeads * self.width * self.headDim,
            ])
          write4(&self.packed!) { dst, _, ds in
            precondition(ss[3] == 1 && ds[3] == 1)
            memcpy(dst, src, self.tailCount * 2)
            for cache in 0..<self.cacheCount {
              for h in 0..<self.kvHeads {
                let a = self.tailCount + (cache * self.kvHeads + h) * self.width * self.headDim
                let b =
                  self.tailCount + (cache * self.kvHeads + h) * self.tier * self.headDim + offset
                  * self.headDim
                memcpy(dst.advanced(by: b), src.advanced(by: a), self.width * self.headDim * 2)
              }
            }
          }
        }
      }
    }
    return result
  }
  func ensureVision() async throws {
    guard visionFunction == nil else { return }
    let assetName = manifest.vision_asset ?? "vision.\(AIModel.deviceArchitectureName).aimodelc"
    let url = root.appendingPathComponent(assetName)
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw HostError("missing vision asset at \(url.path)")
    }
    let model = try await loadAsset(url)
    guard let fn = try model.loadFunction(named: "main") else {
      throw HostError("missing vision function main")
    }
    visionModel = model
    visionFunction = fn
  }

  // Inputs and the features output are row-major in the asset's declared shapes.
  func runVision(_ values: [String: [Float16]]) async throws -> (shape: [Int], features: [Float16]) {
    try await ensureVision()
    for (name, x) in values {
      if visionInputs[name] == nil {
        visionInputs[name] = NDArray(
          descriptor: try descriptor(visionFunction!.descriptor.inputDescriptor(of: name), name))
      }
      var matches = false
      write4(&visionInputs[name]!) { p, shape, s in
        matches = shape.reduce(1, *) == x.count
        if matches { forEachOffset(shape, s) { i, offset in p[offset] = x[i] } }
      }
      guard matches else { throw HostError("vision input \(name) has the wrong element count") }
    }
    guard Set(values.keys) == Set(visionInputs.keys) else {
      throw HostError("vision inputs differ from the previous call")
    }
    var out = try await visionFunction!.run(inputs: visionInputs)
    guard let features = out.remove("features")?.ndArray else {
      throw HostError("missing vision features output")
    }
    return read4(features) { src, shape, s in
      var result = [Float16](repeating: 0, count: shape.reduce(1, *))
      forEachOffset(shape, s) { i, offset in result[i] = src[offset] }
      return (shape, result)
    }
  }
}

// Visit every element of a strided array in row-major order.
func forEachOffset(_ shape: [Int], _ strides: [Int], _ body: (Int, Int) -> Void) {
  let count = shape.reduce(1, *)
  var index = [Int](repeating: 0, count: shape.count)
  var offset = 0
  for linear in 0..<count {
    body(linear, offset)
    var axis = shape.count - 1
    while axis >= 0 {
      index[axis] += 1
      offset += strides[axis]
      if index[axis] < shape[axis] { break }
      offset -= strides[axis] * shape[axis]
      index[axis] = 0
      axis -= 1
    }
  }
}

@main struct Host {
  static func main() async {
    let args = CommandLine.arguments
    if args.count == 2 && args[1] == "--architecture" {
      print(AIModel.deviceArchitectureName)
      exit(0)
    }
    guard args.count == 2 else {
      FileHandle.standardError.write("usage: tinkerbox-ane-host <bundle>\n".data(using: .utf8)!)
      exit(2)
    }
    protectReplies()
    installStaging()
    let model: Model
    do {
      model = try Model(root: URL(fileURLWithPath: args[1]))
      sendJSON(["width": model.width, "context": model.context, "hidden": model.hiddenSize,
                "version": model.manifest.version, "conv_taps": model.manifest.conv_taps])
    } catch {
      sendJSON(["error": "\(error)"])
      exit(1)
    }
    while let op = readExact(1)?.first {
      switch op {
      case UInt8(ascii: "R"):
        guard let header = readExact(8) else { exit(1) }
        let (dim, positions) = (u32(header, 0), u32(header, 4))
        guard let c = readExact(dim * positions * 2), let s = readExact(dim * positions * 2) else {
          exit(1)
        }
        do {
          try model.setRope(cos: halves(c), sin: halves(s), positions: positions)
          writeAll([0])
        } catch { sendError("\(error)") }
      case UInt8(ascii: "S"):
        guard let header = readExact(4) else { exit(1) }
        let tokens = u32(header, 0)
        do {
          let h = try await model.hidden(tokens: tokens)
          var reply: [UInt8] = [0]
          h.withUnsafeBytes { reply += $0 }
          writeAll(reply)
        } catch { sendError("\(error)") }
      case UInt8(ascii: "T"):
        guard let header = readExact(8) else { exit(1) }
        let tokens = u32(header, 0), count = u32(header, 4)
        guard count > 0, count <= tokens, tokens <= model.context,
              let slots = readExact(count * 4) else { exit(1) }
        let answers = stride(from: 0, to: slots.count, by: 4).map { u32(slots, $0) }
        do {
          let h = try await model.hidden(tokens: tokens, answers: answers)
          var reply: [UInt8] = [0]
          h.withUnsafeBytes { reply += $0 }
          writeAll(reply)
        } catch { sendError("\(error)") }
      case UInt8(ascii: "V"):
        // Read every named input before validating, so an error leaves the pipe in sync.
        guard let header = readExact(4) else { exit(1) }
        var values: [String: [Float16]] = [:]
        for _ in 0..<u32(header, 0) {
          guard let length = readExact(4), let name = readExact(u32(length, 0)),
                let count = readExact(4), let x = readHalves(u32(count, 0)) else { exit(1) }
          values[String(decoding: name, as: UTF8.self)] = x
        }
        do {
          let (shape, features) = try await model.runVision(values)
          var reply: [UInt8] = [0] + le32(shape.count) + shape.flatMap(le32)
          features.withUnsafeBytes { reply += $0 }
          writeAll(reply)
        } catch { sendError("\(error)") }
      case UInt8(ascii: "Q"):
        exit(0)
      default:
        sendError("unknown op \(op)")
        exit(1)
      }
    }
    exit(0)
  }
}
