// Chunk-major runner for hybrid convolution/attention stacks (LFM): every layer runs on one
// 128-token chunk at a time, carrying convolution tails and growing attention caches; packed
// trees share a trunk across question branches.

import CoreAI
import Darwin
import Foundation

struct FusedManifest: Decodable {
  let width: Int, context: Int, hidden: Int
  let kv_heads: Int, head_dim: Int, conv_taps: Int
  let asset: String, tiers: [Int], conv_layers: [Int], attention_layers: [Int]
}

final class ChunkedRunner: Runner {
  let manifest: FusedManifest
  let root: URL
  let width: Int, context: Int, hiddenSize: Int
  let tiers: [Int], convCount: Int, cacheCount: Int
  let tailElements: Int, kvHeads: Int, headDim: Int
  var activeModel: AIModel?
  var function: InferenceFunction?
  var packed: NDArray?
  var inputs: [String: NDArray] = [:]
  var cos: [Float16] = [], sin: [Float16] = [], positions = 0
  var tier = 0, visible = 0
  var chainInputsValid = false
  var tailCount: Int { convCount * tailElements }

  init(root: URL) throws {
    self.root = root
    let spec = try JSONDecoder().decode(
      FusedManifest.self, from: Data(contentsOf: root.appendingPathComponent("model.json")))
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
  var ready: [String: Any] { ["width": width, "context": context, "hidden": hiddenSize,
                               "conv_taps": manifest.conv_taps] }

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
      var out = try await timed("c\(tier + width)", function!, args)
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
}
