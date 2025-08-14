//
//  Attention.swift
//  HierarchicalReasoningModel
//
//  Created by Tanmay Bakshi on 2025-08-04.
//

import Foundation
import MLX
import MLXNN
import MLXRandom

public class Attention: Module {
    public let dim: Int
    public let headDim: Int
    public let numHeads: Int
    public let outputSize: Int
    public let keyValueHeadsPerHead: Int
    public let numKeyValueHeads: Int

    public let qkvProj: Linear
    public let outProj: Linear

    public init(
        dim: Int,
        headDim: Int,
        numHeads: Int,
        keyValueHeadsPerHead: Int,
        dtype: DType = .float32,
        key: MLXArray? = nil,
    ) {
        self.dim = dim
        self.headDim = headDim
        self.numHeads = numHeads
        self.outputSize = headDim * numHeads
        self.keyValueHeadsPerHead = keyValueHeadsPerHead
        self.numKeyValueHeads = numHeads * keyValueHeadsPerHead

        let (qkvKey, outKey): (MLXArray?, MLXArray?) =
            if let key = key {
                MLXRandom.split(key: key)
            } else {
                (nil, nil)
            }
        self.qkvProj = Linear(
            inDim: dim,
            outDim: (numHeads + 2 * numKeyValueHeads) * headDim,
            bias: false,
            key: qkvKey
        )
        self.outProj = Linear(inDim: outputSize, outDim: dim, bias: false, key: outKey)
    }

    public func callAsFunction(_ x: MLXArray, rotaryPositionEmbedding: RotaryPositionEmbedding?)
        -> MLXArray
    {
        let (batchSize, seqLen, _) = x.shape3

        let qkv = qkvProj(x)
            .reshaped([batchSize, seqLen, numHeads + 2 * numKeyValueHeads, headDim])
        var query = qkv[0..., 0..., ..<numHeads]
        var key = qkv[0..., 0..., numHeads..<(numHeads + numKeyValueHeads)]
        var value = qkv[0..., 0..., (numHeads + numKeyValueHeads)...]

        if let rotaryPositionEmbedding {
            query = rotaryPositionEmbedding(query)
            key = rotaryPositionEmbedding(key)
        }

        query =
            query
            .reshaped([batchSize, seqLen, numKeyValueHeads, keyValueHeadsPerHead, headDim])
            .transposed(0, 2, 3, 1, 4)
        key =
            key
            .transposed(0, 2, 3, 1)
            .expandedDimensions(axis: 2)
        value =
            value
            .transposed(0, 2, 1, 3)
            .expandedDimensions(axis: 2)

        let attnLogits = matmul(query, key) * (1.0 / sqrtf(Float(headDim)))
        let attnWeights = softmax(attnLogits.asType(DType.float32), axis: -1).asType(
            attnLogits.dtype)
        let combined = matmul(attnWeights, value)
            .transposed(0, 3, 1, 2, 4)
            .reshaped([batchSize, seqLen, dim])

        return outProj(combined)
    }
}

//
//  Embedding.swift
//  HierarchicalReasoningModel
//
//  Created by Tanmay Bakshi on 2025-08-04.
//

import MLX
import MLXNN

public class Embedding: Module, UnaryLayer {
    public let embeddings: MLXArray

    public init(
        vocabSize: Int,
        dim: Int,
        initStd: Float,
        dtype: DType = .float32,
        key: MLXArray? = nil,
    ) {
        self.embeddings = truncNormalInit([vocabSize, dim], std: initStd, dtype: dtype, key: key)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return embeddings[x]
    }
}

//
//  HRM.swift
//  HierarchicalReasoningModel
//
//  Created by Tanmay Bakshi on 2025-08-04.
//

import Foundation
import MLX
import MLXNN
import MLXRandom

public struct HRMACTModelConfig {
    public struct TransformerConfig {
        public let numLayers: Int
        public let hiddenSize: Int
        public let numHeads: Int
        public let expansion: Float
        public let normEpsilon: Float = 1e-5
        public let ropeTheta: Float = 10000.0
    }

    public struct ACTConfig {
        public let haltMaxSteps: Int
        public let haltExplorationProbability: Float
    }

    public let seqLen: Int
    public let vocabSize: Int

    public let highLevelCycles: Int
    public let lowLevelCycles: Int

    public let transformers: TransformerConfig

    public let act: ACTConfig

    public let dtype: DType = .bfloat16
}

public class HRMACTBlock: Module {
    public let selfAttn: Attention
    public let mlp: SwiGLU
    public let normEpsilon: Float

    public init(
        hiddenSize: Int,
        numHeads: Int,
        expansion: Float,
        normEpsilon: Float,
        dtype: DType = .float32,
        key: MLXArray? = nil,
    ) {
        let (selfAttnKey, mlpKey): (MLXArray?, MLXArray?) =
            if let key = key {
                MLXRandom.split(key: key)
            } else {
                (nil, nil)
            }
        self.selfAttn = Attention(
            dim: hiddenSize,
            headDim: hiddenSize / numHeads,
            numHeads: numHeads,
            keyValueHeadsPerHead: 1,
            dtype: dtype,
            key: selfAttnKey,
        )
        self.mlp = SwiGLU(dim: hiddenSize, expansion: expansion, dtype: dtype, key: mlpKey)
        self.normEpsilon = normEpsilon
    }

    public func callAsFunction(_ x: MLXArray, rotaryPositionEmbedding: RotaryPositionEmbedding?)
        -> MLXArray
    {
        var x = x
        x = rmsNorm(
            x + selfAttn(x, rotaryPositionEmbedding: rotaryPositionEmbedding), epsilon: normEpsilon)
        x = rmsNorm(x + mlp(x), epsilon: normEpsilon)
        return x
    }
}

public class HRMACTReasoner: Module {
    public let blocks: [HRMACTBlock]

    public init(
        numLayers: Int,
        hiddenSize: Int,
        numHeads: Int,
        expansion: Float,
        normEpsilon: Float,
        dtype: DType = .float32,
        key: MLXArray? = nil,
    ) {
        let keys =
            if let key = key {
                MLXRandom.split(key: key, into: numLayers)
            } else {
                [MLXArray?](repeating: nil, count: numLayers)
            }
        self.blocks = keys.map { key in
            HRMACTBlock(
                hiddenSize: hiddenSize,
                numHeads: numHeads,
                expansion: expansion,
                normEpsilon: normEpsilon,
                dtype: dtype,
                key: key
            )
        }
    }

    public func callAsFunction(
        hiddenState: MLXArray, inputInjection: MLXArray,
        rotaryPositionEmbedding: RotaryPositionEmbedding?
    ) -> MLXArray {
        var hiddenState = hiddenState + inputInjection
        for block in blocks {
            hiddenState = block(hiddenState, rotaryPositionEmbedding: rotaryPositionEmbedding)
        }
        return hiddenState
    }
}

public class HRMACTInner: Module {
    private class InitialHiddenStates: Module {
        let highLevel: MLXArray
        let lowLevel: MLXArray

        init(hiddenSize: Int, dtype: DType = .float32, key: MLXArray? = nil) {
            let (hlKey, llKey): (MLXArray?, MLXArray?) =
                if let key = key {
                    MLXRandom.split(key: key)
                } else {
                    (nil, nil)
                }
            self.highLevel = truncNormalInit([hiddenSize], std: 1.0, dtype: dtype, key: hlKey)
            self.lowLevel = truncNormalInit([hiddenSize], std: 1.0, dtype: dtype, key: llKey)

            super.init()
            self.freeze()
        }
    }

    public struct HiddenStates: Evaluatable {
        public var highLevel: MLXArray
        public var lowLevel: MLXArray

        public func innerState() -> [MLXArray] {
            [highLevel, lowLevel]
        }

        public func map(_ transform: (MLXArray) -> (MLXArray)) -> HiddenStates {
            HiddenStates(
                highLevel: transform(highLevel),
                lowLevel: transform(lowLevel),
            )
        }
    }

    public struct Output: Evaluatable {
        public let hiddenStates: HiddenStates
        public let output: MLXArray
        public let qACTHalt: MLXArray
        public let qACTContinue: MLXArray

        public func innerState() -> [MLXArray] {
            hiddenStates.innerState() + [output, qACTHalt, qACTContinue]
        }
    }

    public let config: HRMACTModelConfig

    public let clsToken: MLXArray
    public let inputEmbedding: Embedding
    public let outputHead: Linear
    public let qACTHead: Linear

    public let rotaryEmb: RotaryPositionEmbedding
    public let highLevelReasoner: HRMACTReasoner
    public let lowLevelReasoner: HRMACTReasoner

    // TODO: Is this safe — will the parameters still be saved and loaded correctly?
    private let initialHiddenStatesParams: InitialHiddenStates
    public var initialHiddenStates: HiddenStates {
        HiddenStates(
            highLevel: initialHiddenStatesParams.highLevel,
            lowLevel: initialHiddenStatesParams.lowLevel
        )
    }

    public init(config: HRMACTModelConfig, key: MLXArray) {
        self.config = config

        var key = key

        let clsTokenKey: MLXArray
        (key, clsTokenKey) = MLXRandom.split(key: key)
        self.clsToken = truncNormalInit(
            [config.transformers.hiddenSize],
            std: 1.0 / sqrtf(Float(config.transformers.hiddenSize)), dtype: config.dtype,
            key: clsTokenKey)

        let embeddingKey: MLXArray
        (key, embeddingKey) = MLXRandom.split(key: key)
        self.inputEmbedding = Embedding(
            vocabSize: config.vocabSize,
            dim: config.transformers.hiddenSize,
            initStd: 1.0 / sqrtf(Float(config.transformers.hiddenSize)),
            dtype: config.dtype,
            key: embeddingKey
        )

        let outputKey: MLXArray
        (key, outputKey) = MLXRandom.split(key: key)
        self.outputHead = Linear(
            inDim: config.transformers.hiddenSize,
            outDim: config.vocabSize,
            bias: false,
            dtype: config.dtype,
            key: outputKey
        )

        self.qACTHead = Linear(
            weight: MLXArray.zeros([config.transformers.hiddenSize, 2], dtype: config.dtype),
            bias: MLXArray.zeros([2], dtype: config.dtype) - 5,
        )

        self.rotaryEmb = RotaryPositionEmbedding(
            dim: config.transformers.hiddenSize / config.transformers.numHeads,
            maxLength: config.seqLen + 1,  // + CLS token
            base: config.transformers.ropeTheta,
            dtype: config.dtype
        )

        let highLevelReasonerKey: MLXArray
        (key, highLevelReasonerKey) = MLXRandom.split(key: key)
        self.highLevelReasoner = HRMACTReasoner(
            numLayers: config.transformers.numLayers,
            hiddenSize: config.transformers.hiddenSize,
            numHeads: config.transformers.numHeads,
            expansion: config.transformers.expansion,
            normEpsilon: config.transformers.normEpsilon,
            dtype: config.dtype,
            key: highLevelReasonerKey
        )

        let lowLevelReasonerKey: MLXArray
        (key, lowLevelReasonerKey) = MLXRandom.split(key: key)
        self.lowLevelReasoner = HRMACTReasoner(
            numLayers: config.transformers.numLayers,
            hiddenSize: config.transformers.hiddenSize,
            numHeads: config.transformers.numHeads,
            expansion: config.transformers.expansion,
            normEpsilon: config.transformers.normEpsilon,
            dtype: config.dtype,
            key: lowLevelReasonerKey
        )

        self.initialHiddenStatesParams = InitialHiddenStates(
            hiddenSize: config.transformers.hiddenSize,
            dtype: config.dtype,
            key: key
        )
    }

    public func callAsFunction(hiddenStates: HiddenStates, inputs: MLXArray) -> Output {
        let inputEmbeddings =
            concatenated(
                [
                    stacked([MLXArray](repeating: clsToken[.newAxis], count: inputs.shape[0])),
                    inputEmbedding(inputs),
                ],
                axis: 1
            ) * sqrtf(Float(config.transformers.hiddenSize))

        var (lowLevelZ, highLevelZ) = (hiddenStates.lowLevel, hiddenStates.highLevel)
        for cycle in 1...(config.highLevelCycles * config.lowLevelCycles - 1) {
            lowLevelZ = lowLevelReasoner(
                hiddenState: lowLevelZ,
                inputInjection: highLevelZ + inputEmbeddings,
                rotaryPositionEmbedding: rotaryEmb
            )
            eval(lowLevelZ)
            if cycle % config.lowLevelCycles == 0 {
                highLevelZ = highLevelReasoner(
                    hiddenState: highLevelZ,
                    inputInjection: lowLevelZ,
                    rotaryPositionEmbedding: rotaryEmb
                )
                eval(highLevelZ)
            }
        }

        lowLevelZ = stopGradient(lowLevelZ)
        highLevelZ = stopGradient(highLevelZ)

        lowLevelZ = lowLevelReasoner(
            hiddenState: lowLevelZ,
            inputInjection: highLevelZ + inputEmbeddings,
            rotaryPositionEmbedding: rotaryEmb
        )
        highLevelZ = highLevelReasoner(
            hiddenState: highLevelZ,
            inputInjection: lowLevelZ,
            rotaryPositionEmbedding: rotaryEmb
        )

        let outputLogits = outputHead(highLevelZ[0..., 1...])
        let qACTLogits = qACTHead(highLevelZ[0..., 0])

        return Output(
            hiddenStates: .init(
                highLevel: stopGradient(highLevelZ), lowLevel: stopGradient(lowLevelZ)),
            output: outputLogits,
            qACTHalt: qACTLogits[0..., 0],
            qACTContinue: qACTLogits[0..., 1]
        )
    }
}

//
//  Linear.swift
//  HierarchicalReasoningModel
//
//  Created by Tanmay Bakshi on 2025-08-04.
//

import Foundation
import MLX
import MLXNN

public class Linear: Module, UnaryLayer {
    public let weight: MLXArray
    public let bias: MLXArray?

    public init(
        inDim: Int,
        outDim: Int,
        bias: Bool = true,
        dtype: DType = .float32,
        key: MLXArray? = nil,
    ) {
        self.weight = truncNormalInit(
            [inDim, outDim], std: 1.0 / powf(Float(inDim), 0.5), dtype: dtype, key: key)
        self.bias = bias ? MLXArray.zeros([outDim], dtype: dtype) : nil
    }

    public init(weight: MLXArray, bias: MLXArray?) {
        self.weight = weight
        self.bias = bias
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x = matmul(x, weight)
        if let bias = bias {
            return x + bias
        } else {
            return x
        }
    }
}

//
//  RMSNorm.swift
//  HierarchicalReasoningModel
//
//  Created by Tanmay Bakshi on 2025-08-04.
//

import MLX

public func rmsNorm(_ x: MLXArray, epsilon: Float = 1e-6) -> MLXArray {
    let originalDtype = x.dtype
    let x = x.asType(.float32)

    let variance = x.square().mean(axis: -1, keepDims: true)
    return (x * rsqrt(variance + epsilon)).asType(originalDtype)
}

//
//  RotaryPositionEmbedding.swift
//  HierarchicalReasoningModel
//
//  Created by Tanmay Bakshi on 2025-08-04.
//

import MLX
import MLXNN

public class RotaryPositionEmbedding: Module {
    public let cos: MLXArray
    public let sin: MLXArray

    public init(
        dim: Int,
        maxLength: Int,
        base: Float,
        dtype: DType = .float32
    ) {
        let invFreq =
            1.0
            / MLXArray(base, dtype: dtype).pow(
                MLXArray(stride(from: 0, through: dim - 1, by: 2)).asType(dtype) / Float(dim))
        let t = MLXArray(0..<maxLength).asType(dtype)
        let freqs = outer(t, invFreq)

        let emb = concatenated([freqs, freqs], axis: -1)
            .expandedDimensions(axis: -2)
        self.cos = emb.cos()
        self.sin = emb.sin()

        super.init()
        self.freeze()
    }

    private static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let x = x.movedAxis(source: -1, destination: 0)
        let halfDim = x.shape[0] / 2
        let x1 = x[..<halfDim]
        let x2 = x[halfDim...]
        return concatenated([-x2, x1], axis: 0).movedAxis(source: 0, destination: -1)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        (x * self.cos) + (Self.rotateHalf(x) * self.sin)
    }
}

//
//  SwiGLU.swift
//  HierarchicalReasoningModel
//
//  Created by Tanmay Bakshi on 2025-08-04.
//

import MLX
import MLXNN
import MLXRandom

public class SwiGLU: Module, UnaryLayer {
    public let gateUpProj: Linear
    public let downProj: Linear

    public init(
        dim: Int,
        expansion: Float,
        dtype: DType = .float32,
        key: MLXArray? = nil,
    ) {
        let (upKey, downKey): (MLXArray?, MLXArray?) =
            if let key = key {
                MLXRandom.split(key: key)
            } else {
                (nil, nil)
            }
        let interDim = Self.findMultiple(Int(expansion * Float(dim) * 2.0 / 3.0), 256)
        self.gateUpProj = Linear(inDim: dim, outDim: interDim * 2, bias: false, key: upKey)
        self.downProj = Linear(inDim: interDim, outDim: dim, bias: false, key: downKey)
    }

    private static func findMultiple(_ a: Int, _ b: Int) -> Int {
        (-(a / -b)) * b
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (gate, up) = gateUpProj(x).split(axis: -1)
        return downProj(silu(gate) * up)
    }
}

//
//  TruncNormalInit.swift
//  HierarchicalReasoningModel
//
//  Created by Tanmay Bakshi on 2025-08-04.
//

import Foundation
import MLX
import MLXRandom

public func truncNormalInit(
    _ shape: [Int],
    std: Float = 1.0,
    lower: Float = -2.0,
    upper: Float = 2.0,
    dtype: DType = .float32,
    key: MLXArray? = nil,
) -> MLXArray {
    if std == 0.0 {
        return MLXArray.zeros(shape, dtype: dtype)
    }

    let sqrt2 = sqrtf(2.0)
    let a = erff(lower / sqrt2)
    let b = erff(upper / sqrt2)
    let z = (b - a) / 2

    let c = powf(2 * Float.pi, -0.5)
    let pdfU = c * expf(-0.5 * powf(lower, 2))
    let pdfL = c * expf(-0.5 * powf(upper, 2))
    let compStd =
        std / sqrtf(1.0 - (upper * pdfU - lower * pdfL) / z - powf(((pdfU - pdfL) / z), 2))

    var result = MLXRandom.uniform(low: a, high: b, shape, key: key)
    result = erfInverse(result)
    result = result * (sqrt2 * compStd)
    result = clip(result, min: lower * compStd, max: upper * compStd)

    return result
}

//
//  HierarchicalReasoningModel.swift
//  HierarchicalReasoningModel
//
//  Created by Tanmay Bakshi on 2025-08-04.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

@main
struct HierarchicalReasoningModel {
    static func train() throws {
        var key = MLXRandom.key(42)

        let modelKey: MLXArray
        (key, modelKey) = MLXRandom.split(key: key)
        let model = HRMACTInner(
            config: HRMACTModelConfig(
                seqLen: 9 * 9,
                vocabSize: 10,
                highLevelCycles: 2,
                lowLevelCycles: 2,
                transformers: .init(numLayers: 4, hiddenSize: 256, numHeads: 4, expansion: 4),
                act: .init(haltMaxSteps: 16, haltExplorationProbability: 0.1)
            ),
            key: modelKey
        )
        eval(model)

        let optimizer = AdamW(learningRate: 1e-4, betas: (0.9, 0.95))

        var batch = TrainingBatch(initialHiddenState: model.initialHiddenStates, size: 512)

        var stepIdx = 0
        var stepsSinceGraduation = 0
        var accuracyHistory: [Float] = [Float](repeating: 0, count: 300)
        while true {
            stepIdx += 1
            stepsSinceGraduation += 1
            print("Step \(stepIdx)")

            let stepKey: MLXArray
            (key, stepKey) = MLXRandom.split(key: key)
            let ((_, outputAcc), (_, _)) = step(
                model: model, optimizer: optimizer, batch: &batch, key: stepKey)

            if stepIdx == 1 || stepIdx % 250 == 0 {
                try MLX.save(
                    arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
                    url: URL(filePath: "checkpoint-\(stepIdx).safetensors")
                )
            }

            accuracyHistory.removeFirst()
            accuracyHistory.append(outputAcc)
            let avgRollingAccuracy = accuracyHistory.reduce(0, +) / Float(accuracyHistory.count)
            if avgRollingAccuracy >= 0.85 && stepsSinceGraduation >= 300 {
                stepsSinceGraduation = 0
                batch.graduate()
            }

            print()
        }
    }

    static func infer(checkpoint: String, difficulty: Difficulty) throws {
        var key = MLXRandom.key(42)

        let weights = try MLX.loadArrays(url: URL(filePath: checkpoint))
        let parameters = ModuleParameters.unflattened(weights)

        let modelKey: MLXArray
        (key, modelKey) = MLXRandom.split(key: key)
        let model = try HRMACTInner(
            config: HRMACTModelConfig(
                seqLen: 9 * 9,
                vocabSize: 10,
                highLevelCycles: 2,
                lowLevelCycles: 2,
                transformers: .init(numLayers: 4, hiddenSize: 256, numHeads: 4, expansion: 4),
                act: .init(haltMaxSteps: 16, haltExplorationProbability: 0.1)
            ),
            key: modelKey
        ).update(parameters: parameters, verify: .all)
        eval(model)
        print("Loaded model!")

        let (rawPuzzle, rawSolution) = generateSudoku(difficulty: difficulty)
        print("Puzzle:\n\(sudokuBoardString(rawPuzzle))")
        print("Solution:\n\(sudokuBoardString(rawSolution))")

        let puzzleIn = MLXArray(rawPuzzle.flatMap { $0 }).asType(.int32)[.newAxis]
        var hiddenStates = model
            .initialHiddenStates
            .map { $0[.newAxis, .newAxis] }
        for segment in 1...model.config.act.haltMaxSteps {
            print("\nSegment \(segment)")

            let output = model(hiddenStates: hiddenStates, inputs: puzzleIn)
            hiddenStates = output.hiddenStates

            let predictions = output.output[0].argMax(axis: 1)

            var accurateSquares = 0
            var predictedSquares = 0
            let predictedFlatBoard = zip(
                zip(rawPuzzle.flatMap({ $0 }), rawSolution.flatMap({ $0 })),
                predictions.asArray(Int.self)
            )
            .map { puzz, predictedSquare in
                let (puzzleSquare, solutionSquare) = puzz

                if puzzleSquare != 0 {
                    return puzzleSquare
                }

                accurateSquares += predictedSquare == solutionSquare ? 1 : 0
                predictedSquares += 1

                return predictedSquare
            }
            let predictedBoard = stride(from: 0, through: 9 * 9 - 1, by: 9)
                .map { Array(predictedFlatBoard[$0..<($0 + 9)]) }
            print(
                "Predicted solution (\(accurateSquares) / \(predictedSquares)):\n\(sudokuBoardString(predictedBoard))"
            )

            let (qHalt, qContinue): (Float, Float) = (
                sigmoid(output.qACTHalt[0]).item(), sigmoid(output.qACTContinue[0]).item()
            )
            print("Q (halt - continue): \(qHalt) - \(qContinue)")

            if qHalt > qContinue {
                print("Halting.")
                break
            }
        }
    }

    static func main() throws {
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            print("Usage: \(args[0]) train | infer <checkpoint-path> <difficulty>")
            return
        }
        let mode = args[1]

        switch mode {
        case "train":
            try train()
        case "infer":
            guard args.count >= 4 else {
                print("Usage: \(args[0]) infer <checkpoint-path> <difficulty>")
                return
            }
            let checkpoint = args[2]
            let diffStr = args[3]

            let difficulty: Difficulty
            switch diffStr {
            case "very-easy": difficulty = .veryEasy
            case "easy": difficulty = .easy
            case "medium": difficulty = .medium
            case "hard": difficulty = .hard
            case "extreme": difficulty = .extreme
            default:
                fatalError(
                    "Unknown difficulty: \(diffStr). Expected one of veryEasy, easy, medium, hard, extreme."
                )
            }

            try infer(checkpoint: checkpoint, difficulty: difficulty)
        default:
            fatalError("Unknown mode: \(mode). Expected 'train' or 'infer'.")
        }
    }
}

