package com.example.snapfo_app

import androidx.annotation.NonNull
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity: FlutterActivity() {
    private val CHANNEL = "com.example.snapfo_app/onnx"
    private val onnxHelper = OnnxNativeHelper(this)

    override fun configureFlutterEngine(@NonNull flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "initEnv" -> {
                    try {
                        onnxHelper.initEnv()
                        result.success(null)
                    } catch (e: Exception) {
                        result.error("INIT_ERROR", e.message, null)
                    }
                }
                "loadModel" -> {
                    try {
                        val key = call.argument<String>("key")!!
                        val modelBytes = call.argument<ByteArray>("modelBytes")!!
                        val modelPath = call.argument<String>("modelPath")
                        onnxHelper.loadModel(key, modelBytes, modelPath)
                        result.success(null)
                    } catch (e: Exception) {
                        result.error("LOAD_ERROR", e.message, null)
                    }
                }
                "runInference" -> {
                    val sessionKey = call.argument<String>("sessionKey")!!
                    val inputs = call.argument<Map<String, Any>>("inputs")!!
                    val outputs = call.argument<List<String>>("outputNames")!!
                    try {
                        val results = onnxHelper.run(sessionKey, inputs, outputs)
                        result.success(results)
                    } catch (e: Exception) {
                        result.error("INFERENCE_ERROR", e.message, null)
                    }
                }
                "runComplex" -> {
                    try {
                        val sessionKey = call.argument<String>("sessionKey")!!
                        val inputs = call.argument<Map<String, Map<String, Any>>>("inputs")!!
                        val outputNames = call.argument<List<String>>("outputNames")!!

                        // Преобразуем входные данные в нужный формат
                        val processedInputs = inputs.mapValues { (_, tensorMap) ->
                            val type = tensorMap["type"] as String? ?: "float32"
                            // Обрабатываем shape как List<Int> или long[]
                            val shape = when (val shapeData = tensorMap["shape"]) {
                                is List<*> -> shapeData.map { (it as Number).toLong() }.toLongArray()
                                is LongArray -> shapeData
                                else -> throw IllegalArgumentException("Неподдерживаемый тип shape: ${shapeData?.javaClass}")
                            }
                            val data = tensorMap["data"]

                            when {
                                type == "int32" && data is List<*> -> {
                                    val intData = data.map { (it as Number).toInt() }.toIntArray()
                                    mapOf(
                                        "data" to intData,
                                        "shape" to shape,
                                        "type" to type
                                    )
                                }
                                type == "int32" && data is IntArray -> {
                                    mapOf(
                                        "data" to data,
                                        "shape" to shape,
                                        "type" to type
                                    )
                                }
                                type == "float32" && data is List<*> -> {
                                    val floatData = data.map { (it as Number).toFloat() }.toFloatArray()
                                    mapOf(
                                        "data" to floatData,
                                        "shape" to shape,
                                        "type" to type
                                    )
                                }
                                type == "float32" && data is FloatArray -> {
                                    mapOf(
                                        "data" to data,
                                        "shape" to shape,
                                        "type" to type
                                    )
                                }
                                else -> throw IllegalArgumentException("Неподдерживаемый тип данных: ${data?.javaClass} для типа $type")
                            }
                        }

                        val inferenceResults = onnxHelper.runComplex(sessionKey, processedInputs, outputNames)
                        result.success(inferenceResults)
                    } catch (e: Exception) {
                        result.error("COMPLEX_INFERENCE_ERROR", e.message, null)
                    }
                }
                else -> result.notImplemented()
            }
        }
    }
}