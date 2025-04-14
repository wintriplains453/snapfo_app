package com.example.snapfo_app;

import androidx.annotation.NonNull
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity: FlutterActivity() {
    private val CHANNEL = "com.example.snapfo_app/onnx"
    private val onnxHelper = OnnxNativeHelper(this)

    override fun configureFlutterEngine(@NonNull flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        // Регистрируем MethodChannel
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
                        onnxHelper.loadModel(key, modelBytes)
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

                        val processedInputs = inputs.mapValues { (_, tensorMap) ->
                            mapOf(
                                "data" to (tensorMap["data"] as List<Double>).map { it.toFloat() }.toFloatArray(),
                                "shape" to (tensorMap["shape"] as List<Int>).map { it.toLong() }.toLongArray()
                            )
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