package com.example.snapfo_app;

import ai.onnxruntime.*
import android.content.Context
import java.nio.FloatBuffer
import java.io.File
import java.io.IOException

class OnnxNativeHelper(context: Context) {
    private lateinit var env: OrtEnvironment
    private var sessions = mutableMapOf<String, OrtSession>()

    fun initEnv() {
        env = OrtEnvironment.getEnvironment()
    }

    fun loadModel(key: String, modelBytes: ByteArray, modelPath: String? = null) {
        try {
            val sessionOptions = OrtSession.SessionOptions()
            when {
                modelPath != null && key == "pre_editor" -> {
                    val modelFile = File(modelPath)
                    if (modelFile.length() == 0L) {
                        throw IOException("Размер модели мал или она пустая ")
                    }
                    if (!modelFile.exists()) {
                        throw IOException("Файл модели не найден: [ $modelPath ]")
                    }

                    // Проверяем наличие внешних данных
                    val dataFile = File("$modelPath.data")
                    if (!dataFile.exists()) {
                        println("Не найден data для onnx файла $key")
                    }

                    sessions[key] = env.createSession(modelPath, sessionOptions)
                    println("Модель $key успешно загружена из: $modelPath")
                }
                // Для всех остальных случаев загружаем из памяти
                else -> {
                    sessions[key] = env.createSession(modelBytes, sessionOptions)
                    println("Модель $key успешно загрузилась из памяти")
                }
            }
        } catch (e: Exception) {
            throw RuntimeException("Ошибка загрузки $key: ${e.message}", e)
        }
    }

//    fun copyAssetToFilesDir(assetPath: String): String {
//        val targetFile = File(context.filesDir, File(assetPath).name)
//        try {
//            context.assets.open(assetPath).use { input ->
//                FileOutputStream(targetFile).use { output ->
//                    input.copyTo(output)
//                }
//            }
//            return targetFile.absolutePath
//        } catch (e: IOException) {
//            throw RuntimeException("Ошибка копирования из assets: $assetPath", e)
//        }
//    }

    // Добавляем поддержку разных типов тензоров
    private fun createTensor(data: Any, shape: LongArray? = null): OnnxTensor {
        return when (data) {
            is FloatArray -> {
                if (shape != null) {
                    OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
                } else {
                    OnnxTensor.createTensor(env, data)
                }
            }
            is List<*> -> {
                when {
                    data.all { it is Number } -> {
                        val floatArray = FloatArray(data.size) { i -> (data[i] as Number).toFloat() }
                        OnnxTensor.createTensor(env, floatArray)
                    }
                    data.all { it is List<*> } -> {
                        // Рекурсивная обработка многомерных массивов
                        val flattened = data.flatMap { (it as List<Number>).map { num -> num.toFloat() } }
                        val finalShape = shape ?: longArrayOf(data.size.toLong(), (data[0] as List<*>).size.toLong())
                        OnnxTensor.createTensor(env, FloatBuffer.wrap(flattened.toFloatArray()), finalShape)
                    }
                    else -> throw IllegalArgumentException("Unsupported list type")
                }
            }
            else -> throw IllegalArgumentException("Unsupported data type")
        }
    }

    fun run(sessionKey: String, inputs: Map<String, Any>, outputNames: List<String>): Map<String, FloatArray> {
        val session = sessions[sessionKey] ?: throw IllegalStateException("Session not loaded")
        val inputTensors = mutableMapOf<String, OnnxTensor>()

        try {
            inputs.forEach { (name, value) ->
                inputTensors[name] = createTensor(value)
            }

            val results = session.run(inputTensors)
            return outputNames.associate { name ->
                val tensor = results.get(name)?.get() as OnnxTensor
                val floatArray = FloatArray(tensor.info.shape.reduce { acc, i -> acc * i }.toInt())
                tensor.floatBuffer.get(floatArray)
                name to floatArray
            }
        } finally {
            inputTensors.values.forEach { it.close() }
        }
    }

    fun createTensorFromMap(data: Map<String, Any>): OnnxTensor {
        val buffer = data["data"] as FloatArray
        val shape = (data["shape"] as List<Number>).map { it.toLong() }.toLongArray()
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(buffer), shape)
    }

    fun runComplex(sessionKey: String, inputs: Map<String, Map<String, Any>>, outputNames: List<String>): Map<String, List<Float>> {
        val session = sessions[sessionKey] ?: throw IllegalStateException("Session not loaded")
        val inputTensors = mutableMapOf<String, OnnxTensor>()

        try {
            inputs.forEach { (name, tensorData) ->
                val data = tensorData["data"] as FloatArray
                val shape = tensorData["shape"] as LongArray
                inputTensors[name] = OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
            }

            val results = session.run(inputTensors)
            return outputNames.associate { name ->
                val tensor = results.get(name)?.get() as? OnnxTensor
                    ?: throw IllegalStateException("No output tensor for $name")

                val floatBuffer = tensor.floatBuffer
                val floatArray = FloatArray(floatBuffer.remaining()).apply {
                    floatBuffer.get(this)
                }

                name to floatArray.toList()
            }
        } finally {
            inputTensors.values.forEach { it.close() }
        }
    }
}