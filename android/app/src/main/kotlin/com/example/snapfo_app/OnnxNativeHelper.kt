package com.example.snapfo_app

import ai.onnxruntime.*
import android.content.Context
import java.nio.FloatBuffer
import java.nio.IntBuffer
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

                    val dataFile = File("$modelPath.data")
                    if (!dataFile.exists()) {
                        println("Не найден data для onnx файла $key")
                    }

                    sessions[key] = env.createSession(modelPath, sessionOptions)
                    println("Модель $key успешно загружена из: $modelPath")
                }
                else -> {
                    sessions[key] = env.createSession(modelBytes, sessionOptions)
                    println("Модель $key успешно загрузилась из памяти")
                }
            }
        } catch (e: Exception) {
            throw RuntimeException("Ошибка загрузки $key: ${e.message}", e)
        }
    }

    private fun createTensor(data: Any, shape: LongArray? = null): OnnxTensor {
        return when (data) {
            is FloatArray -> {
                if (shape != null) {
                    OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
                } else {
                    OnnxTensor.createTensor(env, data)
                }
            }
            is IntArray -> {
                if (shape != null) {
                    OnnxTensor.createTensor(env, IntBuffer.wrap(data), shape)
                } else {
                    OnnxTensor.createTensor(env, IntBuffer.wrap(data))
                }
            }
            is List<*> -> {
                when {
                    data.all { it is Number } -> {
                        if (data[0] is Int) {
                            val intArray = IntArray(data.size) { i -> (data[i] as Number).toInt() }
                            OnnxTensor.createTensor(env, IntBuffer.wrap(intArray))
                        } else {
                            val floatArray = FloatArray(data.size) { i -> (data[i] as Number).toFloat() }
                            OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArray))
                        }
                    }
                    data.all { it is List<*> } -> {
                        val firstElement = (data[0] as List<*>)[0]
                        if (firstElement is Int) {
                            val flattened = data.flatMap { (it as List<Number>).map { num -> num.toInt() } }
                            val finalShape = shape ?: longArrayOf(data.size.toLong(), (data[0] as List<*>).size.toLong())
                            OnnxTensor.createTensor(env, IntBuffer.wrap(flattened.toIntArray()), finalShape)
                        } else {
                            val flattened = data.flatMap { (it as List<Number>).map { num -> num.toFloat() } }
                            val finalShape = shape ?: longArrayOf(data.size.toLong(), (data[0] as List<*>).size.toLong())
                            OnnxTensor.createTensor(env, FloatBuffer.wrap(flattened.toFloatArray()), finalShape)
                        }
                    }
                    else -> throw IllegalArgumentException("Unsupported list type")
                }
            }
            else -> throw IllegalArgumentException("Unsupported data type: ${data.javaClass}")
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
        val shape = when (val shapeData = data["shape"]) {
            is List<*> -> shapeData.map { (it as Number).toLong() }.toLongArray()
            is LongArray -> shapeData
            else -> throw IllegalArgumentException("Неподдерживаемый тип shape: ${shapeData?.javaClass}")
        }
        val type = data["type"] as String? ?: "float32"
        val tensorData = data["data"] ?: throw IllegalArgumentException("Данные тензора null")

        return when {
            type == "int32" && tensorData is IntArray -> {
                OnnxTensor.createTensor(env, IntBuffer.wrap(tensorData), shape)
            }
            type == "float32" && tensorData is FloatArray -> {
                OnnxTensor.createTensor(env, FloatBuffer.wrap(tensorData), shape)
            }
            else -> throw IllegalArgumentException("Неподдерживаемый тип данных тензора: ${tensorData.javaClass} для типа $type")
        }
    }

    fun runComplex(sessionKey: String, inputs: Map<String, Map<String, Any>>, outputNames: List<String>): Map<String, Any> {
        val session = sessions[sessionKey] ?: throw IllegalStateException("Session not loaded")
        val inputTensors = mutableMapOf<String, OnnxTensor>()

        try {
            inputs.forEach { (name, tensorData) ->
                inputTensors[name] = createTensorFromMap(tensorData)
            }

            val results = session.run(inputTensors)
            return outputNames.associate { name ->
                val tensor = results.get(name)?.get() as? OnnxTensor
                    ?: throw IllegalStateException("No output tensor for $name")
                val shape = tensor.info.shape
                val totalSize = shape.reduce { acc, i -> acc * i }.toInt()

                when (tensor.info.type) {
                    OnnxJavaType.FLOAT -> {
                        val floatArray = FloatArray(totalSize)
                        tensor.floatBuffer.get(floatArray)
                        name to floatArray
                    }
                    OnnxJavaType.INT32 -> {
                        val intArray = IntArray(totalSize)
                        tensor.intBuffer.get(intArray)
                        name to intArray
                    }
                    else -> throw IllegalStateException("Unsupported tensor type: ${tensor.info.type} for output $name")
                }
            }
        } finally {
            inputTensors.values.forEach { it.close() }
        }
    }
}