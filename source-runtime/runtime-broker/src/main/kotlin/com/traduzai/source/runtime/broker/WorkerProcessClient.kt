package com.traduzai.source.runtime.broker

import com.traduzai.source.runtime.protocol.ProtocolCodec
import com.traduzai.source.runtime.protocol.ProtocolRequest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException

class WorkerProcessClient {
    suspend fun dispatch(request: ProtocolRequest) = withContext(Dispatchers.IO) {
        val executable = Path.of(
            System.getProperty("java.home"),
            "bin",
            if (System.getProperty("os.name").startsWith("Windows")) "java.exe" else "java",
        ).toString()
        val argumentsFile = Files.createTempFile("traduzai-source-worker-", ".args")
        Files.writeString(
            argumentsFile,
            listOf(
                "-Xms32m",
                "-Xmx384m",
                "-Dfile.encoding=UTF-8",
                "-Dkotlin-logging.logStartupMessage=false",
                "-cp",
                quoteJavaArgument(System.getProperty("java.class.path")),
                "com.traduzai.source.runtime.worker.WorkerMainKt",
            ).joinToString(System.lineSeparator()),
            StandardCharsets.UTF_8,
        )
        var process: Process? = null
        val outputReader = Executors.newSingleThreadExecutor()
        try {
            val running = ProcessBuilder(
                executable,
                "@$argumentsFile",
            ).redirectError(ProcessBuilder.Redirect.DISCARD).start()
            process = running
            running.outputWriter(Charsets.UTF_8).use { writer ->
                writer.write(kotlinx.serialization.json.Json.encodeToString(ProtocolRequest.serializer(), request))
                writer.write("\n")
            }
            // Read while the worker is still running. Waiting for process exit first
            // deadlocks as soon as a catalog response fills the Windows stdout pipe.
            val responseFuture = outputReader.submit<String> {
                readBoundedLine(running.inputStream, MAX_WORKER_RESPONSE_BYTES)
            }
            val responseLine = try {
                responseFuture.get(request.deadlineMs + 1_000, TimeUnit.MILLISECONDS)
            } catch (_: TimeoutException) {
                running.destroyForcibly()
                throw DispatchException("SOURCE_TIMEOUT", "Worker excedeu o prazo")
            } catch (error: ExecutionException) {
                val cause = error.cause
                if (cause is DispatchException) throw cause
                throw DispatchException("WORKER_CRASHED", cause?.message ?: "Worker encerrou sem resposta")
            }
            if (!running.waitFor(1_000, TimeUnit.MILLISECONDS)) {
                running.destroyForcibly()
            }
            ProtocolCodec.decodeResponse(responseLine)
        } finally {
            if (process?.isAlive == true) process.destroyForcibly()
            outputReader.shutdownNow()
            Files.deleteIfExists(argumentsFile)
        }
    }
}

private const val MAX_WORKER_RESPONSE_BYTES = 1024 * 1024

private fun readBoundedLine(input: InputStream, maxBytes: Int): String {
    val output = ByteArrayOutputStream(minOf(maxBytes, 8 * 1024))
    while (true) {
        val next = input.read()
        if (next < 0) {
            if (output.size() == 0) throw DispatchException("WORKER_CRASHED", "Worker encerrou sem resposta")
            break
        }
        if (next == '\n'.code) break
        if (output.size() >= maxBytes) {
            throw DispatchException("PAYLOAD_TOO_LARGE", "Resposta do worker excedeu o limite")
        }
        output.write(next)
    }
    return output.toString(StandardCharsets.UTF_8).trimEnd('\r')
}

private fun quoteJavaArgument(value: String): String =
    "\"${value.replace("\\", "\\\\").replace("\"", "\\\"")}\""
