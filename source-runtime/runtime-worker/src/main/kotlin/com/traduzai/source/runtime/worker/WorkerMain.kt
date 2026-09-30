package com.traduzai.source.runtime.worker

import com.traduzai.source.runtime.protocol.ProtocolCodec
import com.traduzai.source.runtime.protocol.ProtocolError
import com.traduzai.source.runtime.protocol.ProtocolException
import com.traduzai.source.runtime.protocol.ProtocolResponse
import kotlinx.coroutines.runBlocking
import java.io.FileDescriptor
import java.io.FileOutputStream

fun main() {
    val protocolOutput = FileOutputStream(FileDescriptor.out).bufferedWriter(Charsets.UTF_8)
    // Extension libraries are untrusted protocol peers. Keep their console
    // output away from the one-line JSON response channel.
    System.setOut(System.err)
    runBlocking {
        val input = System.`in`.bufferedReader(Charsets.UTF_8)
        val line = input.readLine() ?: return@runBlocking
        val response = try {
            WorkerDispatcher().dispatch(ProtocolCodec.decodeRequest(line))
        } catch (error: ProtocolException) {
            ProtocolResponse(
                id = "invalid",
                ok = false,
                error = ProtocolError(error.code, error.message ?: "Requisição inválida"),
            )
        }
        protocolOutput.write(ProtocolCodec.encodeResponse(response))
        protocolOutput.newLine()
        protocolOutput.flush()
    }
}
