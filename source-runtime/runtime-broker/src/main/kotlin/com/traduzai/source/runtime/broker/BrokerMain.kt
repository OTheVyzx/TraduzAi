package com.traduzai.source.runtime.broker

import com.traduzai.source.runtime.protocol.ProtocolCodec
import com.traduzai.source.runtime.protocol.ProtocolError
import com.traduzai.source.runtime.protocol.ProtocolException
import com.traduzai.source.runtime.protocol.ProtocolResponse
import kotlinx.coroutines.runBlocking
import java.io.FileDescriptor
import java.io.FileOutputStream

fun main() {
    val protocolOutput = FileOutputStream(FileDescriptor.out).bufferedWriter(Charsets.UTF_8)
    // The broker owns stdout. Third-party logging is diagnostic-only and is
    // redirected before any extension or Android compatibility code starts.
    System.setOut(System.err)
    runBlocking {
        val dispatcher = BrokerDispatcher()
        val input = System.`in`.bufferedReader(Charsets.UTF_8)
        input.lineSequence().forEach { line ->
            val response = try {
                val request = ProtocolCodec.decodeRequest(line)
                dispatcher.dispatch(request)
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
}
