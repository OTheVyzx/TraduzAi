package com.traduzai.source.runtime.protocol

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ProtocolCodecTest {
    @Test
    fun `decodes a versioned request without losing a Long id`() {
        val request = ProtocolCodec.decodeRequest(
            """{"v":1,"id":"req-1","method":"source.search","deadlineMs":15000,"params":{"sourceId":"9223372036854775807"}}""",
        )

        assertEquals(1, request.v)
        assertEquals("req-1", request.id)
        assertEquals("source.search", request.method)
        assertEquals("9223372036854775807", request.params["sourceId"]?.toString()?.trim('"'))
    }

    @Test
    fun `rejects an incompatible protocol before dispatch`() {
        val error = assertFailsWith<ProtocolException> {
            ProtocolCodec.decodeRequest(
                """{"v":2,"id":"req-2","method":"runtime.hello","deadlineMs":1000,"params":{}}""",
            )
        }

        assertEquals("PROTOCOL_INCOMPATIBLE", error.code)
    }

    @Test
    fun `always emits the response protocol version`() {
        val encoded = ProtocolCodec.encodeResponse(
            ProtocolResponse(id = "req-3", ok = true),
        )

        assertTrue(encoded.contains("\"v\":1"), encoded)
    }

    @Test
    fun `accepts the bounded five minute deadline used by chapter downloads`() {
        val request = ProtocolCodec.decodeRequest(
            """{"v":1,"id":"download-1","method":"source.download-pages","deadlineMs":300000,"params":{}}""",
        )

        assertEquals(300_000L, request.deadlineMs)
    }

    @Test
    fun `preserves retry after metadata in a domain error`() {
        val encoded = ProtocolCodec.encodeResponse(ProtocolResponse(
            id = "retry-1",
            ok = false,
            error = ProtocolError("IMAGE_HTTP_ERROR", "HTTP 429", retryable = true, retryAfterSeconds = 7),
        ))

        val decoded = ProtocolCodec.decodeResponse(encoded)
        assertEquals(true, decoded.error?.retryable)
        assertEquals(7, decoded.error?.retryAfterSeconds)
    }
}
