package com.traduzai.source.runtime.protocol

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject

const val PROTOCOL_VERSION = 1
const val MAX_MESSAGE_BYTES = 1024 * 1024
const val MAX_DEADLINE_MS = 300_000L

@Serializable
data class ProtocolRequest(
    val v: Int,
    val id: String,
    val method: String,
    val deadlineMs: Long,
    val params: JsonObject,
)

@Serializable
data class ProtocolError(
    val code: String,
    val message: String,
    val retryable: Boolean = false,
    val retryAfterSeconds: Long? = null,
)

@Serializable
data class ProtocolResponse(
    val v: Int = PROTOCOL_VERSION,
    val id: String,
    val ok: Boolean,
    val result: JsonElement? = null,
    val error: ProtocolError? = null,
)

class ProtocolException(
    val code: String,
    message: String,
) : IllegalArgumentException(message)

object ProtocolCodec {
    private val json = Json {
        ignoreUnknownKeys = false
        explicitNulls = false
        encodeDefaults = true
    }

    fun decodeRequest(line: String): ProtocolRequest {
        if (line.toByteArray(Charsets.UTF_8).size > MAX_MESSAGE_BYTES) {
            throw ProtocolException("PAYLOAD_TOO_LARGE", "A mensagem excede $MAX_MESSAGE_BYTES bytes")
        }

        val request = try {
            json.decodeFromString<ProtocolRequest>(line)
        } catch (error: Exception) {
            throw ProtocolException("INVALID_REQUEST", error.message ?: "JSON inválido")
        }

        if (request.v != PROTOCOL_VERSION) {
            throw ProtocolException("PROTOCOL_INCOMPATIBLE", "Protocolo ${request.v} não suportado")
        }
        if (request.id.isBlank() || request.id.length > 128) {
            throw ProtocolException("INVALID_REQUEST", "id ausente ou inválido")
        }
        if (request.method.isBlank() || request.method.length > 128) {
            throw ProtocolException("INVALID_REQUEST", "method ausente ou inválido")
        }
        if (request.deadlineMs !in 1..MAX_DEADLINE_MS) {
            throw ProtocolException("INVALID_DEADLINE", "deadlineMs fora do limite")
        }

        return request
    }

    fun encodeResponse(response: ProtocolResponse): String = json.encodeToString(response)

    fun decodeResponse(line: String): ProtocolResponse = try {
        json.decodeFromString<ProtocolResponse>(line)
    } catch (error: Exception) {
        throw ProtocolException("INVALID_RESPONSE", error.message ?: "JSON de resposta inválido")
    }
}
