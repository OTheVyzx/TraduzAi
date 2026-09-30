@file:OptIn(kotlinx.serialization.ExperimentalSerializationApi::class)

package com.traduzai.source.runtime.catalog

import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromByteArray
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.protobuf.ProtoBuf
import kotlinx.serialization.protobuf.ProtoNumber
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.net.URI
import java.util.zip.GZIPInputStream

private const val MAX_INDEX_BYTES = 5 * 1024 * 1024

data class RepositorySource(
    val name: String,
    val lang: String,
    val id: String,
    val baseUrl: String?,
)

data class RepositoryExtension(
    val name: String,
    val packageName: String,
    val versionCode: Long,
    val versionName: String,
    val lang: String,
    val nsfw: Int,
    val iconUrl: String?,
    val jarUrl: String?,
    val apkUrl: String?,
    val artifactSha256: String?,
    val sources: List<RepositorySource>,
)

data class RepositoryIndex(
    val sourceUrl: String,
    val extensions: List<RepositoryExtension>,
    val name: String? = null,
    val badgeLabel: String? = null,
    val signingKey: String? = null,
) {
    companion object {
        fun parse(payload: String, sourceUrl: String): RepositoryIndex =
            parse(payload.toByteArray(Charsets.UTF_8), sourceUrl)

        fun parse(payload: ByteArray, sourceUrl: String): RepositoryIndex {
            if (payload.size > MAX_INDEX_BYTES) {
                throw RepositoryException("INDEX_TOO_LARGE", "Índice excede $MAX_INDEX_BYTES bytes")
            }
            return if (payload.isGzip()) {
                parseProtobuf(payload.gunzipBounded(), sourceUrl)
            } else {
                parseJson(payload.toString(Charsets.UTF_8), sourceUrl)
            }
        }

        private fun parseJson(payload: String, sourceUrl: String): RepositoryIndex {
            val base = parseWebUri(sourceUrl)
            val root = try {
                Json.parseToJsonElement(payload)
            } catch (error: Exception) {
                throw RepositoryException("INVALID_INDEX", error.message ?: "JSON inválido")
            }
            val modern = root is JsonObject && root["extensionList"] != null
            val entries = when (root) {
                is JsonArray -> root
                is JsonObject -> root["extensionList"]?.jsonObject?.get("extensions")?.jsonArray
                    ?: root["extensions"]?.jsonArray
                    ?: throw RepositoryException("INVALID_INDEX", "Campo extensions ausente")
                else -> throw RepositoryException("INVALID_INDEX", "Raiz do índice deve ser array ou objeto")
            }
            return RepositoryIndex(
                sourceUrl = sourceUrl,
                extensions = entries.map {
                    if (modern) parseModernExtension(it, base) else parseLegacyExtension(it, base)
                },
                name = (root as? JsonObject)?.optional("name"),
                badgeLabel = (root as? JsonObject)?.optional("badgeLabel"),
                signingKey = (root as? JsonObject)?.optional("signingKey"),
            )
        }

        private fun parseLegacyExtension(element: JsonElement, base: URI): RepositoryExtension {
            val item = element.jsonObject
            val packageName = item.required("pkg")
            val jar = item.optional("jarUrl") ?: item.optional("jar")
            val apk = item.optional("apkUrl") ?: item.optional("apk")
            if (jar == null && apk == null) {
                throw RepositoryException("INVALID_INDEX", "$packageName não declara JAR nem APK")
            }
            return RepositoryExtension(
                name = item.required("name"),
                packageName = packageName,
                versionCode = item.required("code").toLongOrNull()
                    ?: throw RepositoryException("INVALID_INDEX", "versionCode inválido em $packageName"),
                versionName = item.required("version"),
                lang = item.optional("lang") ?: "all",
                nsfw = item.optional("nsfw")?.toIntOrNull() ?: 0,
                iconUrl = (item.optional("iconUrl") ?: item.optional("icon"))?.let { resolveWebUrl(base, it) },
                jarUrl = jar?.let { resolveWebUrl(base, it) },
                apkUrl = apk?.let { resolveWebUrl(base, it) },
                artifactSha256 = item.optional("sha256")?.lowercase(),
                sources = item["sources"]?.jsonArray?.map { source ->
                    val value = source.jsonObject
                    RepositorySource(
                        name = value.required("name"),
                        lang = value.optional("lang") ?: "all",
                        id = value.required("id"),
                        baseUrl = value.optional("baseUrl"),
                    )
                }.orEmpty(),
            )
        }

        private fun parseModernExtension(element: JsonElement, base: URI): RepositoryExtension {
            val item = element.jsonObject
            val packageName = item.required("packageName")
            val resources = item["resources"]?.jsonObject
                ?: throw RepositoryException("INVALID_INDEX", "Campo resources ausente em $packageName")
            val jar = resources.optional("jarUrl")
            val apk = resources.optional("apkUrl")
            if (jar == null && apk == null) {
                throw RepositoryException("INVALID_INDEX", "$packageName não declara JAR nem APK")
            }
            val sources = item["sources"]?.jsonArray?.map { source ->
                val value = source.jsonObject
                RepositorySource(
                    name = value.required("name"),
                    lang = value.optional("language") ?: "all",
                    id = value.required("id"),
                    baseUrl = value.optional("homeUrl"),
                )
            }.orEmpty()
            return RepositoryExtension(
                name = item.required("name"),
                packageName = packageName,
                versionCode = item.required("versionCode").toLongOrNull()
                    ?: throw RepositoryException("INVALID_INDEX", "versionCode inválido em $packageName"),
                versionName = item.required("versionName"),
                lang = sources.map { it.lang }.distinct().singleOrNull() ?: "all",
                nsfw = item.optional("contentWarning").toNsfwLevel(),
                iconUrl = resources.optional("iconUrl")?.let { resolveWebUrl(base, it) },
                jarUrl = jar?.let { resolveWebUrl(base, it) },
                apkUrl = apk?.let { resolveWebUrl(base, it) },
                artifactSha256 = null,
                sources = sources,
            )
        }

        private fun parseProtobuf(payload: ByteArray, sourceUrl: String): RepositoryIndex {
            val base = parseWebUri(sourceUrl)
            val index = try {
                ProtoBuf.decodeFromByteArray<ProtoIndex>(payload)
            } catch (error: Exception) {
                throw RepositoryException("INVALID_INDEX", error.message ?: "Protobuf inválido")
            }
            if (index.extensionListUrl.isNotBlank()) {
                throw RepositoryException("EXTERNAL_INDEX_UNSUPPORTED", "Índice externo ainda não é suportado")
            }
            return RepositoryIndex(
                sourceUrl = sourceUrl,
                extensions = index.extensionList.extensions.map { it.toRepositoryExtension(base) },
                name = index.name.takeIf { it.isNotBlank() },
                badgeLabel = index.badgeLabel.takeIf { it.isNotBlank() },
                signingKey = index.signingKey.takeIf { it.isNotBlank() },
            )
        }
    }
}

private fun ProtoExtension.toRepositoryExtension(base: URI): RepositoryExtension {
    val packageValue = packageName.ifBlank {
        throw RepositoryException("INVALID_INDEX", "Extensão sem packageName")
    }
    val jar = resources.jarUrl.takeIf { it.isNotBlank() }
    val apk = resources.apkUrl.takeIf { it.isNotBlank() }
    if (jar == null && apk == null) {
        throw RepositoryException("INVALID_INDEX", "$packageValue não declara JAR nem APK")
    }
    val parsedSources = sources.map { source ->
        RepositorySource(
            name = source.name,
            lang = source.language.ifBlank { "all" },
            id = source.id.toString(),
            baseUrl = source.homeUrl.takeIf { it.isNotBlank() },
        )
    }
    return RepositoryExtension(
        name = name,
        packageName = packageValue,
        versionCode = versionCode,
        versionName = versionName,
        lang = parsedSources.map { it.lang }.distinct().singleOrNull() ?: "all",
        nsfw = contentWarning.name.toNsfwLevel(),
        iconUrl = resources.iconUrl.takeIf { it.isNotBlank() }?.let { resolveWebUrl(base, it) },
        jarUrl = jar?.let { resolveWebUrl(base, it) },
        apkUrl = apk?.let { resolveWebUrl(base, it) },
        artifactSha256 = null,
        sources = parsedSources,
    )
}

@Serializable
private class ProtoIndex(
    @ProtoNumber(1) val name: String = "",
    @ProtoNumber(2) val badgeLabel: String = "",
    @ProtoNumber(3) val signingKey: String = "",
    @ProtoNumber(101) val extensionList: ProtoExtensionList = ProtoExtensionList(),
    @ProtoNumber(102) val extensionListUrl: String = "",
)

@Serializable
private class ProtoExtensionList(
    @ProtoNumber(1) val extensions: List<ProtoExtension> = emptyList(),
)

@Serializable
private class ProtoExtension(
    @ProtoNumber(1) val name: String = "",
    @ProtoNumber(2) val packageName: String = "",
    @ProtoNumber(3) val resources: ProtoResources = ProtoResources(),
    @ProtoNumber(4) val extensionLib: String = "",
    @ProtoNumber(5) val versionCode: Long = 0,
    @ProtoNumber(6) val versionName: String = "",
    @ProtoNumber(7) val contentWarning: ProtoContentWarning = ProtoContentWarning.CONTENT_WARNING_UNSPECIFIED,
    @ProtoNumber(8) val sources: List<ProtoSource> = emptyList(),
)

@Serializable
private class ProtoResources(
    @ProtoNumber(1) val apkUrl: String = "",
    @ProtoNumber(2) val iconUrl: String = "",
    @ProtoNumber(501) val jarUrl: String = "",
)

@Serializable
private class ProtoSource(
    @ProtoNumber(1) val id: Long = 0,
    @ProtoNumber(2) val name: String = "",
    @ProtoNumber(3) val language: String = "",
    @ProtoNumber(4) val homeUrl: String = "",
    @ProtoNumber(5) val mirrorUrls: List<String> = emptyList(),
    @ProtoNumber(7) val message: String = "",
)

@Serializable
private enum class ProtoContentWarning {
    CONTENT_WARNING_UNSPECIFIED,
    CONTENT_WARNING_SAFE,
    CONTENT_WARNING_MIXED,
    CONTENT_WARNING_NSFW,
}

class RepositoryException(
    val code: String,
    message: String,
) : IllegalArgumentException(message)

private fun JsonObject.required(key: String): String = optional(key)
    ?: throw RepositoryException("INVALID_INDEX", "Campo $key ausente")

private fun JsonObject.optional(key: String): String? = get(key)?.jsonPrimitive?.content

private fun parseWebUri(value: String): URI {
    val uri = try {
        URI.create(value)
    } catch (_: Exception) {
        throw RepositoryException("INVALID_REPOSITORY_URL", "URL inválida")
    }
    if (uri.scheme !in setOf("https", "http") || uri.host.isNullOrBlank()) {
        throw RepositoryException("INVALID_REPOSITORY_URL", "Apenas repositórios HTTP(S) são aceitos")
    }
    return uri
}

private fun resolveWebUrl(base: URI, value: String): String {
    val resolved = base.resolve(value)
    parseWebUri(resolved.toString())
    return resolved.toString()
}

private fun String?.toNsfwLevel(): Int = when (this) {
    "CONTENT_WARNING_MIXED" -> 1
    "CONTENT_WARNING_NSFW" -> 2
    else -> 0
}

private fun ByteArray.isGzip(): Boolean =
    size >= 2 && this[0] == 0x1f.toByte() && this[1] == 0x8b.toByte()

private fun ByteArray.gunzipBounded(): ByteArray = try {
    GZIPInputStream(ByteArrayInputStream(this)).use { input ->
        val output = ByteArrayOutputStream()
        val buffer = ByteArray(8192)
        while (true) {
            val read = input.read(buffer)
            if (read < 0) break
            output.write(buffer, 0, read)
            if (output.size() > MAX_INDEX_BYTES) {
                throw RepositoryException("INDEX_TOO_LARGE", "Índice descompactado excede $MAX_INDEX_BYTES bytes")
            }
        }
        output.toByteArray()
    }
} catch (error: RepositoryException) {
    throw error
} catch (error: Exception) {
    throw RepositoryException("INVALID_INDEX", error.message ?: "Gzip inválido")
}
