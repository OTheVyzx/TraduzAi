package com.traduzai.source.runtime.catalog

import java.io.ByteArrayOutputStream
import java.net.InetAddress
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.security.MessageDigest
import java.time.Duration

data class RepositoryPreview(
    val index: RepositoryIndex,
    val certificateFingerprint: String,
    val indexSha256: String,
)

data class DownloadedArtifact(
    val bytes: ByteArray,
    val certificateFingerprint: String,
    val finalUrl: String,
)

class RepositoryClient(
    private val allowPrivateHosts: Boolean = false,
    private val allowInsecureHttp: Boolean = false,
) {
    private val client = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(10))
        .followRedirects(HttpClient.Redirect.NEVER)
        .build()

    fun preview(sourceUrl: String): RepositoryPreview {
        var uri = validatedUri(sourceUrl)
        repeat(6) { redirectIndex ->
            val request = HttpRequest.newBuilder(uri)
                .timeout(Duration.ofSeconds(20))
                .header("Accept", "application/json, application/octet-stream")
                .header("User-Agent", "TraduzAI-SourceRuntime/1")
                .GET()
                .build()
            val response = client.send(request, HttpResponse.BodyHandlers.ofInputStream())
            if (response.statusCode() in setOf(301, 302, 303, 307, 308)) {
                if (redirectIndex == 5) throw RepositoryException("TOO_MANY_REDIRECTS", "Redirecionamentos demais")
                val location = response.headers().firstValue("Location").orElseThrow {
                    RepositoryException("INVALID_REDIRECT", "Redirect sem Location")
                }
                uri = validatedUri(uri.resolve(location).toString())
                return@repeat
            }
            if (response.statusCode() !in 200..299) {
                throw RepositoryException("REPOSITORY_HTTP_ERROR", "HTTP ${response.statusCode()}")
            }
            val payload = response.body().use { input ->
                val output = ByteArrayOutputStream()
                val buffer = ByteArray(8192)
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    output.write(buffer, 0, read)
                    if (output.size() > 5 * 1024 * 1024) {
                        throw RepositoryException("INDEX_TOO_LARGE", "Índice excede o limite")
                    }
                }
                output.toByteArray()
            }
            val certificateFingerprint = response.sslSession().map { session ->
                session.peerCertificates.firstOrNull()?.encoded?.sha256()
                    ?: throw RepositoryException("CERTIFICATE_MISSING", "Certificado TLS ausente")
            }.orElse("UNENCRYPTED")
            return RepositoryPreview(
                index = RepositoryIndex.parse(payload, uri.toString()),
                certificateFingerprint = certificateFingerprint,
                indexSha256 = payload.sha256(),
            )
        }
        throw RepositoryException("TOO_MANY_REDIRECTS", "Redirecionamentos demais")
    }

    fun downloadArtifact(sourceUrl: String, maxBytes: Int = 100 * 1024 * 1024): DownloadedArtifact {
        var uri = validatedUri(sourceUrl)
        repeat(6) { redirectIndex ->
            val response = client.send(
                HttpRequest.newBuilder(uri)
                    .timeout(Duration.ofSeconds(60))
                    .header("Accept", "application/java-archive, application/vnd.android.package-archive, application/octet-stream")
                    .header("User-Agent", "TraduzAI-SourceRuntime/1")
                    .GET()
                    .build(),
                HttpResponse.BodyHandlers.ofInputStream(),
            )
            if (response.statusCode() in setOf(301, 302, 303, 307, 308)) {
                if (redirectIndex == 5) throw RepositoryException("TOO_MANY_REDIRECTS", "Redirecionamentos demais")
                val location = response.headers().firstValue("Location").orElseThrow {
                    RepositoryException("INVALID_REDIRECT", "Redirect sem Location")
                }
                uri = validatedUri(uri.resolve(location).toString())
                return@repeat
            }
            if (response.statusCode() !in 200..299) {
                throw RepositoryException("ARTIFACT_HTTP_ERROR", "HTTP ${response.statusCode()}")
            }
            val bytes = response.body().use { input ->
                val output = ByteArrayOutputStream()
                val buffer = ByteArray(8192)
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    output.write(buffer, 0, read)
                    if (output.size() > maxBytes) {
                        throw RepositoryException("PACKAGE_SIZE_INVALID", "Pacote excede o limite")
                    }
                }
                output.toByteArray()
            }
            val fingerprint = response.sslSession().map { session ->
                session.peerCertificates.firstOrNull()?.encoded?.sha256()
                    ?: throw RepositoryException("CERTIFICATE_MISSING", "Certificado TLS ausente")
            }.orElse("UNENCRYPTED")
            return DownloadedArtifact(bytes, fingerprint, uri.toString())
        }
        throw RepositoryException("TOO_MANY_REDIRECTS", "Redirecionamentos demais")
    }

    private fun validatedUri(value: String): URI {
        val uri = try { URI.create(value) } catch (_: Exception) {
            throw RepositoryException("INVALID_REPOSITORY_URL", "URL inválida")
        }
        if (uri.scheme == "http" && !allowInsecureHttp) {
            throw RepositoryException("INSECURE_REPOSITORY", "O repositório precisa usar HTTPS")
        }
        if (uri.scheme !in setOf("http", "https") || uri.host.isNullOrBlank()) {
            throw RepositoryException("INVALID_REPOSITORY_URL", "Apenas HTTP(S) é aceito")
        }
        if (!allowPrivateHosts) {
            for (address in InetAddress.getAllByName(uri.host)) {
                if (
                    address.isAnyLocalAddress || address.isLoopbackAddress || address.isLinkLocalAddress ||
                    address.isSiteLocalAddress || address.isMulticastAddress
                ) {
                    throw RepositoryException("PRIVATE_NETWORK_BLOCKED", "Rede privada bloqueada")
                }
            }
        }
        return uri
    }
}

private fun ByteArray.sha256(): String = MessageDigest.getInstance("SHA-256").digest(this)
    .joinToString("") { byte -> "%02x".format(byte) }
