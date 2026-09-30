package com.traduzai.source.runtime.worker

import java.net.InetAddress
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest
import java.time.Duration

data class StagingManifestEntry(
    val relativePath: String,
    val mime: String,
    val size: Long,
    val sha256: String,
)

class StagingDownloader(
    private val allowPrivateHosts: Boolean = false,
    private val maxImageBytes: Int = 20 * 1024 * 1024,
) {
    private val client = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(10))
        .followRedirects(HttpClient.Redirect.NEVER)
        .build()

    fun downloadImage(url: String, stagingRoot: Path, relativeName: String): StagingManifestEntry {
        require(relativeName.isNotBlank() && Path.of(relativeName).nameCount == 1) {
            "Nome relativo de staging inválido"
        }
        val uri = URI.create(url)
        require(uri.scheme == "https" || uri.scheme == "http") { "Apenas HTTP(S) é permitido" }
        require(!uri.host.isNullOrBlank()) { "URL sem host" }
        if (!allowPrivateHosts) rejectPrivateHost(uri.host)

        val request = HttpRequest.newBuilder(uri)
            .timeout(Duration.ofSeconds(30))
            .header("User-Agent", "TraduzAI-SourceRuntime/1")
            .GET()
            .build()
        val response = client.send(request, HttpResponse.BodyHandlers.ofByteArray())
        require(response.statusCode() in 200..299) { "HTTP ${response.statusCode()} ao baixar imagem" }
        val bytes = response.body()
        require(bytes.size in 1..maxImageBytes) { "Imagem vazia ou acima do limite" }
        val mime = response.headers().firstValue("Content-Type").orElse("")
            .substringBefore(';').trim().lowercase()
        require(mime in setOf("image/png", "image/jpeg", "image/webp", "image/gif", "image/avif")) {
            "MIME de imagem não permitido: $mime"
        }

        val normalizedRoot = stagingRoot.toAbsolutePath().normalize()
        Files.createDirectories(normalizedRoot)
        val target = normalizedRoot.resolve(relativeName).normalize()
        require(target.parent == normalizedRoot) { "Destino escapou do staging" }
        Files.write(target, bytes)
        return StagingManifestEntry(
            relativePath = relativeName,
            mime = mime,
            size = bytes.size.toLong(),
            sha256 = MessageDigest.getInstance("SHA-256").digest(bytes).toHex(),
        )
    }

    private fun rejectPrivateHost(host: String) {
        val addresses = InetAddress.getAllByName(host)
        require(addresses.isNotEmpty()) { "Host sem endereço" }
        for (address in addresses) {
            require(
                !address.isAnyLocalAddress &&
                    !address.isLoopbackAddress &&
                    !address.isLinkLocalAddress &&
                    !address.isSiteLocalAddress &&
                    !address.isMulticastAddress,
            ) { "Acesso a rede privada bloqueado" }
        }
    }
}

private fun ByteArray.toHex(): String = joinToString("") { byte -> "%02x".format(byte) }
