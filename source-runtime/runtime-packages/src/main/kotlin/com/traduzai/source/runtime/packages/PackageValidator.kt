package com.traduzai.source.runtime.packages

import java.io.ByteArrayInputStream
import java.security.MessageDigest
import java.nio.file.Files
import java.util.jar.JarFile
import java.util.zip.ZipInputStream

enum class PackageKind { JAR, APK }

data class PackageLimits(
    val maxArchiveBytes: Int = 100 * 1024 * 1024,
    val maxUncompressedBytes: Long = 256L * 1024 * 1024,
    val maxEntries: Int = 4096,
    val maxCompressionRatio: Long = 200,
)

data class ValidatedPackage(
    val kind: PackageKind,
    val sha256: String,
    val entries: Int,
    val uncompressedBytes: Long,
)

class PackageException(
    val code: String,
    message: String,
) : IllegalArgumentException(message)

object PackageValidator {
    fun validate(
        archive: ByteArray,
        kind: PackageKind,
        limits: PackageLimits = PackageLimits(),
    ): ValidatedPackage {
        if (archive.isEmpty() || archive.size > limits.maxArchiveBytes) {
            throw PackageException("PACKAGE_SIZE_INVALID", "Pacote vazio ou acima do limite")
        }
        var entries = 0
        var uncompressed = 0L
        val buffer = ByteArray(8192)
        try {
            ZipInputStream(ByteArrayInputStream(archive)).use { zip ->
                while (true) {
                    val entry = zip.nextEntry ?: break
                    entries += 1
                    if (entries > limits.maxEntries) {
                        throw PackageException("ZIP_BOMB", "Pacote excede o limite de entradas")
                    }
                    validateEntryName(entry.name)
                    while (true) {
                        val read = zip.read(buffer)
                        if (read < 0) break
                        uncompressed += read
                        if (uncompressed > limits.maxUncompressedBytes) {
                            throw PackageException("ZIP_BOMB", "Conteúdo descompactado excede o limite")
                        }
                    }
                }
            }
        } catch (error: PackageException) {
            throw error
        } catch (error: Exception) {
            throw PackageException("INVALID_PACKAGE", error.message ?: "ZIP inválido")
        }
        if (entries == 0) throw PackageException("INVALID_PACKAGE", "Pacote sem entradas")
        if (uncompressed > archive.size.toLong() * limits.maxCompressionRatio) {
            throw PackageException("ZIP_BOMB", "Taxa de compressão suspeita")
        }
        return ValidatedPackage(
            kind = kind,
            sha256 = MessageDigest.getInstance("SHA-256").digest(archive).toHex(),
            entries = entries,
            uncompressedBytes = uncompressed,
        )
    }

    fun verifyJarSignature(archive: ByteArray, expectedSignerSha256: String): String {
        val expected = expectedSignerSha256.normalizeFingerprint()
        if (!expected.matches(Regex("[0-9a-f]{64}"))) {
            throw PackageException("REPOSITORY_SIGNER_INVALID", "Fingerprint SHA-256 do assinante é inválida")
        }

        val temporaryJar = Files.createTempFile("traduzai-extension-signature-", ".jar")
        try {
            Files.write(temporaryJar, archive)
            val signerFingerprints = linkedSetOf<String>()
            var signedEntries = 0
            JarFile(temporaryJar.toFile(), true).use { jar ->
                val buffer = ByteArray(8192)
                val entries = jar.entries()
                while (entries.hasMoreElements()) {
                    val entry = entries.nextElement()
                    if (entry.isDirectory || entry.name.startsWith("META-INF/", ignoreCase = true)) continue

                    jar.getInputStream(entry).use { input ->
                        while (input.read(buffer) >= 0) {
                            // Reading the complete entry makes JarFile verify its signature.
                        }
                    }
                    val signers = entry.codeSigners
                    if (signers.isNullOrEmpty()) {
                        throw PackageException("PACKAGE_UNSIGNED", "O JAR contém conteúdo não assinado: ${entry.name}")
                    }
                    signedEntries += 1
                    signers.flatMap { it.signerCertPath.certificates }.forEach { certificate ->
                        signerFingerprints += MessageDigest.getInstance("SHA-256")
                            .digest(certificate.encoded)
                            .toHex()
                    }
                }
            }
            if (signedEntries == 0 || signerFingerprints.isEmpty()) {
                throw PackageException("PACKAGE_UNSIGNED", "O JAR não possui conteúdo assinado")
            }
            if (expected !in signerFingerprints) {
                throw PackageException("PACKAGE_SIGNER_MISMATCH", "O assinante do JAR diverge da chave confiada")
            }
            return expected
        } catch (error: PackageException) {
            throw error
        } catch (error: SecurityException) {
            throw PackageException("PACKAGE_SIGNATURE_INVALID", error.message ?: "Assinatura JAR inválida")
        } catch (error: Exception) {
            throw PackageException("PACKAGE_SIGNATURE_INVALID", error.message ?: "Falha ao validar assinatura JAR")
        } finally {
            Files.deleteIfExists(temporaryJar)
        }
    }

    private fun validateEntryName(name: String) {
        val normalized = name.replace('\\', '/')
        val segments = normalized.split('/')
        if (normalized.startsWith('/') || normalized.matches(Regex("^[A-Za-z]:.*")) || segments.any { it == ".." }) {
            throw PackageException("ZIP_TRAVERSAL", "Entrada ZIP escapa do pacote: $name")
        }
    }
}

private fun ByteArray.toHex(): String = joinToString("") { byte -> "%02x".format(byte) }

private fun String.normalizeFingerprint(): String = lowercase().replace(Regex("[^0-9a-f]"), "")
