package com.traduzai.source.runtime.packages

import java.io.ByteArrayOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import java.nio.file.Path
import kotlin.io.path.readBytes
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class PackageValidatorTest {
    @Test
    fun `accepts a bounded jar and returns its hash`() {
        val bytes = zip(mapOf("META-INF/MANIFEST.MF" to "Manifest-Version: 1.0\n", "fixture.class" to "class"))
        val result = PackageValidator.validate(bytes, PackageKind.JAR)
        assertEquals(2, result.entries)
        assertEquals(64, result.sha256.length)
    }

    @Test
    fun `blocks traversal and zip bombs before promotion`() {
        val traversal = zip(mapOf("../outside.class" to "bad"))
        assertFailsWith<PackageException> { PackageValidator.validate(traversal, PackageKind.JAR) }

        val bomb = zip(mapOf("huge.bin" to "0".repeat(1024 * 1024)))
        assertFailsWith<PackageException> {
            PackageValidator.validate(bomb, PackageKind.JAR, PackageLimits(maxUncompressedBytes = 1024))
        }
    }

    @Test
    fun `rejects unsigned jars when a repository signer is pinned`() {
        val unsigned = zip(mapOf("META-INF/MANIFEST.MF" to "Manifest-Version: 1.0\n", "fixture.class" to "class"))

        val error = assertFailsWith<PackageException> {
            PackageValidator.verifyJarSignature(unsigned, "9add655a78e96c4ec7a53ef89dccb557cb5d767489fac5e785d671a5a75d4da2")
        }

        assertEquals("PACKAGE_UNSIGNED", error.code)
    }

    @Test
    fun `rejects malformed pinned signer fingerprints`() {
        val unsigned = zip(mapOf("fixture.class" to "class"))

        val error = assertFailsWith<PackageException> {
            PackageValidator.verifyJarSignature(unsigned, "not-a-sha256")
        }

        assertEquals("REPOSITORY_SIGNER_INVALID", error.code)
    }

    @Test
    fun `verifies an official signed jar when the smoke fixture is provided`() {
        val jar = System.getenv("TRADUZAI_OFFICIAL_SIGNED_JAR") ?: return
        val fingerprint = PackageValidator.verifyJarSignature(
            Path.of(jar).readBytes(),
            "9add655a78e96c4ec7a53ef89dccb557cb5d767489fac5e785d671a5a75d4da2",
        )

        assertEquals("9add655a78e96c4ec7a53ef89dccb557cb5d767489fac5e785d671a5a75d4da2", fingerprint)
    }

    private fun zip(entries: Map<String, String>): ByteArray = ByteArrayOutputStream().use { output ->
        ZipOutputStream(output).use { zip ->
            entries.forEach { (name, value) ->
                zip.putNextEntry(ZipEntry(name))
                zip.write(value.toByteArray())
                zip.closeEntry()
            }
        }
        output.toByteArray()
    }
}
