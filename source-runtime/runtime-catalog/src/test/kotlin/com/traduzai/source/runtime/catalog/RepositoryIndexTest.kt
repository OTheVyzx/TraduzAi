package com.traduzai.source.runtime.catalog

import com.sun.net.httpserver.HttpServer
import java.net.InetSocketAddress
import java.util.Base64
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class RepositoryIndexTest {
    @Test
    fun `parses gzip protobuf v2 indexes published by Keiyoushi`() {
        // Generated once from Keiyoushi's public index.proto. Keeping fixed wire bytes here
        // avoids making the test pass merely because the production encoder and decoder agree.
        val payload = Base64.getDecoder().decode(
            "H4sIAAAAAAACCuPi9E7NrMwvLc7IlOKwTExJMTM1TVzFVs9Vy8XulllRUlqUKsSdWlGilwbhSElzcUOZeokF2avk4bysxCIlZkM9M42Je9iMWIEMPUMLRiczjv9QUC/EBTVQISBEirWgRNcpSEk8o6SkoNhKXx9mSmZeWWJOZgoAwyqKjJcAAAA="
        )

        val index = RepositoryIndex.parse(payload, "https://repo.invalid/index.pb")

        assertEquals("Keiyoushi", index.name)
        assertEquals("9add655a", index.signingKey)
        assertEquals("ext.fixture", index.extensions.single().packageName)
        assertEquals("9223372036854775807", index.extensions.single().sources.single().id)
        assertEquals("https://repo.invalid/fixture.jar", index.extensions.single().jarUrl)
    }

    @Test
    fun `parses proto3 JSON indexes`() {
        val index = RepositoryIndex.parse(
            """{"name":"Keiyoushi","signingKey":"9add655a","extensionList":{"extensions":[{"name":"Fixture","packageName":"ext.fixture","resources":{"apkUrl":"fixture.apk","jarUrl":"fixture.jar","iconUrl":"icons/fixture.png"},"extensionLib":"1.6","versionCode":"106001","versionName":"1.6.1","contentWarning":"CONTENT_WARNING_NSFW","sources":[{"id":"9223372036854775807","name":"Fixture PT","language":"pt-BR","homeUrl":"https://fixture.invalid"}]}]}}""".toByteArray(),
            "https://repo.invalid/index.json",
        )

        assertEquals("Keiyoushi", index.name)
        assertEquals("9add655a", index.signingKey)
        assertEquals(106001, index.extensions.single().versionCode)
        assertEquals("pt-BR", index.extensions.single().sources.single().lang)
        assertEquals("9223372036854775807", index.extensions.single().sources.single().id)
        assertEquals("https://repo.invalid/icons/fixture.png", index.extensions.single().iconUrl)
        assertEquals(2, index.extensions.single().nsfw)
    }

    @Test
    fun `parses Keiyoushi JSON while preserving source Long ids`() {
        val index = RepositoryIndex.parse(
            """[{"name":"Fixture","pkg":"ext.fixture","apk":"fixture.apk","jarUrl":"fixture.jar","icon":"icons/fixture.png","sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","lang":"pt-BR","code":42,"version":"1.4.42","nsfw":1,"sources":[{"name":"Fixture PT","lang":"pt-BR","id":"9223372036854775807","baseUrl":"https://fixture.invalid"}]}]""",
            "https://repo.invalid/index.min.json",
        )

        assertEquals("ext.fixture", index.extensions.single().packageName)
        assertEquals("9223372036854775807", index.extensions.single().sources.single().id)
        assertEquals("https://repo.invalid/fixture.jar", index.extensions.single().jarUrl)
        assertEquals("https://repo.invalid/fixture.apk", index.extensions.single().apkUrl)
        assertEquals("https://repo.invalid/icons/fixture.png", index.extensions.single().iconUrl)
        assertEquals(1, index.extensions.single().nsfw)
        assertEquals(64, index.extensions.single().artifactSha256?.length)
    }

    @Test
    fun `rejects non web repository URLs and oversized indexes`() {
        assertFailsWith<RepositoryException> { RepositoryIndex.parse("[]".toByteArray(), "file:///tmp/index.json") }
        assertFailsWith<RepositoryException> {
            RepositoryIndex.parse(" ".repeat(5 * 1024 * 1024 + 1).toByteArray(), "https://repo.invalid/index.json")
        }
    }

    @Test
    fun `repository preview is fail closed for private networks unless explicitly permitted`() {
        val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
        val payload = """[{"name":"Fixture","pkg":"ext.fixture","jar":"fixture.jar","code":1,"version":"1.0","sources":[]}]"""
        server.createContext("/index.json") { exchange ->
            exchange.sendResponseHeaders(200, payload.toByteArray().size.toLong())
            exchange.responseBody.use { it.write(payload.toByteArray()) }
        }
        server.start()
        val url = "http://127.0.0.1:${server.address.port}/index.json"
        try {
            assertFailsWith<RepositoryException> { RepositoryClient().preview(url) }
            val preview = RepositoryClient(allowPrivateHosts = true, allowInsecureHttp = true).preview(url)
            assertEquals("UNENCRYPTED", preview.certificateFingerprint)
            assertEquals("ext.fixture", preview.index.extensions.single().packageName)
            assertEquals(64, preview.indexSha256.length)
        } finally {
            server.stop(0)
        }
    }
}
