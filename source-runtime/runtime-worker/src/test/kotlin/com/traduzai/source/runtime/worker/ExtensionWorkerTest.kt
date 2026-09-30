package com.traduzai.source.runtime.worker

import eu.kanade.tachiyomi.source.model.FilterList
import eu.kanade.tachiyomi.source.model.SChapter
import eu.kanade.tachiyomi.source.model.SManga
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import java.nio.file.Path
import com.sun.net.httpserver.HttpServer
import java.net.InetSocketAddress
import suwayomi.tachidesk.server.ServerConfig
import xyz.nulldev.ts.config.GlobalConfigManager
import kotlin.io.path.createTempDirectory
import kotlin.io.path.readBytes
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

class ExtensionWorkerTest {
    private val fixtureJar = Path.of(requireNotNull(System.getProperty("traduzai.fixture.jar")))

    @Test
    fun `registers the Suwayomi server config required by network interceptors`() {
        CompatibilityRuntime.ensureStarted()

        assertIs<ServerConfig>(GlobalConfigManager.module(ServerConfig::class.java))
    }

    @Test
    fun `loads direct Source and completes search details chapters pages`() = runBlocking {
        ExtensionLoader.load(fixtureJar).use { extension ->
            val source = extension.sources.single()
            val search = source.getSearchManga(1, "luna", FilterList())
            val manga = search.mangas.single()
            val update = source.getMangaUpdate(manga, emptyList(), fetchDetails = true, fetchChapters = true)
            val pages = source.getPageList(update.chapters.single())

            assertEquals("Luna de Teste", update.manga.title)
            assertEquals("fixture-manga", update.manga.memo["fixture"]?.toString()?.trim('"'))
            assertEquals("fixture-chapter", update.chapters.single().memo["fixture"]?.toString()?.trim('"'))
            assertEquals("https://fixture.invalid/pages/001.webp", pages.single().imageUrl)
        }
    }

    @Test
    fun `loads SourceFactory and exposes every produced source`() {
        ExtensionLoader.load(fixtureJar, "com.traduzai.fixture.FixtureSourceFactory").use { extension ->
            assertEquals(listOf("Fixture PT", "Fixture EN"), extension.sources.map { it.name })
        }
    }

    @Test
    fun `discovers Mihon entrypoint from AndroidManifest xml`() {
        assertEquals(
            "com.traduzai.fixture.DirectFixtureSource",
            ExtensionLoader.discoverEntryPoint(fixtureJar),
        )
    }

    @Test
    fun `loads an official Mihon jar when the smoke fixture is provided`() {
        val officialJar = System.getenv("TRADUZAI_OFFICIAL_SIGNED_JAR") ?: return

        ExtensionLoader.load(Path.of(officialJar)).use { extension ->
            assertTrue(extension.sources.isNotEmpty())
            assertTrue(extension.sources.all { it.id.toString().isNotBlank() && it.name.isNotBlank() })
        }
    }

    @Test
    fun `searches a real title through an official Mihon source when provided`() = runBlocking {
        val officialJar = System.getenv("TRADUZAI_OFFICIAL_SIGNED_JAR") ?: return@runBlocking

        ExtensionLoader.load(Path.of(officialJar)).use { extension ->
            val source = extension.sources.first { it.lang == "pt-BR" }
            val page = source.getPopularManga(1)
            assertTrue(page.mangas.isNotEmpty(), "A fonte oficial não retornou obras populares no smoke real")
            assertTrue(page.mangas.all { it.title.isNotBlank() && it.url.isNotBlank() })
        }
    }

    @Test
    fun `three fixture sources complete the API 1_6 flow`() = runBlocking {
        val direct = ExtensionLoader.load(fixtureJar)
        val factory = ExtensionLoader.load(fixtureJar, "com.traduzai.fixture.FixtureSourceFactory")
        try {
            val sources = direct.sources + factory.sources
            assertEquals(3, sources.size)
            for (source in sources) {
                val manga = source.getSearchManga(1, "luna", FilterList()).mangas.single()
                val update = source.getMangaUpdate(manga, emptyList(), true, true)
                assertEquals(1, source.getPageList(update.chapters.single()).size)
            }
        } finally {
            direct.close()
            factory.close()
        }
    }

    @Test
    fun `downloads one image to staging and returns only a relative manifest`() {
        val png = byteArrayOf(
            0x89.toByte(), 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
            0x00, 0x00, 0x00, 0x0d,
        )
        val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
        server.createContext("/page.png") { exchange ->
            exchange.responseHeaders.add("Content-Type", "image/png")
            exchange.sendResponseHeaders(200, png.size.toLong())
            exchange.responseBody.use { it.write(png) }
        }
        server.start()
        try {
            val staging = createTempDirectory("traduzai-staging")
            val manifest = StagingDownloader(allowPrivateHosts = true).downloadImage(
                "http://127.0.0.1:${server.address.port}/page.png",
                staging,
                "page-001.png",
            )

            assertEquals("page-001.png", manifest.relativePath)
            assertEquals("image/png", manifest.mime)
            assertEquals(png.size.toLong(), manifest.size)
            assertEquals("218ad85a233eff829618a6865ab681222b734c62d35a32b3eabd5c37d8945f86", manifest.sha256)
            assertTrue(manifest.relativePath.none { it == ':' || it == '\\' })
            assertTrue(staging.resolve(manifest.relativePath).readBytes().contentEquals(png))
        } finally {
            server.stop(0)
        }
    }

    @Test
    fun `rejects a class that is not a Source contract`() {
        val error = kotlin.runCatching {
            ExtensionLoader.load(fixtureJar, "com.traduzai.fixture.NotASource")
        }.exceptionOrNull()

        assertIs<ExtensionLoadException>(error)
        assertEquals("UNSUPPORTED_ENTRYPOINT", error.code)
    }

    @Test
    fun `persists manga chapter memo cookies and preferences after reopen`() {
        val database = createTempDirectory("traduzai-runtime-store").resolve("reader.sqlite")
        val manga = SManga.create().apply {
            url = "/manga/luna"
            title = "Luna de Teste"
            memo = buildJsonObject { put("token", "manga-secret") }
        }
        val chapter = SChapter.create().apply {
            url = "/chapter/1"
            name = "Capítulo 1"
            memo = buildJsonObject { put("token", "chapter-secret") }
        }

        RuntimeStateStore(database).use { store ->
            store.saveManga("ext.fixture", "9223372036854775807", manga)
            store.saveChapter("ext.fixture", "9223372036854775807", manga.url, chapter)
            store.putPreference("ext.fixture", "quality", "original")
            store.putCookie("ext.fixture", "fixture.invalid", "session", "abc123")
            store.saveRepository("repo-1", "https://repo.invalid/index.json", "cert-sha", "index-sha", "signer-sha")
            store.saveExtension(StoredExtension(
                packageName = "ext.fixture",
                name = "Fixture",
                versionCode = 1,
                versionName = "1.0",
                repositoryId = "repo-1",
                artifactSha256 = "artifact-sha",
                jarPath = "extensions/ext.fixture/1/extension.jar",
                enabled = true,
            ))
        }

        RuntimeStateStore(database).use { store ->
            assertEquals("manga-secret", store.loadManga("ext.fixture", "9223372036854775807", manga.url)?.memo?.get("token")?.toString()?.trim('"'))
            assertEquals("chapter-secret", store.loadChapter("ext.fixture", "9223372036854775807", chapter.url)?.memo?.get("token")?.toString()?.trim('"'))
            assertEquals("original", store.getPreference("ext.fixture", "quality"))
            assertEquals("abc123", store.getCookie("ext.fixture", "fixture.invalid", "session"))
            assertEquals("cert-sha", store.listRepositories().single().certificateFingerprint)
            assertEquals("signer-sha", store.listRepositories().single().signingKey)
            assertEquals("artifact-sha", store.listExtensions().single().artifactSha256)
            assertTrue(database.toFile().length() > 0)
        }
    }

    @Test
    fun `dispatcher exposes details chapters and pages while preserving memo`() = runBlocking {
        val escapedJar = fixtureJar.toString().replace("\\", "\\\\")
        val dispatcher = WorkerDispatcher()
        val details = dispatcher.dispatch(com.traduzai.source.runtime.protocol.ProtocolCodec.decodeRequest(
            """{"v":1,"id":"details","method":"source.manga-details","deadlineMs":5000,"params":{"jar":"$escapedJar","sourceId":"9223372036854775807","manga":{"url":"/manga/luna","title":"Luna","memo":{"fixture":"fixture-manga"}}}}""",
        ))
        assertTrue(details.ok)
        val detailedManga = requireNotNull(details.result).jsonObject["manga"]!!.jsonObject
        assertEquals("Luna", detailedManga["title"]?.jsonPrimitive?.content)
        assertEquals("Resposta determinística para testes", detailedManga["description"]?.jsonPrimitive?.content)
        assertEquals("Autora Fixture", detailedManga["author"]?.jsonPrimitive?.content)
        assertEquals("Artista Fixture", detailedManga["artist"]?.jsonPrimitive?.content)
        assertEquals("Ação, Fantasia", detailedManga["genre"]?.jsonPrimitive?.content)
        assertEquals(1, detailedManga["status"]?.jsonPrimitive?.content?.toInt())
        assertEquals("fixture-manga", detailedManga["memo"]?.jsonObject?.get("fixture")?.jsonPrimitive?.content)

        val chapters = dispatcher.dispatch(com.traduzai.source.runtime.protocol.ProtocolCodec.decodeRequest(
            """{"v":1,"id":"chapters","method":"source.chapters","deadlineMs":5000,"params":{"jar":"$escapedJar","sourceId":"9223372036854775807","manga":{"url":"/manga/luna","title":"Luna","memo":{"fixture":"fixture-manga"}}}}""",
        ))
        val chapter = requireNotNull(chapters.result).jsonObject["chapters"]!!.jsonArray.single().jsonObject
        assertEquals("fixture-chapter", chapter["memo"]?.jsonObject?.get("fixture")?.jsonPrimitive?.content)

        val pages = dispatcher.dispatch(com.traduzai.source.runtime.protocol.ProtocolCodec.decodeRequest(
            """{"v":1,"id":"pages","method":"source.pages","deadlineMs":5000,"params":{"jar":"$escapedJar","sourceId":"9223372036854775807","chapter":{"url":"/chapter/1","name":"Capítulo 1","memo":{"fixture":"fixture-chapter"}}}}""",
        ))
        val page = requireNotNull(pages.result).jsonObject["pages"]!!.jsonArray.single().jsonObject
        assertEquals("https://fixture.invalid/pages/001.webp", page["imageUrl"]?.jsonPrimitive?.content)
    }

    @Test
    fun `dispatcher exposes popular and latest source catalogs`() = runBlocking {
        val escapedJar = fixtureJar.toString().replace("\\", "\\\\")
        for (method in listOf("source.popular", "source.latest")) {
            val response = WorkerDispatcher().dispatch(com.traduzai.source.runtime.protocol.ProtocolCodec.decodeRequest(
                """{"v":1,"id":"catalog","method":"$method","deadlineMs":5000,"params":{"jar":"$escapedJar","sourceId":"9223372036854775807","page":1}}""",
            ))

            assertTrue(response.ok, response.error?.message)
            val result = requireNotNull(response.result).jsonObject
            assertEquals("Fixture PT", result["sourceName"]?.jsonPrimitive?.content)
            assertTrue(result["manga"]!!.jsonArray.isNotEmpty())
        }
    }

    @Test
    fun `dispatcher exposes and applies source filters`() = runBlocking {
        val escapedJar = fixtureJar.toString().replace("\\", "\\\\")
        val filters = WorkerDispatcher().dispatch(com.traduzai.source.runtime.protocol.ProtocolCodec.decodeRequest(
            """{"v":1,"id":"filters","method":"source.filters","deadlineMs":5000,"params":{"jar":"$escapedJar","sourceId":"9223372036854775807"}}""",
        ))

        assertTrue(filters.ok, filters.error?.message)
        val descriptors = requireNotNull(filters.result).jsonObject["filters"]!!.jsonArray
        assertEquals(listOf("header", "select", "checkbox"), descriptors.map { it.jsonObject["type"]!!.jsonPrimitive.content })
        assertEquals("Fantasia", descriptors[1].jsonObject["values"]!!.jsonArray[2].jsonPrimitive.content)

        val search = WorkerDispatcher().dispatch(com.traduzai.source.runtime.protocol.ProtocolCodec.decodeRequest(
            """{"v":1,"id":"search-filtered","method":"source.search","deadlineMs":5000,"params":{"jar":"$escapedJar","sourceId":"9223372036854775807","query":"luna","filters":[{"path":[1],"value":2},{"path":[2],"value":true}]}}""",
        ))
        assertTrue(search.ok, search.error?.message)
        val manga = requireNotNull(search.result).jsonObject["manga"]!!.jsonArray.single().jsonObject
        assertEquals("luna|category=2|completed=true", manga["memo"]?.jsonObject?.get("search")?.jsonPrimitive?.content)
    }

    @Test
    fun `dispatcher waits for dynamically loaded source filters`() = runBlocking {
        val escapedJar = fixtureJar.toString().replace("\\", "\\\\")
        val filters = WorkerDispatcher().dispatch(com.traduzai.source.runtime.protocol.ProtocolCodec.decodeRequest(
            """{"v":1,"id":"dynamic-filters","method":"source.filters","deadlineMs":5000,"params":{"jar":"$escapedJar","entryPoint":"com.traduzai.fixture.DynamicFilterFixtureSource"}}""",
        ))

        assertTrue(filters.ok, filters.error?.message)
        assertEquals("select", requireNotNull(filters.result).jsonObject["filters"]!!.jsonArray.single().jsonObject["type"]!!.jsonPrimitive.content)
    }

    @Test
    fun `cloudflare challenge returns a stable localized capability error`() = runBlocking {
        val escapedJar = fixtureJar.toString().replace("\\", "\\\\")
        val response = WorkerDispatcher().dispatch(com.traduzai.source.runtime.protocol.ProtocolCodec.decodeRequest(
            """{"v":1,"id":"cloudflare","method":"source.popular","deadlineMs":5000,"params":{"jar":"$escapedJar","entryPoint":"com.traduzai.fixture.CloudflareBlockedFixtureSource"}}""",
        ))

        assertEquals(false, response.ok)
        assertEquals("WEBVIEW_REQUIRED", response.error?.code)
        assertEquals("Esta fonte exige uma verificação Cloudflare que ainda não está disponível no leitor integrado.", response.error?.message)
        assertEquals(false, response.error?.retryable)
    }

    @Test
    fun `dispatcher downloads one authenticated source page for a resumable queue`() = runBlocking {
        val png = byteArrayOf(
            0x89.toByte(), 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
            0x00, 0x00, 0x00, 0x0d,
        )
        val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
        server.createContext("/page.png") { exchange ->
            exchange.responseHeaders.add("Content-Type", "image/png")
            exchange.sendResponseHeaders(200, png.size.toLong())
            exchange.responseBody.use { it.write(png) }
        }
        server.start()
        try {
            val staging = createTempDirectory("traduzai-download-page")
            val escapedJar = fixtureJar.toString().replace("\\", "\\\\")
            val escapedStaging = staging.toString().replace("\\", "\\\\")
            val response = WorkerDispatcher().dispatch(com.traduzai.source.runtime.protocol.ProtocolCodec.decodeRequest(
                """{"v":1,"id":"download-page","method":"source.download-page","deadlineMs":5000,"params":{"jar":"$escapedJar","entryPoint":"com.traduzai.fixture.DownloadFixtureHttpSource","page":{"index":0,"url":"/page/1","imageUrl":"http://127.0.0.1:${server.address.port}/page.png"},"number":1,"stagingRoot":"$escapedStaging"}}""",
            ))

            assertTrue(response.ok, response.error?.message)
            val result = requireNotNull(response.result).jsonObject
            val file = result["files"]!!.jsonArray.single().jsonObject
            assertEquals(1, file["number"]?.jsonPrimitive?.content?.toInt())
            assertEquals("page-0001.png", file["relativePath"]?.jsonPrimitive?.content)
            assertTrue(staging.resolve("page-0001.png").readBytes().contentEquals(png))
        } finally {
            server.stop(0)
        }
    }

    @Test
    fun `download page exposes retry after from a rate limited source`() = runBlocking {
        val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
        server.createContext("/limited.png") { exchange ->
            exchange.responseHeaders.add("Retry-After", "7")
            exchange.sendResponseHeaders(429, -1)
            exchange.close()
        }
        server.start()
        try {
            val staging = createTempDirectory("traduzai-download-rate-limit")
            val escapedJar = fixtureJar.toString().replace("\\", "\\\\")
            val escapedStaging = staging.toString().replace("\\", "\\\\")
            val response = WorkerDispatcher().dispatch(com.traduzai.source.runtime.protocol.ProtocolCodec.decodeRequest(
                """{"v":1,"id":"download-limited","method":"source.download-page","deadlineMs":5000,"params":{"jar":"$escapedJar","entryPoint":"com.traduzai.fixture.DownloadFixtureHttpSource","page":{"index":0,"url":"/page/1","imageUrl":"http://127.0.0.1:${server.address.port}/limited.png"},"number":1,"stagingRoot":"$escapedStaging"}}""",
            ))

            assertEquals(false, response.ok)
            assertEquals("IMAGE_HTTP_ERROR", response.error?.code)
            assertEquals(true, response.error?.retryable)
            assertEquals(7, response.error?.retryAfterSeconds)
        } finally {
            server.stop(0)
        }
    }

    @Test
    fun `reader library progress history and categories persist in sqlite`() {
        val database = createTempDirectory("traduzai-reader-state").resolve("reader.sqlite")
        val manga = buildJsonObject {
            put("id", "temporary-client-id")
            put("extensionPackage", "ext.fixture")
            put("sourceId", "9223372036854775807")
            put("sourceName", "Fixture PT")
            put("mangaUrl", "/manga/luna")
            put("title", "Luna de Teste")
            put("chapters", buildJsonArray { })
        }
        val firstId: String
        RuntimeStateStore(database).use { store ->
            val saved = store.saveReaderManga(manga)
            firstId = saved["id"]!!.jsonPrimitive.content
            assertTrue(firstId != "temporary-client-id")
            assertEquals(firstId, store.saveReaderManga(manga)["id"]!!.jsonPrimitive.content)
            store.saveReaderProgress(firstId, "chapter-1", 3, 12, false)
            store.replaceReaderCategories(buildJsonArray {
                add(buildJsonObject { put("id", "favorites"); put("name", "Favoritos"); put("order", 0) })
            })
            store.putRuntimeSetting("reader.automation", buildJsonObject {
                put("enabled", true)
                put("intervalHours", 12)
            })
            store.saveReaderDownload(buildJsonObject {
                put("jobId", "download-fixture")
                put("mangaId", firstId)
                put("chapterId", "chapter-1")
                put("status", "completed")
                put("pages", buildJsonArray { })
            })
        }

        RuntimeStateStore(database).use { store ->
            assertEquals(firstId, store.listReaderManga().single()["id"]!!.jsonPrimitive.content)
            assertEquals(3, store.listReaderHistory().single()["lastPage"]!!.jsonPrimitive.content.toInt())
            assertEquals("Favoritos", store.listReaderCategories().single()["name"]!!.jsonPrimitive.content)
            assertEquals(true, store.getRuntimeSetting("reader.automation")!!["enabled"]!!.jsonPrimitive.content.toBoolean())
            assertEquals("download-fixture", store.listReaderDownloads().single()["jobId"]!!.jsonPrimitive.content)
            assertTrue(store.removeReaderDownload("download-fixture"))
            assertTrue(store.removeReaderManga(firstId))
        }
    }
}
