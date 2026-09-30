package com.traduzai.source.runtime.broker

import com.traduzai.source.runtime.protocol.ProtocolCodec
import com.traduzai.source.runtime.protocol.ProtocolRequest
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import java.nio.file.Path
import java.security.MessageDigest
import com.sun.net.httpserver.HttpServer
import java.net.InetSocketAddress
import com.traduzai.source.runtime.catalog.RepositoryClient
import com.traduzai.source.runtime.worker.RuntimeStateStore
import com.traduzai.source.runtime.worker.StoredExtension
import kotlinx.serialization.json.buildJsonArray
import kotlin.io.path.createTempDirectory
import kotlin.io.path.readBytes
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue

class BrokerDispatcherTest {
    @Test
    fun `handshake declares protocol api package and sandbox capabilities`() = runBlocking {
        val response = BrokerDispatcher().dispatch(
            ProtocolCodec.decodeRequest(
                """{"v":1,"id":"hello-1","method":"runtime.hello","deadlineMs":1000,"params":{}}""",
            ),
        )

        assertTrue(response.ok, response.error?.let { "${it.code}: ${it.message}" })
        val result = requireNotNull(response.result).jsonObject
        assertEquals(1, result["protocol"]?.jsonPrimitive?.content?.toInt())
        assertEquals("1.6", result["mihonApi"]?.jsonPrimitive?.content)
        assertEquals(false, result["webView"]?.jsonPrimitive?.content?.toBoolean())
        assertEquals("required", result["sandbox"]?.jsonPrimitive?.content)
        assertEquals(true, result["workerIsolation"]?.jsonPrimitive?.content?.toBoolean())
    }

    @Test
    fun `searches through a loaded extension without Suwayomi`() = runBlocking {
        val jar = Path.of(requireNotNull(System.getProperty("traduzai.fixture.jar"))).toString().replace("\\", "\\\\")
        val response = BrokerDispatcher().dispatch(
            ProtocolCodec.decodeRequest(
                """{"v":1,"id":"search-1","method":"source.search","deadlineMs":5000,"params":{"jar":"$jar","query":"luna"}}""",
            ),
        )

        assertTrue(response.ok, response.error?.let { "${it.code}: ${it.message}" })
        val manga = requireNotNull(response.result).jsonObject["manga"]!!.jsonArray.single().jsonObject
        assertEquals("Luna de Teste", manga["title"]?.jsonPrimitive?.content)
        assertEquals("9223372036854775807", manga["sourceId"]?.jsonPrimitive?.content)
    }

    @Test
    fun `forwards source owned filters to the isolated worker`() = runBlocking {
        val jar = Path.of(requireNotNull(System.getProperty("traduzai.fixture.jar"))).toString().replace("\\", "\\\\")
        val response = BrokerDispatcher().dispatch(ProtocolCodec.decodeRequest(
            """{"v":1,"id":"filters-1","method":"source.filters","deadlineMs":5000,"params":{"jar":"$jar"}}""",
        ))

        assertTrue(response.ok, response.error?.let { "${it.code}: ${it.message}" })
        val filters = requireNotNull(response.result).jsonObject["filters"]!!.jsonArray
        assertEquals("select", filters[1].jsonObject["type"]?.jsonPrimitive?.content)
    }

    @Test
    fun `drains a worker response larger than the operating system pipe buffer`() = runBlocking {
        val jar = Path.of(requireNotNull(System.getProperty("traduzai.fixture.jar"))).toString().replace("\\", "\\\\")
        val response = BrokerDispatcher().dispatch(
            ProtocolCodec.decodeRequest(
                """{"v":1,"id":"large-1","method":"source.search","deadlineMs":5000,"params":{"jar":"$jar","query":"large"}}""",
            ),
        )

        assertTrue(response.ok, response.error?.let { "${it.code}: ${it.message}" })
        assertEquals(800, requireNotNull(response.result).jsonObject["manga"]!!.jsonArray.size)
    }

    @Test
    fun `keeps manga identity when a source returns partial details`() = runBlocking {
        val jar = Path.of(requireNotNull(System.getProperty("traduzai.fixture.jar"))).toString().replace("\\", "\\\\")
        val response = BrokerDispatcher().dispatch(
            ProtocolCodec.decodeRequest(
                """{"v":1,"id":"details-1","method":"source.manga-details","deadlineMs":5000,"params":{"jar":"$jar","manga":{"url":"/manga/partial","title":"Detalhes parciais"}}}""",
            ),
        )

        assertTrue(response.ok, response.error?.let { "${it.code}: ${it.message}" })
        val manga = requireNotNull(response.result).jsonObject["manga"]!!.jsonObject
        assertEquals("/manga/partial", manga["url"]?.jsonPrimitive?.content)
        assertEquals("Detalhes parciais completos", manga["title"]?.jsonPrimitive?.content)
    }

    @Test
    fun `loads chapter pages in a separate worker process`() = runBlocking {
        val jar = Path.of(requireNotNull(System.getProperty("traduzai.fixture.jar"))).toString().replace("\\", "\\\\")
        val response = BrokerDispatcher().dispatch(
            ProtocolCodec.decodeRequest(
                """{"v":1,"id":"pages-1","method":"source.pages","deadlineMs":5000,"params":{"jar":"$jar","sourceId":"9223372036854775807","chapter":{"url":"/chapter/1","name":"Capítulo 1","memo":{"fixture":"fixture-chapter"}}}}""",
            ),
        )

        assertTrue(response.ok, response.error?.let { "${it.code}: ${it.message}" })
        val pages = requireNotNull(response.result).jsonObject["pages"]!!.jsonArray
        assertEquals("https://fixture.invalid/pages/001.webp", pages.single().jsonObject["imageUrl"]?.jsonPrimitive?.content)
    }

    @Test
    fun `repository preview rejects non HTTPS and private URLs by default`() = runBlocking {
        val response = BrokerDispatcher().dispatch(
            ProtocolCodec.decodeRequest(
                """{"v":1,"id":"repo-1","method":"repository.preview","deadlineMs":1000,"params":{"url":"http://127.0.0.1/index.json"}}""",
            ),
        )

        assertEquals(false, response.ok)
        assertEquals("INSECURE_REPOSITORY", response.error?.code)
    }

    @Test
    fun `repository preview and refresh expose icon and content warning metadata`() = runBlocking {
        val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
        val index = """[{"name":"Fixture","pkg":"ext.fixture","jar":"fixture.jar","icon":"icons/fixture.png","nsfw":2,"code":1,"version":"1.0","sources":[]}]""".toByteArray()
        server.createContext("/index.json") { exchange ->
            exchange.sendResponseHeaders(200, index.size.toLong())
            exchange.responseBody.use { it.write(index) }
        }
        server.start()
        val root = createTempDirectory("traduzai-broker-metadata")
        RuntimeStateStore(root.resolve("reader.sqlite")).use { store ->
            val url = "http://127.0.0.1:${server.address.port}/index.json"
            val dispatcher = BrokerDispatcher(
                stateStore = store,
                repositoryClient = RepositoryClient(allowPrivateHosts = true, allowInsecureHttp = true),
            )
            val preview = dispatcher.dispatch(ProtocolRequest(
                v = 1,
                id = "metadata-preview",
                method = "repository.preview",
                deadlineMs = 5_000,
                params = buildJsonObject { put("url", url) },
            ))
            assertTrue(preview.ok, preview.error?.message)
            val previewExtension = requireNotNull(preview.result).jsonObject["extensions"]!!.jsonArray.single().jsonObject
            assertEquals("http://127.0.0.1:${server.address.port}/icons/fixture.png", previewExtension["iconUrl"]!!.jsonPrimitive.content)
            assertEquals(2, previewExtension["nsfw"]!!.jsonPrimitive.content.toInt())

            store.saveRepository("repo-fixture", url, "UNENCRYPTED", index.sha256())
            val refresh = dispatcher.dispatch(ProtocolRequest(
                v = 1,
                id = "metadata-refresh",
                method = "repository.refresh",
                deadlineMs = 5_000,
                params = buildJsonObject { },
            ))
            assertTrue(refresh.ok, refresh.error?.message)
            val refreshedExtension = requireNotNull(refresh.result).jsonObject["extensions"]!!.jsonArray.single().jsonObject
            assertEquals(previewExtension["iconUrl"], refreshedExtension["iconUrl"])
            assertEquals(previewExtension["nsfw"], refreshedExtension["nsfw"])
        }
        server.stop(0)
    }

    @Test
    fun `installs a hash pinned jar through quarantine and records it`() = runBlocking {
        val jarBytes = Path.of(requireNotNull(System.getProperty("traduzai.fixture.jar"))).readBytes()
        val jarSha = jarBytes.sha256()
        val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
        val index = """[{"name":"Fixture","pkg":"ext.fixture","jar":"fixture.jar","sha256":"$jarSha","code":1,"version":"1.0","sources":[{"name":"Fixture PT","lang":"pt-BR","id":"9223372036854775807"}]}]""".toByteArray()
        server.createContext("/index.json") { exchange ->
            exchange.sendResponseHeaders(200, index.size.toLong())
            exchange.responseBody.use { it.write(index) }
        }
        server.createContext("/fixture.jar") { exchange ->
            exchange.sendResponseHeaders(200, jarBytes.size.toLong())
            exchange.responseBody.use { it.write(jarBytes) }
        }
        server.start()
        val root = createTempDirectory("traduzai-broker-install")
        RuntimeStateStore(root.resolve("reader.sqlite")).use { store ->
            val url = "http://127.0.0.1:${server.address.port}/index.json"
            store.saveRepository("repo-fixture", url, "UNENCRYPTED", index.sha256())
            val dispatcher = BrokerDispatcher(
                stateStore = store,
                repositoryClient = RepositoryClient(allowPrivateHosts = true, allowInsecureHttp = true),
            )
            val response = dispatcher.dispatch(ProtocolCodec.decodeRequest(
                """{"v":1,"id":"install-1","method":"extension.install","deadlineMs":10000,"params":{"repositoryId":"repo-fixture","packageName":"ext.fixture","versionCode":1}}""",
            ))

            assertTrue(response.ok, response.error?.message)
            assertEquals(jarSha, store.listExtensions().single().artifactSha256)
            assertTrue(Path.of(store.listExtensions().single().jarPath).toFile().isFile)
        }
        server.stop(0)
    }

    @Test
    fun `automation settings persist and run-now refreshes opted-in manga`() = runBlocking {
        val fixtureJar = Path.of(requireNotNull(System.getProperty("traduzai.fixture.jar")))
        val root = createTempDirectory("traduzai-broker-automation")
        RuntimeStateStore(root.resolve("reader.sqlite")).use { store ->
            store.saveExtension(StoredExtension(
                packageName = "ext.fixture",
                name = "Fixture",
                versionCode = 1,
                versionName = "1.0",
                repositoryId = "fixture",
                artifactSha256 = fixtureJar.readBytes().sha256(),
                jarPath = fixtureJar.toString(),
                enabled = true,
            ))
            store.saveReaderManga(buildJsonObject {
                put("id", "temporary")
                put("extensionPackage", "ext.fixture")
                put("sourceId", "9223372036854775807")
                put("sourceName", "Fixture PT")
                put("mangaUrl", "/manga/luna")
                put("title", "Luna de Teste")
                put("chapters", buildJsonArray { })
                put("autoUpdate", true)
                put("autoDownload", false)
            })
            val dispatcher = BrokerDispatcher(stateStore = store)
            val saved = dispatcher.dispatch(ProtocolRequest(
                v = 1,
                id = "automation-set",
                method = "reader.automation.set",
                deadlineMs = 5_000,
                params = buildJsonObject { put("enabled", true); put("intervalHours", 12) },
            ))
            assertTrue(saved.ok, saved.error?.message)

            val run = dispatcher.dispatch(ProtocolRequest(
                v = 1,
                id = "automation-run",
                method = "reader.automation.run-now",
                deadlineMs = 10_000,
                params = buildJsonObject { },
            ))
            assertTrue(run.ok, run.error?.message)
            val summary = requireNotNull(run.result).jsonObject["summary"]!!.jsonObject
            assertEquals(1, summary["checked"]!!.jsonPrimitive.content.toInt())
            assertEquals(1, summary["updated"]!!.jsonPrimitive.content.toInt())
            assertTrue(store.listReaderManga().single()["chapters"]!!.jsonArray.isNotEmpty())
            assertEquals(true, store.getRuntimeSetting("reader.automation")!!["enabled"]!!.jsonPrimitive.content.toBoolean())
        }
    }

    @Test
    fun `rolls back only to a previously validated installed jar`() = runBlocking {
        val fixtureJar = Path.of(requireNotNull(System.getProperty("traduzai.fixture.jar")))
        val root = createTempDirectory("traduzai-broker-rollback")
        RuntimeStateStore(root.resolve("reader.sqlite")).use { store ->
            val base = StoredExtension("ext.fixture", "Fixture", 1, "1.0", "fixture", fixtureJar.readBytes().sha256(), fixtureJar.toString(), true)
            store.saveExtension(base)
            store.saveExtension(base.copy(versionCode = 2, versionName = "2.0"))
            val response = BrokerDispatcher(stateStore = store).dispatch(ProtocolRequest(
                v = 1,
                id = "rollback",
                method = "extension.rollback",
                deadlineMs = 10_000,
                params = buildJsonObject { put("packageName", "ext.fixture") },
            ))

            assertTrue(response.ok, response.error?.message)
            assertEquals(1, store.getExtension("ext.fixture")!!.versionCode)
            assertEquals(2, store.listExtensionVersions("ext.fixture").size)
        }
    }

    @Test
    fun `trusts installs and browses an official signed repository when requested`() = runBlocking {
        assumeTrue(System.getenv("TRADUZAI_OFFICIAL_REPOSITORY_SMOKE") == "1")
        val root = createTempDirectory("traduzai-official-repository")
        RuntimeStateStore(root.resolve("reader.sqlite")).use { store ->
            val dispatcher = BrokerDispatcher(stateStore = store)
            val repositoryUrl = "https://raw.githubusercontent.com/keiyoushi/extensions/repo/index.pb"
            val preview = dispatcher.dispatch(ProtocolRequest(
                v = 1,
                id = "official-preview",
                method = "repository.preview",
                deadlineMs = 120_000,
                params = buildJsonObject { put("url", repositoryUrl) },
            ))
            assertTrue(preview.ok, preview.error?.message)
            val previewResult = requireNotNull(preview.result).jsonObject
            val signingKey = requireNotNull(previewResult["signingKey"]?.jsonPrimitive?.content)

            val added = dispatcher.dispatch(ProtocolRequest(
                v = 1,
                id = "official-add",
                method = "repository.add",
                deadlineMs = 120_000,
                params = buildJsonObject {
                    put("url", repositoryUrl)
                    put("certificateFingerprint", previewResult["certificateFingerprint"]!!.jsonPrimitive.content)
                    put("indexSha256", previewResult["indexSha256"]!!.jsonPrimitive.content)
                    put("signingKey", signingKey)
                },
            ))
            assertTrue(added.ok, added.error?.message)
            val repositoryId = requireNotNull(added.result).jsonObject["id"]!!.jsonPrimitive.content

            val refreshed = dispatcher.dispatch(ProtocolRequest(
                v = 1,
                id = "official-refresh",
                method = "repository.refresh",
                deadlineMs = 120_000,
                params = buildJsonObject { },
            ))
            assertTrue(refreshed.ok, refreshed.error?.message)
            val extension = requireNotNull(refreshed.result).jsonObject["extensions"]!!.jsonArray
                .map { it.jsonObject }
                .first { it["name"]?.jsonPrimitive?.content == "Comikey" }

            val installed = dispatcher.dispatch(ProtocolRequest(
                v = 1,
                id = "official-install",
                method = "extension.install",
                deadlineMs = 120_000,
                params = buildJsonObject {
                    put("repositoryId", repositoryId)
                    put("packageName", extension["packageName"]!!.jsonPrimitive.content)
                    put("versionCode", extension["versionCode"]!!.jsonPrimitive.content.toLong())
                },
            ))
            assertTrue(installed.ok, installed.error?.let { "${it.code}: ${it.message}" })

            val source = extension["sources"]!!.jsonArray.map { it.jsonObject }
                .first { it["lang"]?.jsonPrimitive?.content == "pt-BR" }
            val popular = dispatcher.dispatch(ProtocolRequest(
                v = 1,
                id = "official-popular",
                method = "source.popular",
                deadlineMs = 120_000,
                params = buildJsonObject {
                    put("packageName", extension["packageName"]!!.jsonPrimitive.content)
                    put("sourceId", source["id"]!!.jsonPrimitive.content)
                    put("page", 1)
                },
            ))
            assertTrue(popular.ok, popular.error?.let { "${it.code}: ${it.message}" })
            assertTrue(requireNotNull(popular.result).jsonObject["manga"]!!.jsonArray.isNotEmpty())
            assertEquals(signingKey, "9add655a78e96c4ec7a53ef89dccb557cb5d767489fac5e785d671a5a75d4da2")
        }
    }
}

private fun ByteArray.sha256(): String = MessageDigest.getInstance("SHA-256").digest(this)
    .joinToString("") { byte -> "%02x".format(byte) }
