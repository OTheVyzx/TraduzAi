package com.traduzai.source.runtime.broker

import com.traduzai.source.runtime.protocol.ProtocolError
import com.traduzai.source.runtime.protocol.ProtocolRequest
import com.traduzai.source.runtime.protocol.ProtocolResponse
import com.traduzai.source.runtime.worker.ExtensionLoadException
import com.traduzai.source.runtime.worker.RuntimeStateStore
import com.traduzai.source.runtime.worker.StoredExtension
import com.traduzai.source.runtime.worker.ExtensionLoader
import com.traduzai.source.runtime.catalog.RepositoryClient
import com.traduzai.source.runtime.catalog.RepositoryException
import com.traduzai.source.runtime.packages.PackageException
import com.traduzai.source.runtime.packages.PackageKind
import com.traduzai.source.runtime.packages.PackageValidator
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import java.nio.file.Path
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.UUID

class BrokerDispatcher(
    private val stateStore: RuntimeStateStore? = createStateStore(),
    private val repositoryClient: RepositoryClient = RepositoryClient(),
) {
    suspend fun dispatch(request: ProtocolRequest): ProtocolResponse = try {
        val result = withTimeout(request.deadlineMs) {
            when (request.method) {
                "runtime.hello" -> handshake()
                "repository.preview" -> repositoryPreview(request.params)
                "repository.add" -> repositoryAdd(request.params)
                "repository.list" -> repositoryList()
                "repository.remove" -> repositoryRemove(request.params)
                "repository.refresh" -> repositoryRefresh()
                "extension.install" -> extensionInstall(request.params)
                "extension.list" -> extensionList()
                "extension.set-enabled" -> extensionSetEnabled(request.params)
                "extension.uninstall" -> extensionUninstall(request.params)
                "extension.rollback" -> extensionRollback(request.params)
                "reader.library.list" -> readerLibraryList()
                "reader.library.add",
                "reader.library.update" -> readerLibrarySave(request.params)
                "reader.library.remove" -> readerLibraryRemove(request.params)
                "reader.history" -> readerHistory()
                "reader.progress" -> readerProgress(request.params)
                "reader.categories" -> readerCategories(request.params)
                "reader.automation.get" -> readerAutomationGet()
                "reader.automation.set" -> readerAutomationSet(request.params)
                "reader.automation.run-now" -> readerAutomationRunNow(request)
                "reader.download.list" -> readerDownloadList()
                "reader.download.record" -> readerDownloadRecord(request.params)
                "reader.download.remove" -> readerDownloadRemove(request.params)
                "source.search",
                "source.filters",
                "source.popular",
                "source.latest",
                "source.manga-details",
                "source.chapters",
                "source.pages",
                "source.download-page",
                "source.download-pages" -> workerRequest(request)
                else -> throw DispatchException("METHOD_NOT_FOUND", "Método ${request.method} não suportado")
            }
        }
        ProtocolResponse(id = request.id, ok = true, result = result)
    } catch (_: TimeoutCancellationException) {
        failure(request.id, "SOURCE_TIMEOUT", "A fonte excedeu o prazo", retryable = true)
    } catch (error: ExtensionLoadException) {
        failure(request.id, error.code, error.message ?: "Falha ao carregar extensão")
    } catch (error: RepositoryException) {
        failure(request.id, error.code, error.message ?: "Falha no repositório")
    } catch (error: PackageException) {
        failure(request.id, error.code, error.message ?: "Falha no pacote")
    } catch (error: DispatchException) {
        failure(request.id, error.code, error.message ?: "Falha na requisição", error.retryable, error.retryAfterSeconds)
    } catch (error: Exception) {
        failure(request.id, "SOURCE_FAILURE", error.message ?: "Falha interna da fonte", retryable = true)
    }

    private fun handshake(): JsonObject = buildJsonObject {
        put("protocol", 1)
        put("mihonApi", "1.6")
        put("packages", buildJsonArray { add(kotlinx.serialization.json.JsonPrimitive("jar")) })
        put("apkConversion", false)
        put("sandbox", "required")
        put("webView", false)
        put("workerIsolation", true)
    }

    private suspend fun workerRequest(request: ProtocolRequest): kotlinx.serialization.json.JsonElement {
        val resolvedRequest = if (request.params["jar"] != null) request else {
            val packageName = request.params["packageName"]?.jsonPrimitive?.content
                ?: throw DispatchException("INVALID_REQUEST", "params.packageName é obrigatório")
            val extension = requireStateStore().getExtension(packageName)
                ?: throw DispatchException("EXTENSION_NOT_INSTALLED", "Extensão não instalada")
            if (!extension.enabled) throw DispatchException("EXTENSION_DISABLED", "Extensão desativada")
            request.copy(params = JsonObject(request.params + ("jar" to JsonPrimitive(extension.jarPath))))
        }
        val response = WorkerProcessClient().dispatch(resolvedRequest)
        if (!response.ok) {
            val error = response.error
            throw DispatchException(
                error?.code ?: "SOURCE_FAILURE",
                error?.message ?: "Falha no worker",
                retryable = error?.retryable ?: false,
                retryAfterSeconds = error?.retryAfterSeconds,
            )
        }
        return response.result ?: buildJsonObject { }
    }

    private fun repositoryPreview(params: JsonObject): JsonObject {
        val url = params["url"]?.jsonPrimitive?.content
            ?: throw DispatchException("INVALID_REQUEST", "params.url é obrigatório")
        val preview = repositoryClient.preview(url)
        return buildJsonObject {
            put("url", preview.index.sourceUrl)
            put("certificateFingerprint", preview.certificateFingerprint)
            put("indexSha256", preview.indexSha256)
            preview.index.signingKey?.let { put("signingKey", it) }
            put("extensions", buildJsonArray {
                preview.index.extensions.forEach { extension ->
                    add(buildJsonObject {
                        put("name", extension.name)
                        put("packageName", extension.packageName)
                        put("versionCode", extension.versionCode)
                        put("versionName", extension.versionName)
                        put("lang", extension.lang)
                        put("nsfw", extension.nsfw)
                        extension.iconUrl?.let { put("iconUrl", it) }
                        extension.jarUrl?.let { put("jarUrl", it) }
                        extension.apkUrl?.let { put("apkUrl", it) }
                        extension.artifactSha256?.let { put("sha256", it) }
                        put("sources", buildJsonArray {
                            extension.sources.forEach { source ->
                                add(buildJsonObject {
                                    put("name", source.name)
                                    put("lang", source.lang)
                                    put("id", source.id)
                                    source.baseUrl?.let { put("baseUrl", it) }
                                })
                            }
                        })
                    })
                }
            })
        }
    }

    private fun repositoryAdd(params: JsonObject): JsonObject {
        val url = params["url"]?.jsonPrimitive?.content
            ?: throw DispatchException("INVALID_REQUEST", "params.url é obrigatório")
        val expectedCertificate = params["certificateFingerprint"]?.jsonPrimitive?.content
            ?: throw DispatchException("INVALID_REQUEST", "fingerprint do certificado é obrigatória")
        val expectedIndex = params["indexSha256"]?.jsonPrimitive?.content
            ?: throw DispatchException("INVALID_REQUEST", "hash do índice é obrigatório")
        val expectedSigningKey = params["signingKey"]?.jsonPrimitive?.content
        val preview = repositoryClient.preview(url)
        if (preview.certificateFingerprint != expectedCertificate ||
            preview.indexSha256 != expectedIndex ||
            preview.index.signingKey != expectedSigningKey
        ) {
            throw DispatchException("REPOSITORY_CHANGED", "O repositório mudou desde a prévia; revise novamente")
        }
        val id = UUID.nameUUIDFromBytes(url.toByteArray()).toString()
        requireStateStore().saveRepository(id, url, expectedCertificate, expectedIndex, expectedSigningKey)
        return buildJsonObject {
            put("id", id)
            put("url", url)
            put("certificateFingerprint", expectedCertificate)
            put("indexSha256", expectedIndex)
            expectedSigningKey?.let { put("signingKey", it) }
        }
    }

    private fun repositoryList() = buildJsonObject {
        put("repositories", buildJsonArray {
            requireStateStore().listRepositories().forEach { repository ->
                add(buildJsonObject {
                    put("id", repository.id)
                    put("url", repository.url)
                    put("certificateFingerprint", repository.certificateFingerprint)
                    put("indexSha256", repository.indexSha256)
                    repository.signingKey?.let { put("signingKey", it) }
                })
            }
        })
    }

    private fun repositoryRemove(params: JsonObject) = buildJsonObject {
        val id = params["id"]?.jsonPrimitive?.content
            ?: throw DispatchException("INVALID_REQUEST", "params.id é obrigatório")
        put("removed", requireStateStore().removeRepository(id))
    }

    private fun repositoryRefresh() = buildJsonObject {
        put("extensions", buildJsonArray {
            requireStateStore().listRepositories().forEach { repository ->
                val preview = repositoryClient.preview(repository.url)
                if (preview.certificateFingerprint != repository.certificateFingerprint) {
                    throw DispatchException("REPOSITORY_SIGNER_CHANGED", "O certificado do repositório mudou")
                }
                if (preview.index.signingKey != repository.signingKey) {
                    throw DispatchException("REPOSITORY_SIGNER_CHANGED", "A chave de assinatura do repositório mudou")
                }
                requireStateStore().saveRepository(
                    repository.id,
                    repository.url,
                    repository.certificateFingerprint,
                    preview.indexSha256,
                    repository.signingKey,
                )
                preview.index.extensions.forEach { extension ->
                    add(buildJsonObject {
                        put("repositoryId", repository.id)
                        put("name", extension.name)
                        put("packageName", extension.packageName)
                        put("versionCode", extension.versionCode)
                        put("versionName", extension.versionName)
                        put("lang", extension.lang)
                        put("nsfw", extension.nsfw)
                        extension.iconUrl?.let { put("iconUrl", it) }
                        extension.jarUrl?.let { put("jarUrl", it) }
                        extension.apkUrl?.let { put("apkUrl", it) }
                        extension.artifactSha256?.let { put("sha256", it) }
                        put("sources", buildJsonArray {
                            extension.sources.forEach { source -> add(buildJsonObject {
                                put("id", source.id)
                                put("name", source.name)
                                put("lang", source.lang)
                                source.baseUrl?.let { put("baseUrl", it) }
                            }) }
                        })
                    })
                }
            }
        })
    }

    private fun extensionInstall(params: JsonObject): JsonObject {
        val repositoryId = params["repositoryId"]?.jsonPrimitive?.content
            ?: throw DispatchException("INVALID_REQUEST", "params.repositoryId é obrigatório")
        val packageName = params["packageName"]?.jsonPrimitive?.content
            ?: throw DispatchException("INVALID_REQUEST", "params.packageName é obrigatório")
        val requestedVersion = params["versionCode"]?.jsonPrimitive?.content?.toLongOrNull()
            ?: throw DispatchException("INVALID_REQUEST", "params.versionCode é obrigatório")
        val store = requireStateStore()
        val repository = store.listRepositories().firstOrNull { it.id == repositoryId }
            ?: throw DispatchException("REPOSITORY_NOT_FOUND", "Repositório não encontrado")
        val preview = repositoryClient.preview(repository.url)
        if (preview.certificateFingerprint != repository.certificateFingerprint) {
            throw DispatchException("REPOSITORY_SIGNER_CHANGED", "O certificado do repositório mudou")
        }
        val extension = preview.index.extensions.firstOrNull {
            it.packageName == packageName && it.versionCode == requestedVersion
        } ?: throw DispatchException("EXTENSION_NOT_FOUND", "Extensão ou versão não encontrada")
        val installed = store.getExtension(packageName)
        if (installed != null && installed.versionCode > extension.versionCode) {
            throw DispatchException("DOWNGRADE_BLOCKED", "Downgrade bloqueado")
        }
        val jarUrl = extension.jarUrl
            ?: throw DispatchException("APK_CONVERSION_UNAVAILABLE", "Esta versão exige conversão APK, ainda incompatível")
        val artifact = repositoryClient.downloadArtifact(jarUrl)
        val validated = PackageValidator.validate(artifact.bytes, PackageKind.JAR)
        val expectedSha = extension.artifactSha256?.takeIf { it.matches(Regex("[0-9a-f]{64}")) }
        val expectedSigner = repository.signingKey
        when {
            expectedSha != null && validated.sha256 != expectedSha ->
                throw DispatchException("PACKAGE_HASH_MISMATCH", "SHA-256 do pacote diverge do índice confiável")
            expectedSigner != null ->
                PackageValidator.verifyJarSignature(artifact.bytes, expectedSigner)
            expectedSha == null ->
                throw DispatchException("PACKAGE_TRUST_MISSING", "O repositório não fornece hash nem chave de assinatura")
        }

        val extensionRoot = store.dataDirectory.resolve("extensions").toAbsolutePath().normalize()
        val quarantine = extensionRoot.resolve(".quarantine").resolve(UUID.randomUUID().toString()).normalize()
        if (!quarantine.startsWith(extensionRoot)) throw DispatchException("PATH_ESCAPE", "Destino de quarentena inválido")
        Files.createDirectories(quarantine)
        val quarantinedJar = quarantine.resolve("extension.jar")
        Files.write(quarantinedJar, artifact.bytes)
        val sources = try {
            ExtensionLoader.load(quarantinedJar).use { loaded -> loaded.sources.map { it.id.toString() to it.name } }
        } catch (error: Exception) {
            quarantine.toFile().deleteRecursively()
            throw error
        }
        val activeDirectory = extensionRoot.resolve(packageName).resolve(extension.versionCode.toString()).normalize()
        if (!activeDirectory.startsWith(extensionRoot)) {
            quarantine.toFile().deleteRecursively()
            throw DispatchException("PATH_ESCAPE", "Destino de extensão inválido")
        }
        Files.createDirectories(activeDirectory)
        val activeJar = activeDirectory.resolve("extension.jar")
        Files.move(quarantinedJar, activeJar, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
        quarantine.toFile().deleteRecursively()
        store.saveExtension(StoredExtension(
            packageName = extension.packageName,
            name = extension.name,
            versionCode = extension.versionCode,
            versionName = extension.versionName,
            repositoryId = repositoryId,
            artifactSha256 = validated.sha256,
            jarPath = activeJar.toString(),
            enabled = true,
        ))
        return buildJsonObject {
            put("packageName", extension.packageName)
            put("versionCode", extension.versionCode)
            put("versionName", extension.versionName)
            put("sha256", validated.sha256)
            put("sources", buildJsonArray {
                sources.forEach { (id, name) -> add(buildJsonObject { put("id", id); put("name", name) }) }
            })
        }
    }

    private fun extensionList() = buildJsonObject {
        put("extensions", buildJsonArray {
            val store = requireStateStore()
            store.listExtensions().forEach { extension ->
                add(extension.toJson(store.listExtensionVersions(extension.packageName).any { it.versionCode < extension.versionCode }))
            }
        })
    }

    private fun readerLibraryList() = buildJsonObject {
        put("manga", buildJsonArray { requireStateStore().listReaderManga().forEach(::add) })
    }

    private fun readerLibrarySave(params: JsonObject): JsonObject {
        val manga = params["manga"]?.jsonObject
            ?: throw DispatchException("INVALID_REQUEST", "params.manga é obrigatório")
        val saved = try {
            requireStateStore().saveReaderManga(manga)
        } catch (error: IllegalArgumentException) {
            throw DispatchException("INVALID_REQUEST", error.message ?: "Registro do leitor inválido")
        }
        return buildJsonObject { put("manga", saved) }
    }

    private fun readerLibraryRemove(params: JsonObject) = buildJsonObject {
        val recordId = params["recordId"]?.jsonPrimitive?.content
            ?: throw DispatchException("INVALID_REQUEST", "params.recordId é obrigatório")
        put("removed", requireStateStore().removeReaderManga(recordId))
    }

    private fun readerHistory() = buildJsonObject {
        put("history", buildJsonArray { requireStateStore().listReaderHistory().forEach(::add) })
    }

    private fun readerProgress(params: JsonObject): JsonObject {
        val mangaId = params["mangaId"]?.jsonPrimitive?.content
            ?: throw DispatchException("INVALID_REQUEST", "params.mangaId é obrigatório")
        val chapterId = params["chapterId"]?.jsonPrimitive?.content
            ?: throw DispatchException("INVALID_REQUEST", "params.chapterId é obrigatório")
        val lastPage = params["lastPage"]?.jsonPrimitive?.content?.toIntOrNull()
            ?: throw DispatchException("INVALID_REQUEST", "params.lastPage é obrigatório")
        val pageCount = params["pageCount"]?.jsonPrimitive?.content?.toIntOrNull()
        val read = params["read"]?.jsonPrimitive?.content?.toBooleanStrictOrNull() ?: false
        return buildJsonObject {
            put("progress", requireStateStore().saveReaderProgress(mangaId, chapterId, lastPage, pageCount, read))
        }
    }

    private fun readerCategories(params: JsonObject) = buildJsonObject {
        val categories = params["categories"] as? kotlinx.serialization.json.JsonArray
        val current = if (categories == null) requireStateStore().listReaderCategories()
            else try {
                requireStateStore().replaceReaderCategories(categories)
            } catch (error: IllegalArgumentException) {
                throw DispatchException("INVALID_REQUEST", error.message ?: "Categorias inválidas")
            }
        put("categories", buildJsonArray { current.forEach(::add) })
    }

    private fun readerAutomationGet() = buildJsonObject {
        put("automation", automationSettings())
    }

    private fun readerAutomationSet(params: JsonObject): JsonObject {
        val enabled = params["enabled"]?.jsonPrimitive?.content?.toBooleanStrictOrNull()
            ?: throw DispatchException("INVALID_REQUEST", "params.enabled é obrigatório")
        val intervalHours = params["intervalHours"]?.jsonPrimitive?.content?.toIntOrNull()
            ?: throw DispatchException("INVALID_REQUEST", "params.intervalHours é obrigatório")
        if (intervalHours !in 1..168) {
            throw DispatchException("INVALID_REQUEST", "O intervalo deve ficar entre 1 e 168 horas")
        }
        val current = automationSettings()
        val saved = buildJsonObject {
            put("enabled", enabled)
            put("intervalHours", intervalHours)
            current["lastRunAt"]?.let { put("lastRunAt", it) }
            current["lastRunSummary"]?.let { put("lastRunSummary", it) }
        }
        requireStateStore().putRuntimeSetting(AUTOMATION_SETTINGS_KEY, saved)
        return buildJsonObject { put("automation", saved) }
    }

    private suspend fun readerAutomationRunNow(parentRequest: ProtocolRequest): JsonObject {
        val store = requireStateStore()
        val candidates = store.listReaderManga().filter {
            it["autoUpdate"]?.jsonPrimitive?.content?.toBooleanStrictOrNull() == true
        }
        var updated = 0
        val failures = mutableListOf<JsonObject>()
        candidates.forEach { manga ->
            val mangaId = manga["id"]?.jsonPrimitive?.content ?: "unknown"
            try {
                val params = buildJsonObject {
                    put("packageName", manga.requiredAutomationString("extensionPackage"))
                    put("sourceId", manga.requiredAutomationString("sourceId"))
                    put("manga", buildJsonObject {
                        put("url", manga.requiredAutomationString("mangaUrl"))
                        put("title", manga.requiredAutomationString("title"))
                        manga["thumbnailUrl"]?.let { put("thumbnailUrl", it) }
                    })
                }
                val result = workerRequest(parentRequest.copy(
                    id = "${parentRequest.id}:$mangaId",
                    method = "source.chapters",
                    params = params,
                )).jsonObject
                val remoteChapters = result["chapters"] as? kotlinx.serialization.json.JsonArray
                    ?: throw DispatchException("SOURCE_FAILURE", "A fonte não devolveu capítulos")
                val existingByUrl = (manga["chapters"] as? kotlinx.serialization.json.JsonArray)
                    ?.map { it.jsonObject }
                    ?.associateBy { it["url"]?.jsonPrimitive?.content }
                    .orEmpty()
                val chapters = kotlinx.serialization.json.JsonArray(remoteChapters.map { element ->
                    val chapter = element.jsonObject
                    val url = chapter.requiredAutomationString("url")
                    val existing = existingByUrl[url]
                    JsonObject(chapter + mapOf(
                        "id" to JsonPrimitive(existing?.get("id")?.jsonPrimitive?.content
                            ?: UUID.nameUUIDFromBytes("$mangaId\u0000$url".toByteArray()).toString()),
                        "read" to (existing?.get("read") ?: JsonPrimitive(false)),
                        "bookmarked" to (existing?.get("bookmarked") ?: JsonPrimitive(false)),
                        "lastPageRead" to (existing?.get("lastPageRead") ?: JsonPrimitive(0)),
                        "pageCount" to (existing?.get("pageCount") ?: kotlinx.serialization.json.JsonNull),
                    ))
                })
                store.saveReaderManga(JsonObject(manga + ("chapters" to chapters)))
                updated += 1
            } catch (error: Exception) {
                failures += buildJsonObject {
                    put("mangaId", mangaId)
                    put("message", error.message ?: "Falha ao atualizar a obra")
                }
            }
        }
        val summary = buildJsonObject {
            put("checked", candidates.size)
            put("updated", updated)
            put("failed", failures.size)
            put("failures", buildJsonArray { failures.forEach(::add) })
        }
        val now = java.time.Instant.now().toString()
        val current = automationSettings()
        store.putRuntimeSetting(AUTOMATION_SETTINGS_KEY, buildJsonObject {
            put("enabled", current["enabled"] ?: JsonPrimitive(false))
            put("intervalHours", current["intervalHours"] ?: JsonPrimitive(12))
            put("lastRunAt", now)
            put("lastRunSummary", summary)
        })
        return buildJsonObject {
            put("ranAt", now)
            put("summary", summary)
        }
    }

    private fun automationSettings(): JsonObject = requireStateStore().getRuntimeSetting(AUTOMATION_SETTINGS_KEY)
        ?: buildJsonObject {
            put("enabled", false)
            put("intervalHours", 12)
        }

    private fun readerDownloadList() = buildJsonObject {
        put("downloads", buildJsonArray { requireStateStore().listReaderDownloads().forEach(::add) })
    }

    private fun readerDownloadRecord(params: JsonObject): JsonObject {
        val download = params["download"]?.jsonObject
            ?: throw DispatchException("INVALID_REQUEST", "params.download é obrigatório")
        return buildJsonObject { put("download", requireStateStore().saveReaderDownload(download)) }
    }

    private fun readerDownloadRemove(params: JsonObject) = buildJsonObject {
        val jobId = params["jobId"]?.jsonPrimitive?.content
            ?: throw DispatchException("INVALID_REQUEST", "params.jobId é obrigatório")
        put("removed", requireStateStore().removeReaderDownload(jobId))
    }

    private fun extensionSetEnabled(params: JsonObject) = buildJsonObject {
        val packageName = params["packageName"]?.jsonPrimitive?.content
            ?: throw DispatchException("INVALID_REQUEST", "params.packageName é obrigatório")
        val enabled = params["enabled"]?.jsonPrimitive?.content?.toBooleanStrictOrNull()
            ?: throw DispatchException("INVALID_REQUEST", "params.enabled é obrigatório")
        put("changed", requireStateStore().setExtensionEnabled(packageName, enabled))
    }

    private fun extensionUninstall(params: JsonObject) = buildJsonObject {
        val packageName = params["packageName"]?.jsonPrimitive?.content
            ?: throw DispatchException("INVALID_REQUEST", "params.packageName é obrigatório")
        val store = requireStateStore()
        val removed = store.removeExtension(packageName)
        if (removed != null) {
            val extensionRoot = store.dataDirectory.resolve("extensions").toAbsolutePath().normalize()
            val packageDirectory = Path.of(removed.jarPath).toAbsolutePath().normalize().parent?.parent
                ?: throw DispatchException("PATH_ESCAPE", "Pacote instalado sem diretório válido")
            if (!packageDirectory.startsWith(extensionRoot) || packageDirectory == extensionRoot) {
                throw DispatchException("PATH_ESCAPE", "Pacote instalado fora da raiz permitida")
            }
            packageDirectory.toFile().deleteRecursively()
        }
        put("removed", removed != null)
    }

    private fun extensionRollback(params: JsonObject): JsonObject {
        val packageName = params["packageName"]?.jsonPrimitive?.content
            ?: throw DispatchException("INVALID_REQUEST", "params.packageName é obrigatório")
        val store = requireStateStore()
        val current = store.getExtension(packageName)
            ?: throw DispatchException("EXTENSION_NOT_INSTALLED", "Extensão não instalada")
        val previous = store.listExtensionVersions(packageName).firstOrNull {
            it.versionCode < current.versionCode && Files.isRegularFile(Path.of(it.jarPath))
        } ?: throw DispatchException("ROLLBACK_UNAVAILABLE", "Nenhuma versão anterior validada está disponível")
        ExtensionLoader.load(Path.of(previous.jarPath)).close()
        store.saveExtension(previous.copy(enabled = current.enabled))
        return buildJsonObject {
            put("packageName", previous.packageName)
            put("versionCode", previous.versionCode)
            put("versionName", previous.versionName)
        }
    }

    private fun StoredExtension.toJson(rollbackAvailable: Boolean = false) = buildJsonObject {
        put("packageName", packageName)
        put("name", name)
        put("versionCode", versionCode)
        put("versionName", versionName)
        put("repositoryId", repositoryId)
        put("sha256", artifactSha256)
        put("enabled", enabled)
        put("rollbackAvailable", rollbackAvailable)
    }

    private fun requireStateStore(): RuntimeStateStore = stateStore
        ?: throw DispatchException("RUNTIME_STORAGE_UNAVAILABLE", "Armazenamento do runtime não foi configurado")

    companion object {
        private const val AUTOMATION_SETTINGS_KEY = "reader.automation"

        private fun createStateStore(): RuntimeStateStore? {
            val root = System.getenv("TRADUZAI_SOURCE_RUNTIME_DATA")?.takeIf { it.isNotBlank() } ?: return null
            val directory = Path.of(root).toAbsolutePath().normalize()
            java.nio.file.Files.createDirectories(directory)
            return RuntimeStateStore(directory.resolve("reader.sqlite"))
        }
    }

    private fun failure(id: String, code: String, message: String, retryable: Boolean = false, retryAfterSeconds: Long? = null) = ProtocolResponse(
        id = id,
        ok = false,
        error = ProtocolError(code = code, message = message, retryable = retryable, retryAfterSeconds = retryAfterSeconds),
    )
}

private fun JsonObject.requiredAutomationString(name: String): String =
    this[name]?.jsonPrimitive?.content?.takeIf { it.isNotBlank() }
        ?: throw DispatchException("INVALID_REQUEST", "$name é obrigatório na obra do leitor")

internal class DispatchException(
    val code: String,
    message: String,
    val retryable: Boolean = false,
    val retryAfterSeconds: Long? = null,
) : IllegalArgumentException(message)
