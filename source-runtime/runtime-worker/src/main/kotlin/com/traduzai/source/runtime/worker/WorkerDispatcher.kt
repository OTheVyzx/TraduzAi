package com.traduzai.source.runtime.worker

import com.traduzai.source.runtime.protocol.ProtocolError
import com.traduzai.source.runtime.protocol.ProtocolRequest
import com.traduzai.source.runtime.protocol.ProtocolResponse
import eu.kanade.tachiyomi.source.model.Filter
import eu.kanade.tachiyomi.source.model.FilterList
import eu.kanade.tachiyomi.source.model.Page
import eu.kanade.tachiyomi.source.model.SChapter
import eu.kanade.tachiyomi.source.model.SManga
import eu.kanade.tachiyomi.source.online.HttpSource
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.delay
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import java.io.IOException
import java.nio.file.Path
import java.nio.file.Files
import java.security.MessageDigest

class WorkerDispatcher {
    suspend fun dispatch(request: ProtocolRequest): ProtocolResponse = try {
        val result = withTimeout(request.deadlineMs) {
            when (request.method) {
                "source.search" -> search(request)
                "source.filters" -> filters(request)
                "source.popular" -> catalog(request, latest = false)
                "source.latest" -> catalog(request, latest = true)
                "source.manga-details" -> mangaDetails(request)
                "source.chapters" -> chapters(request)
                "source.pages" -> pages(request)
                "source.download-page" -> downloadPage(request)
                "source.download-pages" -> downloadPages(request)
                else -> throw ExtensionLoadException("METHOD_NOT_FOUND", "Método de worker não suportado")
            }
        }
        ProtocolResponse(id = request.id, ok = true, result = result)
    } catch (_: TimeoutCancellationException) {
        ProtocolResponse(
            id = request.id,
            ok = false,
            error = ProtocolError("SOURCE_TIMEOUT", "A fonte excedeu o prazo", retryable = true),
        )
    } catch (error: ExtensionLoadException) {
        ProtocolResponse(
            id = request.id,
            ok = false,
            error = ProtocolError(
                error.code,
                error.message ?: "Falha na extensão",
                retryable = error.retryable,
                retryAfterSeconds = error.retryAfterSeconds,
            ),
        )
    } catch (error: Exception) {
        ProtocolResponse(
            id = request.id,
            ok = false,
            error = error.toProtocolError(),
        )
    }

    private suspend fun search(request: ProtocolRequest) = withSource(request) { source, params ->
        val query = params["query"]?.jsonPrimitive?.content.orEmpty()
            val filterList = awaitSourceFilters(source)
        params["filters"]?.jsonArray?.let { applyFilterChanges(filterList, it) }
        source.getSearchManga(1, query, filterList).toJson(source.id.toString(), source.name)
    }

    private suspend fun filters(request: ProtocolRequest) = withSource(request) { source, _ ->
        buildJsonObject { put("filters", serializeFilters(awaitSourceFilters(source))) }
    }

    private suspend fun catalog(request: ProtocolRequest, latest: Boolean) = withSource(request) { source, params ->
        val pageNumber = params["page"]?.jsonPrimitive?.content?.toIntOrNull()?.coerceAtLeast(1) ?: 1
        val page = if (latest) source.getLatestUpdates(pageNumber) else source.getPopularManga(pageNumber)
        page.toJson(source.id.toString(), source.name)
    }

    private suspend fun mangaDetails(request: ProtocolRequest) = withSource(request) { source, params ->
        val manga = params.requireObject("manga").toManga()
        val update = source.getMangaUpdate(manga, emptyList(), fetchDetails = true, fetchChapters = false)
        buildJsonObject { put("manga", update.manga.mergeMissingIdentityFrom(manga).toJson()) }
    }

    private suspend fun chapters(request: ProtocolRequest) = withSource(request) { source, params ->
        val manga = params.requireObject("manga").toManga()
        val update = source.getMangaUpdate(manga, emptyList(), fetchDetails = false, fetchChapters = true)
        buildJsonObject {
            put("chapters", buildJsonArray { update.chapters.forEach { add(it.toJson()) } })
        }
    }

    private suspend fun pages(request: ProtocolRequest) = withSource(request) { source, params ->
        val chapter = params.requireObject("chapter").toChapter()
        buildJsonObject {
            put("pages", buildJsonArray {
                source.getPageList(chapter).forEach { page ->
                    add(buildJsonObject {
                        put("index", page.index)
                put("number", page.index + 1)
                        put("url", page.url)
                        page.imageUrl?.let { put("imageUrl", it) }
                    })
                }
            })
        }
    }

    private suspend fun downloadPages(request: ProtocolRequest) = withSource(request) { source, params ->
        val httpSource = source as? HttpSource
            ?: throw ExtensionLoadException("SOURCE_DOWNLOAD_UNSUPPORTED", "A fonte não oferece download HTTP autenticado")
        val chapter = params.requireObject("chapter").toChapter()
        val stagingRoot = params["stagingRoot"]?.jsonPrimitive?.content?.let(Path::of)?.toAbsolutePath()?.normalize()
            ?: throw ExtensionLoadException("INVALID_REQUEST", "params.stagingRoot é obrigatório")
        Files.createDirectories(stagingRoot)
        val pages = source.getPageList(chapter)
        if (pages.isEmpty() || pages.size > MAX_CHAPTER_PAGES) {
            throw ExtensionLoadException("PAGE_COUNT_INVALID", "O capítulo deve ter entre 1 e $MAX_CHAPTER_PAGES páginas")
        }
        var totalBytes = 0L
        val manifest = buildJsonArray {
            pages.sortedBy { it.index }.forEachIndexed { position, page ->
                httpSource.getImage(page).use { response ->
                    if (!response.isSuccessful) {
                        throw ExtensionLoadException("IMAGE_HTTP_ERROR", "HTTP ${response.code} ao baixar a página ${position + 1}")
                    }
                    val body = response.body
                    val declaredLength = body.contentLength()
                    if (declaredLength > MAX_IMAGE_BYTES) {
                        throw ExtensionLoadException("IMAGE_TOO_LARGE", "A página ${position + 1} excede o limite")
                    }
                    val bytes = body.byteStream().use { input ->
                        val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                        val output = java.io.ByteArrayOutputStream()
                        while (true) {
                            val read = input.read(buffer)
                            if (read < 0) break
                            if (output.size() + read > MAX_IMAGE_BYTES) {
                                throw ExtensionLoadException("IMAGE_TOO_LARGE", "A página ${position + 1} excede o limite")
                            }
                            output.write(buffer, 0, read)
                        }
                        output.toByteArray()
                    }
                    if (bytes.isEmpty()) throw ExtensionLoadException("IMAGE_EMPTY", "A página ${position + 1} está vazia")
                    totalBytes += bytes.size
                    if (totalBytes > MAX_CHAPTER_BYTES) {
                        throw ExtensionLoadException("CHAPTER_TOO_LARGE", "O capítulo excede o limite de armazenamento temporário")
                    }
                    val mime = response.header("Content-Type").orEmpty().substringBefore(';').trim().lowercase()
                    val extension = imageExtension(mime, bytes)
                    val relativePath = "page-${(position + 1).toString().padStart(4, '0')}.$extension"
                    val target = stagingRoot.resolve(relativePath).normalize()
                    if (target.parent != stagingRoot) throw ExtensionLoadException("PATH_ESCAPE", "Destino de página inválido")
                    Files.write(target, bytes)
                    add(buildJsonObject {
                        put("relativePath", relativePath)
                        put("mime", mime.ifBlank { mimeFromExtension(extension) })
                        put("size", bytes.size)
                        put("sha256", MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) })
                        put("number", position + 1)
                    })
                }
            }
        }
        buildJsonObject {
            put("pageCount", manifest.size)
            put("totalBytes", totalBytes)
            put("files", manifest)
        }
    }

    private suspend fun downloadPage(request: ProtocolRequest) = withSource(request) { source, params ->
        val httpSource = source as? HttpSource
            ?: throw ExtensionLoadException("SOURCE_DOWNLOAD_UNSUPPORTED", "A fonte não oferece download HTTP autenticado")
        val pageJson = params.requireObject("page")
        val number = params["number"]?.jsonPrimitive?.content?.toIntOrNull()
            ?.takeIf { it in 1..MAX_CHAPTER_PAGES }
            ?: throw ExtensionLoadException("INVALID_REQUEST", "params.number deve identificar uma página válida")
        val index = pageJson["index"]?.jsonPrimitive?.content?.toIntOrNull() ?: number - 1
        val url = pageJson["url"]?.jsonPrimitive?.content.orEmpty()
        val imageUrl = pageJson["imageUrl"]?.jsonPrimitive?.contentOrNull
        val stagingRoot = params["stagingRoot"]?.jsonPrimitive?.content?.let(Path::of)?.toAbsolutePath()?.normalize()
            ?: throw ExtensionLoadException("INVALID_REQUEST", "params.stagingRoot é obrigatório")
        Files.createDirectories(stagingRoot)

        val entry = writePage(httpSource, Page(index, url, imageUrl), stagingRoot, number)
        buildJsonObject {
            put("pageCount", 1)
            put("totalBytes", entry["size"]!!)
            put("files", buildJsonArray { add(entry) })
        }
    }

    private suspend fun writePage(
        source: HttpSource,
        page: Page,
        stagingRoot: Path,
        number: Int,
    ): JsonObject {
        executeImageRequest(source, page).use { response ->
            if (!response.isSuccessful) {
                val retryAfter = response.header("Retry-After")?.trim()?.toLongOrNull()
                throw ExtensionLoadException(
                    "IMAGE_HTTP_ERROR",
                    "HTTP ${response.code} ao baixar a página $number",
                    retryable = response.code == 408 || response.code == 425 || response.code == 429 || response.code >= 500,
                    retryAfterSeconds = retryAfter,
                )
            }
            val body = response.body
            val declaredLength = body.contentLength()
            if (declaredLength > MAX_IMAGE_BYTES) {
                throw ExtensionLoadException("IMAGE_TOO_LARGE", "A página $number excede o limite")
            }
            val bytes = body.byteStream().use { input ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                val output = java.io.ByteArrayOutputStream()
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    if (output.size() + read > MAX_IMAGE_BYTES) {
                        throw ExtensionLoadException("IMAGE_TOO_LARGE", "A página $number excede o limite")
                    }
                    output.write(buffer, 0, read)
                }
                output.toByteArray()
            }
            if (bytes.isEmpty()) throw ExtensionLoadException("IMAGE_EMPTY", "A página $number está vazia")
            val mime = response.header("Content-Type").orEmpty().substringBefore(';').trim().lowercase()
            val extension = imageExtension(mime, bytes)
            val relativePath = "page-${number.toString().padStart(4, '0')}.$extension"
            val target = stagingRoot.resolve(relativePath).normalize()
            if (target.parent != stagingRoot) throw ExtensionLoadException("PATH_ESCAPE", "Destino de página inválido")
            Files.write(target, bytes)
            return buildJsonObject {
                put("relativePath", relativePath)
                put("mime", mime.ifBlank { mimeFromExtension(extension) })
                put("size", bytes.size)
                put("sha256", MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) })
                put("number", number)
            }
        }
    }

    private suspend fun executeImageRequest(source: HttpSource, page: Page) = withContext(Dispatchers.IO) {
        val method = generateSequence(source.javaClass as Class<*>?) { it.superclass }
            .flatMap { type -> type.declaredMethods.asSequence() }
            .firstOrNull { candidate ->
                candidate.name == "imageRequest" && candidate.parameterCount == 1 &&
                    candidate.parameterTypes[0].isAssignableFrom(Page::class.java)
            }
            ?: throw ExtensionLoadException("SOURCE_DOWNLOAD_UNSUPPORTED", "A fonte não expõe a requisição da imagem")
        if (!method.trySetAccessible()) {
            throw ExtensionLoadException("SOURCE_DOWNLOAD_UNSUPPORTED", "A requisição da imagem não pode ser acessada")
        }
        val request = method.invoke(source, page) as? okhttp3.Request
            ?: throw ExtensionLoadException("SOURCE_DOWNLOAD_UNSUPPORTED", "A fonte devolveu uma requisição de imagem inválida")
        source.client.newCall(request).execute()
    }

    private suspend fun withSource(
        request: ProtocolRequest,
        block: suspend (eu.kanade.tachiyomi.source.Source, JsonObject) -> kotlinx.serialization.json.JsonElement,
    ): kotlinx.serialization.json.JsonElement {
        val jar = request.params["jar"]?.jsonPrimitive?.content
            ?: throw ExtensionLoadException("INVALID_REQUEST", "params.jar é obrigatório")
        val requestedSourceId = request.params["sourceId"]?.jsonPrimitive?.contentOrNull
        val entryPoint = request.params["entryPoint"]?.jsonPrimitive?.contentOrNull
        return ExtensionLoader.load(Path.of(jar), entryPoint).use { extension ->
            val source = requestedSourceId?.let { id -> extension.sources.firstOrNull { it.id.toString() == id } }
                ?: extension.sources.first()
            block(source, request.params)
        }
    }
}

private fun Exception.toProtocolError(): ProtocolError {
    val cloudflareChallenge = generateSequence(this as Throwable?) { it.cause }
        .filterIsInstance<IOException>()
        .any { it.message == "Cloudflare bypass currently disabled" }
    return if (cloudflareChallenge) {
        ProtocolError(
            "WEBVIEW_REQUIRED",
            "Esta fonte exige uma verificação Cloudflare que ainda não está disponível no leitor integrado.",
            retryable = false,
        )
    } else {
        ProtocolError("SOURCE_FAILURE", message ?: "Falha da fonte", retryable = true)
    }
}

private suspend fun awaitSourceFilters(source: eu.kanade.tachiyomi.source.Source): FilterList {
    var filters = source.getFilterList()
    repeat(8) {
        val retryHeader = filters.singleOrNull() as? Filter.Header
        val hint = retryHeader?.name?.lowercase().orEmpty()
        if (retryHeader == null || listOf("redefin", "reset", "carreg", "load").none(hint::contains)) return filters
        delay(500)
        filters = source.getFilterList()
    }
    return filters
}

private const val MAX_CHAPTER_PAGES = 500
private const val MAX_IMAGE_BYTES = 25 * 1024 * 1024
private const val MAX_CHAPTER_BYTES = 1024L * 1024 * 1024
private const val MAX_SOURCE_FILTERS = 256

private fun serializeFilters(filters: List<Filter<*>>, prefix: List<Int> = emptyList()): JsonArray = buildJsonArray {
    filters.forEachIndexed { index, filter ->
        val path = prefix + index
        add(buildJsonObject {
            put("path", buildJsonArray { path.forEach { add(JsonPrimitive(it)) } })
            put("name", filter.name)
            when (filter) {
                is Filter.Header -> put("type", "header")
                is Filter.Separator -> put("type", "separator")
                is Filter.Select<*> -> {
                    put("type", "select")
                    put("value", filter.state)
                    put("values", buildJsonArray { filter.displayValues.forEach { add(JsonPrimitive(it)) } })
                }
                is Filter.Text -> { put("type", "text"); put("value", filter.state) }
                is Filter.CheckBox -> { put("type", "checkbox"); put("value", filter.state) }
                is Filter.TriState -> { put("type", "tristate"); put("value", filter.state) }
                is Filter.Sort -> {
                    put("type", "sort")
                    put("values", buildJsonArray { filter.values.forEach { add(JsonPrimitive(it)) } })
                    put("value", filter.state?.let { selection -> buildJsonObject {
                        put("index", selection.index)
                        put("ascending", selection.ascending)
                    } } ?: JsonNull)
                }
                is Filter.Group<*> -> {
                    put("type", "group")
                    @Suppress("UNCHECKED_CAST")
                    put("children", serializeFilters(filter.state.filterIsInstance<Filter<*>>(), path))
                }
                else -> throw ExtensionLoadException("FILTER_UNSUPPORTED", "A fonte expôs um filtro não compatível")
            }
        })
    }
}

private fun applyFilterChanges(filters: List<Filter<*>>, changes: JsonArray) {
    if (changes.size > MAX_SOURCE_FILTERS) throw ExtensionLoadException("INVALID_REQUEST", "Quantidade de filtros inválida")
    changes.forEach { changeElement ->
        val change = changeElement.jsonObject
        val path = change["path"]?.jsonArray?.map { it.jsonPrimitive.content.toIntOrNull() ?: -1 }.orEmpty()
        if (path.isEmpty() || path.size > 8 || path.any { it < 0 }) throw ExtensionLoadException("INVALID_REQUEST", "Caminho de filtro inválido")
        val filter = resolveFilter(filters, path)
        val value = change["value"] ?: JsonNull
        when (filter) {
            is Filter.Select<*> -> filter.state = value.jsonPrimitive.content.toIntOrNull()?.takeIf { it in filter.values.indices }
                ?: throw ExtensionLoadException("INVALID_REQUEST", "Seleção de filtro inválida")
            is Filter.Text -> filter.state = value.jsonPrimitive.content.take(500)
            is Filter.CheckBox -> filter.state = value.jsonPrimitive.content.toBooleanStrictOrNull()
                ?: throw ExtensionLoadException("INVALID_REQUEST", "Valor de filtro inválido")
            is Filter.TriState -> filter.state = value.jsonPrimitive.content.toIntOrNull()?.takeIf { it in 0..2 }
                ?: throw ExtensionLoadException("INVALID_REQUEST", "Estado de filtro inválido")
            is Filter.Sort -> {
                if (value is JsonNull) filter.state = null else {
                    val sort = value.jsonObject
                    val selected = sort["index"]?.jsonPrimitive?.content?.toIntOrNull()?.takeIf { it in filter.values.indices }
                        ?: throw ExtensionLoadException("INVALID_REQUEST", "Ordenação inválida")
                    val ascending = sort["ascending"]?.jsonPrimitive?.content?.toBooleanStrictOrNull()
                        ?: throw ExtensionLoadException("INVALID_REQUEST", "Direção de ordenação inválida")
                    filter.state = Filter.Sort.Selection(selected, ascending)
                }
            }
            else -> throw ExtensionLoadException("INVALID_REQUEST", "Filtro não editável")
        }
    }
}

private fun resolveFilter(filters: List<Filter<*>>, path: List<Int>): Filter<*> {
    var current = filters.getOrNull(path.first()) ?: throw ExtensionLoadException("INVALID_REQUEST", "Filtro inexistente")
    path.drop(1).forEach { index ->
        val group = current as? Filter.Group<*> ?: throw ExtensionLoadException("INVALID_REQUEST", "Grupo de filtro inválido")
        current = group.state.filterIsInstance<Filter<*>>().getOrNull(index)
            ?: throw ExtensionLoadException("INVALID_REQUEST", "Filtro inexistente")
    }
    return current
}

private fun imageExtension(mime: String, bytes: ByteArray): String = when {
    mime == "image/png" || bytes.size >= 8 && bytes.copyOfRange(0, 8).contentEquals(byteArrayOf(0x89.toByte(), 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a)) -> "png"
    mime == "image/jpeg" || bytes.size >= 3 && bytes[0] == 0xff.toByte() && bytes[1] == 0xd8.toByte() && bytes[2] == 0xff.toByte() -> "jpg"
    mime == "image/webp" || bytes.size >= 12 && String(bytes, 0, 4) == "RIFF" && String(bytes, 8, 4) == "WEBP" -> "webp"
    mime == "image/gif" || bytes.size >= 6 && String(bytes, 0, 3) == "GIF" -> "gif"
    mime == "image/avif" || bytes.size >= 12 && String(bytes, 4, 4) == "ftyp" && String(bytes, 8, 4).contains("avif") -> "avif"
    else -> throw ExtensionLoadException("IMAGE_MIME_INVALID", "A resposta não é uma imagem compatível")
}

private fun mimeFromExtension(extension: String) = when (extension) {
    "png" -> "image/png"
    "jpg" -> "image/jpeg"
    "webp" -> "image/webp"
    "gif" -> "image/gif"
    "avif" -> "image/avif"
    else -> "application/octet-stream"
}

private fun eu.kanade.tachiyomi.source.model.MangasPage.toJson(sourceId: String, sourceName: String) = buildJsonObject {
    put("sourceId", sourceId)
    put("sourceName", sourceName)
    put("hasNextPage", hasNextPage)
    put("manga", buildJsonArray {
        mangas.forEach { manga ->
            add(buildJsonObject {
                put("sourceId", sourceId)
                put("url", manga.url)
                put("title", manga.title)
                manga.thumbnail_url?.let { put("thumbnailUrl", it) }
                put("status", manga.status)
                put("memo", manga.memo)
            })
        }
    })
}

private fun JsonObject.requireObject(key: String): JsonObject = this[key]?.jsonObject
    ?: throw ExtensionLoadException("INVALID_REQUEST", "params.$key é obrigatório")

private fun JsonObject.toManga(): SManga = SManga.create().also { manga ->
    manga.url = this["url"]?.jsonPrimitive?.content
        ?: throw ExtensionLoadException("INVALID_REQUEST", "manga.url é obrigatório")
    manga.title = this["title"]?.jsonPrimitive?.content.orEmpty()
    manga.thumbnail_url = this["thumbnailUrl"]?.jsonPrimitive?.contentOrNull
    manga.artist = this["artist"]?.jsonPrimitive?.contentOrNull
    manga.author = this["author"]?.jsonPrimitive?.contentOrNull
    manga.status = this["status"]?.jsonPrimitive?.content?.toIntOrNull() ?: SManga.UNKNOWN
    manga.description = this["description"]?.jsonPrimitive?.contentOrNull
    manga.genre = this["genre"]?.jsonPrimitive?.contentOrNull
    manga.memo = this["memo"]?.jsonObject ?: JsonObject(emptyMap())
}

private fun JsonObject.toChapter(): SChapter = SChapter.create().also { chapter ->
    chapter.url = this["url"]?.jsonPrimitive?.content
        ?: throw ExtensionLoadException("INVALID_REQUEST", "chapter.url é obrigatório")
    chapter.name = this["name"]?.jsonPrimitive?.content.orEmpty()
    chapter.chapter_number = this["chapterNumber"]?.jsonPrimitive?.content?.toFloatOrNull() ?: -1F
    chapter.scanlator = this["scanlator"]?.jsonPrimitive?.contentOrNull
    chapter.date_upload = this["dateUpload"]?.jsonPrimitive?.content?.toLongOrNull() ?: 0L
    chapter.memo = this["memo"]?.jsonObject ?: JsonObject(emptyMap())
}

private fun SManga.mergeMissingIdentityFrom(fallback: SManga): SManga = apply {
    val currentUrl = runCatching { url }.getOrNull()
    if (currentUrl.isNullOrBlank()) url = fallback.url

    val currentTitle = runCatching { title }.getOrNull()
    if (currentTitle.isNullOrBlank()) title = fallback.title

    val currentThumbnail = runCatching { thumbnail_url }.getOrNull()
    if (currentThumbnail.isNullOrBlank()) thumbnail_url = runCatching { fallback.thumbnail_url }.getOrNull()

    val currentMemo = runCatching { memo }.getOrNull() ?: JsonObject(emptyMap())
    val fallbackMemo = runCatching { fallback.memo }.getOrNull() ?: JsonObject(emptyMap())
    memo = JsonObject(fallbackMemo + currentMemo)

    if (status == SManga.UNKNOWN && fallback.status != SManga.UNKNOWN) status = fallback.status
}

private fun SManga.toJson() = buildJsonObject {
    put("url", this@toJson.url)
    put("title", this@toJson.title)
    this@toJson.thumbnail_url?.let { put("thumbnailUrl", it) }
    this@toJson.artist?.let { put("artist", it) }
    this@toJson.author?.let { put("author", it) }
    put("status", this@toJson.status)
    this@toJson.description?.let { put("description", it) }
    this@toJson.genre?.let { put("genre", it) }
    put("initialized", this@toJson.initialized)
    put("memo", this@toJson.memo)
}

private fun SChapter.toJson() = buildJsonObject {
    put("url", this@toJson.url)
    put("name", this@toJson.name)
    put("chapterNumber", this@toJson.chapter_number)
    this@toJson.scanlator?.let { put("scanlator", it) }
    put("dateUpload", this@toJson.date_upload)
    put("memo", this@toJson.memo)
}
