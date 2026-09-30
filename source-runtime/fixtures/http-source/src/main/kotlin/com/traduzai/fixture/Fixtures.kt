package com.traduzai.fixture

import eu.kanade.tachiyomi.source.Source
import eu.kanade.tachiyomi.source.SourceFactory
import eu.kanade.tachiyomi.source.model.FilterList
import eu.kanade.tachiyomi.source.model.Filter
import eu.kanade.tachiyomi.source.model.MangasPage
import eu.kanade.tachiyomi.source.model.Page
import eu.kanade.tachiyomi.source.model.SChapter
import eu.kanade.tachiyomi.source.model.SManga
import eu.kanade.tachiyomi.source.model.SMangaUpdate
import eu.kanade.tachiyomi.source.online.HttpSource
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.io.IOException

open class DirectFixtureSource(
    override val id: Long = Long.MAX_VALUE,
    override val name: String = "Fixture PT",
    override val lang: String = "pt-BR",
) : Source {
    override val supportsLatest = true

    override suspend fun getPopularManga(page: Int): MangasPage = result("popular")

    override suspend fun getLatestUpdates(page: Int): MangasPage = result("latest")

    override suspend fun getSearchManga(page: Int, query: String, filters: FilterList): MangasPage {
        val category = (filters.getOrNull(1) as? FixtureSelect)?.state ?: 0
        val completed = (filters.getOrNull(2) as? FixtureCheckBox)?.state ?: false
        return result("$query|category=$category|completed=$completed")
    }

    override fun getFilterList(): FilterList = FilterList(
        Filter.Header("Pesquisa avançada"),
        FixtureSelect("Categoria", arrayOf("Todas", "Ação", "Fantasia")),
        FixtureCheckBox("Somente concluídas"),
    )

    override suspend fun getMangaUpdate(
        manga: SManga,
        chapters: List<SChapter>,
        fetchDetails: Boolean,
        fetchChapters: Boolean,
    ): SMangaUpdate {
        if (fetchDetails && manga.title == "Detalhes parciais") {
            return SMangaUpdate(SManga.create().apply {
                title = "Detalhes parciais completos"
                description = "A fonte devolveu somente os campos atualizados"
            }, chapters)
        }
        if (fetchDetails) {
            manga.description = "Resposta determinística para testes"
            manga.author = "Autora Fixture"
            manga.artist = "Artista Fixture"
            manga.genre = "Ação, Fantasia"
            manga.status = SManga.ONGOING
            manga.initialized = true
            manga.memo = buildJsonObject { put("fixture", "fixture-manga") }
        }
        val resolvedChapters = if (fetchChapters) {
            listOf(SChapter.create().apply {
                url = "/chapter/1"
                name = "Capítulo 1"
                chapter_number = 1F
                memo = buildJsonObject { put("fixture", "fixture-chapter") }
            })
        } else {
            chapters
        }
        return SMangaUpdate(manga, resolvedChapters)
    }

    override suspend fun getPageList(chapter: SChapter): List<Page> = listOf(
        Page(0, "/pages/001", "https://fixture.invalid/pages/001.webp"),
    )

    private fun result(query: String): MangasPage = MangasPage(
        mangas = (if (query.startsWith("large")) 0 until 800 else 0 until 1).map { index -> SManga.create().apply {
            url = if (query.startsWith("large")) "/manga/luna-$index" else "/manga/luna"
            title = if (query.startsWith("large")) "Luna de Teste $index ${"x".repeat(96)}" else "Luna de Teste"
            thumbnail_url = if (query.startsWith("large")) "https://fixture.invalid/covers/luna-$index.webp" else "https://fixture.invalid/covers/luna.webp"
            description = "Busca por $query"
            memo = buildJsonObject { put("search", query) }
        } },
        hasNextPage = false,
    )
}

private class FixtureSelect(name: String, values: Array<String>) : Filter.Select<String>(name, values)
private class FixtureCheckBox(name: String) : Filter.CheckBox(name)

class DynamicFilterFixtureSource : DirectFixtureSource() {
    private var filterRequests = 0
    override fun getFilterList(): FilterList = if (filterRequests++ == 0) {
        FilterList(Filter.Header("Redefinir para carregar"))
    } else {
        FilterList(FixtureSelect("Gênero", arrayOf("Todos", "Ação")))
    }
}

class CloudflareBlockedFixtureSource : DirectFixtureSource() {
    override suspend fun getPopularManga(page: Int): MangasPage {
        throw IOException("Cloudflare bypass currently disabled")
    }
}

class FixtureSourceFactory : SourceFactory {
    override fun createSources(): List<Source> = listOf(
        DirectFixtureSource(),
        DirectFixtureSource(id = Long.MAX_VALUE - 1, name = "Fixture EN", lang = "en"),
    )
}

class DownloadFixtureHttpSource : HttpSource() {
    override val name = "Fixture HTTP"
    override val lang = "pt-BR"
    override val baseUrl = "https://fixture.invalid"
    override val supportsLatest = true

    override suspend fun getPopularManga(page: Int): MangasPage = MangasPage(emptyList(), false)
    override suspend fun getLatestUpdates(page: Int): MangasPage = MangasPage(emptyList(), false)
    override suspend fun getSearchManga(page: Int, query: String, filters: FilterList): MangasPage = MangasPage(emptyList(), false)
    override suspend fun getMangaUpdate(manga: SManga, chapters: List<SChapter>, fetchDetails: Boolean, fetchChapters: Boolean) = SMangaUpdate(manga, chapters)
    override suspend fun getPageList(chapter: SChapter): List<Page> = emptyList()
}

class NotASource
