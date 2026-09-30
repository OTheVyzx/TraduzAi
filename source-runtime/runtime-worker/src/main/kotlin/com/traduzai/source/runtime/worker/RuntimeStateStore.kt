package com.traduzai.source.runtime.worker

import eu.kanade.tachiyomi.source.model.SChapter
import eu.kanade.tachiyomi.source.model.SManga
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.nio.charset.StandardCharsets
import java.nio.file.Path
import java.sql.Connection
import java.sql.DriverManager
import java.time.Instant
import java.util.UUID

data class StoredRepository(
    val id: String,
    val url: String,
    val certificateFingerprint: String,
    val indexSha256: String,
    val signingKey: String?,
)

data class StoredExtension(
    val packageName: String,
    val name: String,
    val versionCode: Long,
    val versionName: String,
    val repositoryId: String,
    val artifactSha256: String,
    val jarPath: String,
    val enabled: Boolean,
)

class RuntimeStateStore(private val database: Path) : AutoCloseable {
    val dataDirectory: Path = requireNotNull(database.toAbsolutePath().parent)
    private val connection: Connection
    private val json = Json { ignoreUnknownKeys = true }

    init {
        Class.forName("org.sqlite.JDBC")
        connection = DriverManager.getConnection("jdbc:sqlite:${database.toAbsolutePath()}")
        connection.createStatement().use { statement ->
            statement.execute("PRAGMA foreign_keys = ON")
            statement.execute("PRAGMA journal_mode = WAL")
            statement.execute(
                """CREATE TABLE IF NOT EXISTS repositories (
                    id TEXT PRIMARY KEY,
                    url TEXT NOT NULL UNIQUE,
                    certificate_fingerprint TEXT NOT NULL,
                    index_sha256 TEXT NOT NULL,
                    signing_key TEXT
                )""",
            )
            val repositoryColumns = statement.executeQuery("PRAGMA table_info(repositories)").use { rows ->
                buildSet { while (rows.next()) add(rows.getString("name")) }
            }
            if ("signing_key" !in repositoryColumns) {
                statement.execute("ALTER TABLE repositories ADD COLUMN signing_key TEXT")
            }
            statement.execute(
                """CREATE TABLE IF NOT EXISTS manga_state (
                    package_name TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    manga_url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    memo_json TEXT NOT NULL,
                    PRIMARY KEY(package_name, source_id, manga_url)
                )""",
            )
            statement.execute(
                """CREATE TABLE IF NOT EXISTS chapter_state (
                    package_name TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    manga_url TEXT NOT NULL,
                    chapter_url TEXT NOT NULL,
                    name TEXT NOT NULL,
                    memo_json TEXT NOT NULL,
                    PRIMARY KEY(package_name, source_id, chapter_url)
                )""",
            )
            statement.execute(
                """CREATE TABLE IF NOT EXISTS preferences (
                    package_name TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    PRIMARY KEY(package_name, key)
                )""",
            )
            statement.execute(
                """CREATE TABLE IF NOT EXISTS cookies (
                    package_name TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value TEXT NOT NULL,
                    PRIMARY KEY(package_name, domain, name)
                )""",
            )
            statement.execute(
                """CREATE TABLE IF NOT EXISTS installed_extensions (
                    package_name TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version_code INTEGER NOT NULL,
                    version_name TEXT NOT NULL,
                    repository_id TEXT NOT NULL,
                    artifact_sha256 TEXT NOT NULL,
                    jar_path TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1
                )""",
            )
            statement.execute(
                """CREATE TABLE IF NOT EXISTS extension_versions (
                    package_name TEXT NOT NULL,
                    name TEXT NOT NULL,
                    version_code INTEGER NOT NULL,
                    version_name TEXT NOT NULL,
                    repository_id TEXT NOT NULL,
                    artifact_sha256 TEXT NOT NULL,
                    jar_path TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY(package_name, version_code)
                )""",
            )
            statement.execute(
                """CREATE TABLE IF NOT EXISTS reader_library (
                    record_id TEXT PRIMARY KEY,
                    identity_key TEXT NOT NULL UNIQUE,
                    package_name TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    manga_url TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )""",
            )
            statement.execute(
                """CREATE TABLE IF NOT EXISTS reader_progress (
                    manga_id TEXT NOT NULL,
                    chapter_id TEXT NOT NULL,
                    last_page INTEGER NOT NULL DEFAULT 0,
                    page_count INTEGER,
                    read INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(manga_id, chapter_id)
                )""",
            )
            statement.execute(
                """CREATE TABLE IF NOT EXISTS reader_categories (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    position INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )""",
            )
            statement.execute(
                """CREATE TABLE IF NOT EXISTS runtime_settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )""",
            )
            statement.execute(
                """CREATE TABLE IF NOT EXISTS reader_downloads (
                    job_id TEXT PRIMARY KEY,
                    manga_id TEXT NOT NULL,
                    chapter_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(manga_id, chapter_id)
                )""",
            )
        }
    }

    fun saveManga(packageName: String, sourceId: String, manga: SManga) {
        connection.prepareStatement(
            """INSERT INTO manga_state(package_name, source_id, manga_url, title, memo_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(package_name, source_id, manga_url)
                DO UPDATE SET title=excluded.title, memo_json=excluded.memo_json""",
        ).use { query ->
            query.setString(1, packageName)
            query.setString(2, sourceId)
            query.setString(3, manga.url)
            query.setString(4, manga.title)
            query.setString(5, manga.memo.toString())
            query.executeUpdate()
        }
    }

    fun loadManga(packageName: String, sourceId: String, mangaUrl: String): SManga? =
        connection.prepareStatement(
            "SELECT title, memo_json FROM manga_state WHERE package_name=? AND source_id=? AND manga_url=?",
        ).use { query ->
            query.setString(1, packageName)
            query.setString(2, sourceId)
            query.setString(3, mangaUrl)
            query.executeQuery().use { rows ->
                if (!rows.next()) return@use null
                SManga.create().apply {
                    url = mangaUrl
                    title = rows.getString("title")
                    memo = json.parseToJsonElement(rows.getString("memo_json")) as JsonObject
                }
            }
        }

    fun saveChapter(packageName: String, sourceId: String, mangaUrl: String, chapter: SChapter) {
        connection.prepareStatement(
            """INSERT INTO chapter_state(package_name, source_id, manga_url, chapter_url, name, memo_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(package_name, source_id, chapter_url)
                DO UPDATE SET manga_url=excluded.manga_url, name=excluded.name, memo_json=excluded.memo_json""",
        ).use { query ->
            query.setString(1, packageName)
            query.setString(2, sourceId)
            query.setString(3, mangaUrl)
            query.setString(4, chapter.url)
            query.setString(5, chapter.name)
            query.setString(6, chapter.memo.toString())
            query.executeUpdate()
        }
    }

    fun loadChapter(packageName: String, sourceId: String, chapterUrl: String): SChapter? =
        connection.prepareStatement(
            "SELECT name, memo_json FROM chapter_state WHERE package_name=? AND source_id=? AND chapter_url=?",
        ).use { query ->
            query.setString(1, packageName)
            query.setString(2, sourceId)
            query.setString(3, chapterUrl)
            query.executeQuery().use { rows ->
                if (!rows.next()) return@use null
                SChapter.create().apply {
                    url = chapterUrl
                    name = rows.getString("name")
                    memo = json.parseToJsonElement(rows.getString("memo_json")) as JsonObject
                }
            }
        }

    fun putPreference(packageName: String, key: String, value: String) = upsertKeyValue(
        table = "preferences",
        columns = listOf("package_name", "key", "value"),
        values = listOf(packageName, key, value),
        conflict = "package_name, key",
    )

    fun getPreference(packageName: String, key: String): String? = selectValue(
        "SELECT value FROM preferences WHERE package_name=? AND key=?",
        packageName,
        key,
    )

    fun putCookie(packageName: String, domain: String, name: String, value: String) = upsertKeyValue(
        table = "cookies",
        columns = listOf("package_name", "domain", "name", "value"),
        values = listOf(packageName, domain, name, value),
        conflict = "package_name, domain, name",
    )

    fun getCookie(packageName: String, domain: String, name: String): String? = selectValue(
        "SELECT value FROM cookies WHERE package_name=? AND domain=? AND name=?",
        packageName,
        domain,
        name,
    )

    fun saveRepository(
        id: String,
        url: String,
        certificateFingerprint: String,
        indexSha256: String,
        signingKey: String? = null,
    ) {
        connection.prepareStatement(
            """INSERT INTO repositories(id, url, certificate_fingerprint, index_sha256, signing_key)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    certificate_fingerprint=excluded.certificate_fingerprint,
                    index_sha256=excluded.index_sha256,
                    signing_key=excluded.signing_key""",
        ).use { query ->
            query.setString(1, id)
            query.setString(2, url)
            query.setString(3, certificateFingerprint)
            query.setString(4, indexSha256)
            query.setString(5, signingKey)
            query.executeUpdate()
        }
    }

    fun listRepositories(): List<StoredRepository> = connection.prepareStatement(
        "SELECT id, url, certificate_fingerprint, index_sha256, signing_key FROM repositories ORDER BY url",
    ).use { query ->
        query.executeQuery().use { rows ->
            buildList {
                while (rows.next()) {
                    add(StoredRepository(
                        id = rows.getString("id"),
                        url = rows.getString("url"),
                        certificateFingerprint = rows.getString("certificate_fingerprint"),
                        indexSha256 = rows.getString("index_sha256"),
                        signingKey = rows.getString("signing_key"),
                    ))
                }
            }
        }
    }

    fun removeRepository(id: String): Boolean = connection.prepareStatement(
        "DELETE FROM repositories WHERE id=?",
    ).use { query ->
        query.setString(1, id)
        query.executeUpdate() > 0
    }

    fun saveExtension(extension: StoredExtension) {
        connection.prepareStatement(
            """INSERT INTO extension_versions(
                package_name, name, version_code, version_name, repository_id, artifact_sha256, jar_path, enabled
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(package_name, version_code) DO UPDATE SET
                name=excluded.name, version_name=excluded.version_name,
                repository_id=excluded.repository_id, artifact_sha256=excluded.artifact_sha256,
                jar_path=excluded.jar_path, enabled=excluded.enabled""",
        ).use { query ->
            query.bindStoredExtension(extension)
            query.executeUpdate()
        }
        connection.prepareStatement(
            """INSERT INTO installed_extensions(
                package_name, name, version_code, version_name, repository_id, artifact_sha256, jar_path, enabled
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(package_name) DO UPDATE SET
                name=excluded.name,
                version_code=excluded.version_code,
                version_name=excluded.version_name,
                repository_id=excluded.repository_id,
                artifact_sha256=excluded.artifact_sha256,
                jar_path=excluded.jar_path,
                enabled=excluded.enabled""",
        ).use { query ->
            query.bindStoredExtension(extension)
            query.executeUpdate()
        }
    }

    fun getExtension(packageName: String): StoredExtension? = connection.prepareStatement(
        """SELECT package_name, name, version_code, version_name, repository_id,
            artifact_sha256, jar_path, enabled FROM installed_extensions WHERE package_name=?""",
    ).use { query ->
        query.setString(1, packageName)
        query.executeQuery().use { rows -> if (rows.next()) rows.toStoredExtension() else null }
    }

    fun listExtensions(): List<StoredExtension> = connection.prepareStatement(
        """SELECT package_name, name, version_code, version_name, repository_id,
            artifact_sha256, jar_path, enabled FROM installed_extensions ORDER BY name""",
    ).use { query ->
        query.executeQuery().use { rows -> buildList { while (rows.next()) add(rows.toStoredExtension()) } }
    }

    fun listExtensionVersions(packageName: String): List<StoredExtension> = connection.prepareStatement(
        """SELECT package_name, name, version_code, version_name, repository_id,
            artifact_sha256, jar_path, enabled FROM extension_versions WHERE package_name=? ORDER BY version_code DESC""",
    ).use { query ->
        query.setString(1, packageName)
        query.executeQuery().use { rows -> buildList { while (rows.next()) add(rows.toStoredExtension()) } }
    }

    fun setExtensionEnabled(packageName: String, enabled: Boolean): Boolean = connection.prepareStatement(
        "UPDATE installed_extensions SET enabled=? WHERE package_name=?",
    ).use { query ->
        query.setInt(1, if (enabled) 1 else 0)
        query.setString(2, packageName)
        query.executeUpdate() > 0
    }

    fun removeExtension(packageName: String): StoredExtension? {
        val extension = getExtension(packageName) ?: return null
        connection.prepareStatement("DELETE FROM installed_extensions WHERE package_name=?").use { query ->
            query.setString(1, packageName)
            query.executeUpdate()
        }
        connection.prepareStatement("DELETE FROM extension_versions WHERE package_name=?").use { query ->
            query.setString(1, packageName)
            query.executeUpdate()
        }
        return extension
    }

    fun saveReaderManga(manga: JsonObject): JsonObject {
        val packageName = manga.requiredString("extensionPackage")
        val sourceId = manga.requiredString("sourceId")
        val mangaUrl = manga.requiredString("mangaUrl")
        val identity = "$packageName\u0000$sourceId\u0000$mangaUrl"
        val recordId = UUID.nameUUIDFromBytes(identity.toByteArray(StandardCharsets.UTF_8)).toString()
        val normalized = JsonObject(manga + ("id" to JsonPrimitive(recordId)))
        connection.prepareStatement(
            """INSERT INTO reader_library(record_id, identity_key, package_name, source_id, manga_url, payload_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(identity_key) DO UPDATE SET
                    package_name=excluded.package_name,
                    source_id=excluded.source_id,
                    manga_url=excluded.manga_url,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at""",
        ).use { query ->
            query.setString(1, recordId)
            query.setString(2, identity)
            query.setString(3, packageName)
            query.setString(4, sourceId)
            query.setString(5, mangaUrl)
            query.setString(6, normalized.toString())
            query.setString(7, Instant.now().toString())
            query.executeUpdate()
        }
        return normalized
    }

    fun listReaderManga(): List<JsonObject> = connection.prepareStatement(
        "SELECT payload_json FROM reader_library ORDER BY updated_at DESC, record_id",
    ).use { query ->
        query.executeQuery().use { rows ->
            buildList {
                while (rows.next()) add(mergeReaderProgress(json.parseToJsonElement(rows.getString("payload_json")).jsonObject))
            }
        }
    }

    fun removeReaderManga(recordId: String): Boolean = connection.prepareStatement(
        "DELETE FROM reader_library WHERE record_id=?",
    ).use { query ->
        query.setString(1, recordId)
        query.executeUpdate() > 0
    }

    fun saveReaderProgress(
        mangaId: String,
        chapterId: String,
        lastPage: Int,
        pageCount: Int?,
        read: Boolean,
    ): JsonObject {
        val updatedAt = Instant.now().toString()
        connection.prepareStatement(
            """INSERT INTO reader_progress(manga_id, chapter_id, last_page, page_count, read, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(manga_id, chapter_id) DO UPDATE SET
                    last_page=excluded.last_page,
                    page_count=excluded.page_count,
                    read=excluded.read,
                    updated_at=excluded.updated_at""",
        ).use { query ->
            query.setString(1, mangaId)
            query.setString(2, chapterId)
            query.setInt(3, lastPage.coerceAtLeast(0))
            if (pageCount == null) query.setNull(4, java.sql.Types.INTEGER) else query.setInt(4, pageCount.coerceAtLeast(0))
            query.setInt(5, if (read) 1 else 0)
            query.setString(6, updatedAt)
            query.executeUpdate()
        }
        return JsonObject(mapOf(
            "mangaId" to JsonPrimitive(mangaId),
            "chapterId" to JsonPrimitive(chapterId),
            "lastPage" to JsonPrimitive(lastPage.coerceAtLeast(0)),
            "pageCount" to (pageCount?.let(::JsonPrimitive) ?: kotlinx.serialization.json.JsonNull),
            "read" to JsonPrimitive(read),
            "updatedAt" to JsonPrimitive(updatedAt),
        ))
    }

    fun listReaderHistory(): List<JsonObject> = connection.prepareStatement(
        "SELECT manga_id, chapter_id, last_page, page_count, read, updated_at FROM reader_progress ORDER BY updated_at DESC",
    ).use { query ->
        query.executeQuery().use { rows ->
            buildList {
                while (rows.next()) add(JsonObject(mapOf(
                    "mangaId" to JsonPrimitive(rows.getString("manga_id")),
                    "chapterId" to JsonPrimitive(rows.getString("chapter_id")),
                    "lastPage" to JsonPrimitive(rows.getInt("last_page")),
                    "pageCount" to (rows.getObject("page_count")?.let { JsonPrimitive(rows.getInt("page_count")) } ?: kotlinx.serialization.json.JsonNull),
                    "read" to JsonPrimitive(rows.getInt("read") != 0),
                    "updatedAt" to JsonPrimitive(rows.getString("updated_at")),
                )))
            }
        }
    }

    private fun mergeReaderProgress(manga: JsonObject): JsonObject {
        val mangaId = manga["id"]?.jsonPrimitive?.content ?: return manga
        val progress = connection.prepareStatement(
            "SELECT chapter_id, last_page, page_count, read FROM reader_progress WHERE manga_id=?",
        ).use { query ->
            query.setString(1, mangaId)
            query.executeQuery().use { rows ->
                buildMap {
                    while (rows.next()) put(rows.getString("chapter_id"), JsonObject(mapOf(
                        "lastPageRead" to JsonPrimitive(rows.getInt("last_page")),
                        "pageCount" to (rows.getObject("page_count")?.let { JsonPrimitive(rows.getInt("page_count")) } ?: kotlinx.serialization.json.JsonNull),
                        "read" to JsonPrimitive(rows.getInt("read") != 0),
                    )))
                }
            }
        }
        val chapters = manga["chapters"] as? JsonArray ?: return manga
        val merged = JsonArray(chapters.map { element ->
            val chapter = element.jsonObject
            val chapterId = chapter["id"]?.jsonPrimitive?.content
            val saved = chapterId?.let(progress::get)
            if (saved == null) chapter else JsonObject(chapter + saved)
        })
        return JsonObject(manga + ("chapters" to merged))
    }

    fun replaceReaderCategories(categories: JsonArray): List<JsonObject> {
        connection.autoCommit = false
        try {
            connection.createStatement().use { it.executeUpdate("DELETE FROM reader_categories") }
            connection.prepareStatement(
                "INSERT INTO reader_categories(id, name, position, updated_at) VALUES (?, ?, ?, ?)",
            ).use { query ->
                categories.forEachIndexed { index, element ->
                    val category = element.jsonObject
                    query.setString(1, category.requiredString("id"))
                    query.setString(2, category.requiredString("name"))
                    query.setInt(3, category["order"]?.jsonPrimitive?.content?.toIntOrNull() ?: index)
                    query.setString(4, Instant.now().toString())
                    query.addBatch()
                }
                query.executeBatch()
            }
            connection.commit()
        } catch (error: Exception) {
            connection.rollback()
            throw error
        } finally {
            connection.autoCommit = true
        }
        return listReaderCategories()
    }

    fun listReaderCategories(): List<JsonObject> = connection.prepareStatement(
        "SELECT id, name, position FROM reader_categories ORDER BY position, name",
    ).use { query ->
        query.executeQuery().use { rows ->
            buildList {
                while (rows.next()) add(JsonObject(mapOf(
                    "id" to JsonPrimitive(rows.getString("id")),
                    "name" to JsonPrimitive(rows.getString("name")),
                    "order" to JsonPrimitive(rows.getInt("position")),
                )))
            }
        }
    }

    fun saveReaderDownload(download: JsonObject): JsonObject {
        val jobId = download.requiredString("jobId")
        val mangaId = download.requiredString("mangaId")
        val chapterId = download.requiredString("chapterId")
        val status = download.requiredString("status")
        connection.prepareStatement(
            """INSERT INTO reader_downloads(job_id, manga_id, chapter_id, status, payload_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(manga_id, chapter_id) DO UPDATE SET
                    job_id=excluded.job_id, status=excluded.status,
                    payload_json=excluded.payload_json, updated_at=excluded.updated_at""",
        ).use { query ->
            query.setString(1, jobId)
            query.setString(2, mangaId)
            query.setString(3, chapterId)
            query.setString(4, status)
            query.setString(5, download.toString())
            query.setString(6, Instant.now().toString())
            query.executeUpdate()
        }
        return download
    }

    fun listReaderDownloads(): List<JsonObject> = connection.prepareStatement(
        "SELECT payload_json FROM reader_downloads ORDER BY updated_at DESC",
    ).use { query ->
        query.executeQuery().use { rows -> buildList {
            while (rows.next()) add(json.parseToJsonElement(rows.getString("payload_json")).jsonObject)
        } }
    }

    fun removeReaderDownload(jobId: String): Boolean = connection.prepareStatement(
        "DELETE FROM reader_downloads WHERE job_id=?",
    ).use { query ->
        query.setString(1, jobId)
        query.executeUpdate() > 0
    }

    fun putRuntimeSetting(key: String, value: JsonObject) {
        connection.prepareStatement(
            """INSERT INTO runtime_settings(key, value_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at""",
        ).use { query ->
            query.setString(1, key)
            query.setString(2, value.toString())
            query.setString(3, Instant.now().toString())
            query.executeUpdate()
        }
    }

    fun getRuntimeSetting(key: String): JsonObject? = connection.prepareStatement(
        "SELECT value_json FROM runtime_settings WHERE key=?",
    ).use { query ->
        query.setString(1, key)
        query.executeQuery().use { rows ->
            if (rows.next()) json.parseToJsonElement(rows.getString("value_json")).jsonObject else null
        }
    }

    private fun upsertKeyValue(table: String, columns: List<String>, values: List<String>, conflict: String) {
        val placeholders = values.joinToString { "?" }
        connection.prepareStatement(
            "INSERT INTO $table(${columns.joinToString()}) VALUES ($placeholders) " +
                "ON CONFLICT($conflict) DO UPDATE SET value=excluded.value",
        ).use { query ->
            values.forEachIndexed { index, value -> query.setString(index + 1, value) }
            query.executeUpdate()
        }
    }

    private fun selectValue(sql: String, vararg values: String): String? =
        connection.prepareStatement(sql).use { query ->
            values.forEachIndexed { index, value -> query.setString(index + 1, value) }
            query.executeQuery().use { rows -> if (rows.next()) rows.getString("value") else null }
        }

    override fun close() = connection.close()
}

private fun JsonObject.requiredString(name: String): String =
    this[name]?.jsonPrimitive?.content?.takeIf { it.isNotBlank() }
        ?: throw IllegalArgumentException("$name é obrigatório")

private fun java.sql.ResultSet.toStoredExtension() = StoredExtension(
    packageName = getString("package_name"),
    name = getString("name"),
    versionCode = getLong("version_code"),
    versionName = getString("version_name"),
    repositoryId = getString("repository_id"),
    artifactSha256 = getString("artifact_sha256"),
    jarPath = getString("jar_path"),
    enabled = getInt("enabled") != 0,
)

private fun java.sql.PreparedStatement.bindStoredExtension(extension: StoredExtension) {
    setString(1, extension.packageName)
    setString(2, extension.name)
    setLong(3, extension.versionCode)
    setString(4, extension.versionName)
    setString(5, extension.repositoryId)
    setString(6, extension.artifactSha256)
    setString(7, extension.jarPath)
    setInt(8, if (extension.enabled) 1 else 0)
}
