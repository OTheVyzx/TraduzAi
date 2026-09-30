package com.traduzai.source.runtime.worker

import eu.kanade.tachiyomi.source.Source
import eu.kanade.tachiyomi.source.SourceFactory
import java.net.URLClassLoader
import java.nio.file.Path
import java.util.jar.JarFile
import javax.xml.parsers.DocumentBuilderFactory

class ExtensionLoadException(
    val code: String,
    message: String,
    cause: Throwable? = null,
    val retryable: Boolean = false,
    val retryAfterSeconds: Long? = null,
) : RuntimeException(message, cause)

class LoadedExtension internal constructor(
    private val classLoader: URLClassLoader,
    val sources: List<Source>,
) : AutoCloseable {
    override fun close() = classLoader.close()
}

object ExtensionLoader {
    fun discoverEntryPoint(jar: Path): String {
        if (!jar.toFile().isFile) {
            throw ExtensionLoadException("PACKAGE_NOT_FOUND", "JAR de extensão não encontrado")
        }
        JarFile(jar.toFile()).use { archive ->
            archive.manifest?.mainAttributes?.getValue("TraduzAI-Source-Class")?.let { return it }
            val manifest = archive.getJarEntry("AndroidManifest.xml")
                ?: throw ExtensionLoadException("ENTRYPOINT_MISSING", "JAR não contém AndroidManifest.xml")
            try {
                val factory = DocumentBuilderFactory.newInstance().apply {
                    isNamespaceAware = true
                    setFeature("http://apache.org/xml/features/disallow-doctype-decl", true)
                    setFeature("http://xml.org/sax/features/external-general-entities", false)
                    setFeature("http://xml.org/sax/features/external-parameter-entities", false)
                    setAttribute("http://javax.xml.XMLConstants/property/accessExternalDTD", "")
                    setAttribute("http://javax.xml.XMLConstants/property/accessExternalSchema", "")
                }
                val document = archive.getInputStream(manifest).use { factory.newDocumentBuilder().parse(it) }
                val nodes = document.getElementsByTagName("meta-data")
                for (index in 0 until nodes.length) {
                    val attributes = nodes.item(index).attributes ?: continue
                    val name = attributes.getNamedItemNS("http://schemas.android.com/apk/res/android", "name")
                        ?.nodeValue ?: attributes.getNamedItem("android:name")?.nodeValue
                    if (name != "tachiyomi.extension.class") continue
                    val value = attributes.getNamedItemNS("http://schemas.android.com/apk/res/android", "value")
                        ?.nodeValue ?: attributes.getNamedItem("android:value")?.nodeValue
                    if (!value.isNullOrBlank()) return value.trim()
                }
            } catch (error: ExtensionLoadException) {
                throw error
            } catch (error: Exception) {
                throw ExtensionLoadException("MANIFEST_INVALID", "AndroidManifest.xml inválido", error)
            }
        }
        throw ExtensionLoadException("ENTRYPOINT_MISSING", "AndroidManifest.xml não declara tachiyomi.extension.class")
    }

    fun load(jar: Path, entryPointOverride: String? = null): LoadedExtension {
        if (!jar.toFile().isFile) {
            throw ExtensionLoadException("PACKAGE_NOT_FOUND", "JAR de extensão não encontrado")
        }
        val entryPoint = entryPointOverride ?: discoverEntryPoint(jar)

        CompatibilityRuntime.ensureStarted()
        val loader = URLClassLoader(arrayOf(jar.toUri().toURL()), Source::class.java.classLoader)
        try {
            val instance = loader.loadClass(entryPoint).getDeclaredConstructor().newInstance()
            val sources = when (instance) {
                is Source -> listOf(instance)
                is SourceFactory -> instance.createSources()
                else -> throw ExtensionLoadException(
                    "UNSUPPORTED_ENTRYPOINT",
                    "$entryPoint não implementa Source nem SourceFactory",
                )
            }
            if (sources.isEmpty()) {
                throw ExtensionLoadException("EMPTY_SOURCE_FACTORY", "$entryPoint não produziu fontes")
            }
            return LoadedExtension(loader, sources)
        } catch (error: ExtensionLoadException) {
            loader.close()
            throw error
        } catch (error: ReflectiveOperationException) {
            loader.close()
            throw ExtensionLoadException("ENTRYPOINT_LOAD_FAILED", "Falha ao instanciar $entryPoint", error)
        }
    }
}
