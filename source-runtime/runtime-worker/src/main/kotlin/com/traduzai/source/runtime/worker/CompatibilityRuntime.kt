package com.traduzai.source.runtime.worker

import eu.kanade.tachiyomi.App
import eu.kanade.tachiyomi.createAppModule
import org.koin.core.context.GlobalContext
import org.koin.core.context.startKoin
import xyz.nulldev.androidcompat.AndroidCompat
import xyz.nulldev.androidcompat.AndroidCompatInitializer
import xyz.nulldev.androidcompat.androidCompatModule
import xyz.nulldev.ts.config.GlobalConfigManager
import xyz.nulldev.ts.config.configManagerModule
import suwayomi.tachidesk.server.ServerConfig
import suwayomi.tachidesk.server.util.ConfigTypeRegistration
import java.nio.file.Files
import java.nio.file.Path

/**
 * Initializes only the compatibility services required by Mihon extensions.
 * No Suwayomi HTTP/GraphQL server is started.
 */
object CompatibilityRuntime {
    @Volatile
    private var started = false

    @Synchronized
    fun ensureStarted() {
        if (started) return

        val dataRoot = System.getenv("TRADUZAI_SOURCE_RUNTIME_DATA")
            ?.takeIf { it.isNotBlank() }
            ?.let(Path::of)
            ?: Path.of(System.getProperty("java.io.tmpdir"), "traduzai-source-runtime")
        val compatibilityRoot = dataRoot.toAbsolutePath().normalize().resolve("android-compat")
        Files.createDirectories(compatibilityRoot)
        System.setProperty("suwayomi.tachidesk.config.server.rootDir", compatibilityRoot.toString())

        ConfigTypeRegistration.registerCustomTypes()
        GlobalConfigManager.registerModule(
            ServerConfig.register { GlobalConfigManager.config },
        )

        if (GlobalContext.getOrNull() == null) {
            val app = App()
            startKoin {
                modules(
                    createAppModule(app),
                    androidCompatModule(),
                    configManagerModule(),
                )
            }
            AndroidCompatInitializer().init()
            AndroidCompat().startApp(app)
        }
        started = true
    }
}
