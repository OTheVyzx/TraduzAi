plugins {
    kotlin("jvm")
    `java-library`
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    val toolsRoot = providers.environmentVariable("TRADUZAI_SOURCE_RUNTIME_TOOLS")
        .orNull
        ?.let(rootProject::file)
        ?: rootProject.file("../.source-runtime-tools")
    val compatibilityLibs = toolsRoot.resolve("suwayomi-server/server/build/install/server/lib")
    require(compatibilityLibs.isDirectory) {
        "Snapshot de compatibilidade ausente. Execute source-runtime/scripts/bootstrap-windows.ps1."
    }
    api(fileTree(compatibilityLibs) { include("*.jar") })
}
