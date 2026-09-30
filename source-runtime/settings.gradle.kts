pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        mavenCentral()
    }
}

rootProject.name = "traduzai-source-runtime"

include(
    ":runtime-protocol",
    ":mihon-api",
    ":runtime-catalog",
    ":runtime-packages",
    ":runtime-worker",
    ":runtime-broker",
    ":fixtures:http-source",
)
