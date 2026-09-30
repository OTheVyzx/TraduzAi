plugins {
    kotlin("jvm")
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    compileOnly(project(":mihon-api"))
    compileOnly("org.jetbrains.kotlinx:kotlinx-serialization-json:1.11.0")
}

tasks.jar {
    manifest {
        attributes["TraduzAI-Source-Class"] = "com.traduzai.fixture.DirectFixtureSource"
        attributes["Implementation-Title"] = "TraduzAI deterministic source fixture"
        attributes["Implementation-Version"] = project.version
    }
}
