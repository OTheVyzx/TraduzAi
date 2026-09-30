plugins {
    kotlin("jvm")
    kotlin("plugin.serialization")
    application
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    implementation(project(":runtime-protocol"))
    implementation(project(":runtime-worker"))
    implementation(project(":mihon-api"))
    implementation(project(":runtime-catalog"))
    implementation(project(":runtime-packages"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.11.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.11.0")

    testImplementation(kotlin("test"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.12.2")
}

application {
    mainClass.set("com.traduzai.source.runtime.broker.BrokerMainKt")
}

tasks.named<Sync>("installDist") {
    // The pinned Suwayomi distribution already contains a few byte-identical
    // Kotlin runtime jars that Gradle also resolves for this application.
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
}

evaluationDependsOn(":fixtures:http-source")
val fixtureJar = project(":fixtures:http-source").tasks.named<Jar>("jar")
val fixtureJarPath = fixtureJar.flatMap { it.archiveFile }.get().asFile.absolutePath

tasks.test {
    dependsOn(fixtureJar)
    systemProperty("traduzai.fixture.jar", fixtureJarPath)
}
