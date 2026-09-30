plugins {
    kotlin("jvm")
    kotlin("plugin.serialization")
    application
}

application {
    mainClass.set("com.traduzai.source.runtime.worker.WorkerMainKt")
}

kotlin {
    jvmToolchain(21)
}

dependencies {
    implementation(project(":mihon-api"))
    implementation(project(":runtime-protocol"))
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.11.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.11.0")
    implementation("org.xerial:sqlite-jdbc:3.49.1.0")

    testImplementation(kotlin("test"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.12.2")
}

evaluationDependsOn(":fixtures:http-source")
val fixtureJar = project(":fixtures:http-source").tasks.named<Jar>("jar")
val fixtureJarPath = fixtureJar.flatMap { it.archiveFile }.get().asFile.absolutePath

tasks.test {
    dependsOn(fixtureJar)
    systemProperty("traduzai.fixture.jar", fixtureJarPath)
}
