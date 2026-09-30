plugins { kotlin("jvm") }
kotlin { jvmToolchain(21) }
dependencies {
    testImplementation(kotlin("test"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.12.2")
}
