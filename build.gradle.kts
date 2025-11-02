// Top-level build file where you can add configuration options common to all sub-projects/modules.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.android) apply false
}

// Configure Kotlin JVM toolchain and jvmTarget for all subprojects so Kotlin and Java
// compilation targets remain consistent (useful when importing external modules).
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

subprojects {
    // Configure Java compile options first
    tasks.withType(JavaCompile::class.java).configureEach {
        sourceCompatibility = JavaVersion.VERSION_11.toString()
        targetCompatibility = JavaVersion.VERSION_11.toString()
    }
    
    // Configure compileOptions for Android modules
    plugins.withId("com.android.library") {
        extensions.configure(com.android.build.gradle.LibraryExtension::class.java) {
            compileOptions {
                sourceCompatibility = JavaVersion.VERSION_11
                targetCompatibility = JavaVersion.VERSION_11
            }
        }
    }
    
    // If Kotlin plugin is applied in the subproject, configure its JVM toolchain and kotlinOptions
    plugins.withType(org.jetbrains.kotlin.gradle.plugin.KotlinBasePluginWrapper::class.java) {
        // Configure the Kotlin JVM toolchain (recommended over setting jvmTarget directly)
        try {
            extensions.findByName("kotlin")?.let { kotlinExt ->
                val kotlinJvmExtClass = org.jetbrains.kotlin.gradle.dsl.KotlinJvmProjectExtension::class.java
                if (kotlinExt::class.java == kotlinJvmExtClass || kotlinExt is org.jetbrains.kotlin.gradle.dsl.KotlinJvmProjectExtension) {
                    (kotlinExt as org.jetbrains.kotlin.gradle.dsl.KotlinJvmProjectExtension).jvmToolchain(11)
                } else {
                    // Try to configure via extensions.configure if available
                    try {
                        extensions.configure(kotlinJvmExtClass) {
                            (this as org.jetbrains.kotlin.gradle.dsl.KotlinJvmProjectExtension).jvmToolchain(11)
                        }
                    } catch (_: Exception) {
                        // ignore if not supported
                    }
                }
            }
        } catch (_: Exception) {
            // ignore errors configuring the kotlin extension (graceful fallback)
        }

        // Also set kotlinOptions.jvmTarget as fallback for Kotlin compile tasks
        tasks.withType(KotlinCompile::class.java).configureEach {
            kotlinOptions {
                jvmTarget = "11"
            }
        }
    }
}

// As a final safeguard, after all projects are evaluated force any KotlinCompile task
// and JavaCompile task to use JVM target 11. This overrides any module-local setting.
gradle.projectsEvaluated {
    rootProject.allprojects.forEach { proj ->
        proj.tasks.withType(JavaCompile::class.java).configureEach {
            sourceCompatibility = JavaVersion.VERSION_11.toString()
            targetCompatibility = JavaVersion.VERSION_11.toString()
        }
        
        proj.tasks.withType(org.jetbrains.kotlin.gradle.tasks.KotlinCompile::class.java).configureEach {
            kotlinOptions {
                jvmTarget = "11"
            }
        }
        
        // Also configure Android compileOptions if present
        proj.extensions.findByType(com.android.build.gradle.LibraryExtension::class.java)?.let { android ->
            android.compileOptions {
                sourceCompatibility = JavaVersion.VERSION_11
                targetCompatibility = JavaVersion.VERSION_11
            }
        }
    }
}