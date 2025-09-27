# CI Build Fix - LWJGL Native Libraries

## Problem
The CI build is failing with:
```
java.lang.UnsatisfiedLinkError: Failed to locate library: liblwjgl.so
[LWJGL] Platform/architecture mismatch detected for module: org.lwjgl
```

The parent `pom.xml` hardcodes `lwjgl.natives=natives-macos-arm64` which doesn't work on Linux CI (which runs on `linux-x64`).

## Root Cause
- Parent POM line 24: `<lwjgl.natives>natives-macos-arm64</lwjgl.natives>`
- CI runs on Ubuntu Linux x64 (GraalVM Java 24)
- LWJGL can't find Linux natives, only macOS ARM64 natives are configured

## Solution
Add platform detection profiles to the parent `pom.xml` that automatically select the correct LWJGL natives based on the OS and architecture.

## Implementation

Add these profiles to `/Users/hal.hildebrand/git/ART/pom.xml` after the `</dependencyManagement>` section:

```xml
<profiles>
    <!-- LWJGL Native Platform Detection -->
    <profile>
        <id>lwjgl-natives-linux-amd64</id>
        <activation>
            <os>
                <name>Linux</name>
                <arch>amd64</arch>
            </os>
        </activation>
        <properties>
            <lwjgl.natives>natives-linux</lwjgl.natives>
        </properties>
    </profile>

    <profile>
        <id>lwjgl-natives-linux-x86_64</id>
        <activation>
            <os>
                <name>Linux</name>
                <arch>x86_64</arch>
            </os>
        </activation>
        <properties>
            <lwjgl.natives>natives-linux</lwjgl.natives>
        </properties>
    </profile>

    <profile>
        <id>lwjgl-natives-linux-arm64</id>
        <activation>
            <os>
                <name>Linux</name>
                <arch>aarch64</arch>
            </os>
        </activation>
        <properties>
            <lwjgl.natives>natives-linux-arm64</lwjgl.natives>
        </properties>
    </profile>

    <profile>
        <id>lwjgl-natives-windows-amd64</id>
        <activation>
            <os>
                <family>windows</family>
                <arch>amd64</arch>
            </os>
        </activation>
        <properties>
            <lwjgl.natives>natives-windows</lwjgl.natives>
        </properties>
    </profile>

    <profile>
        <id>lwjgl-natives-windows-x86_64</id>
        <activation>
            <os>
                <family>windows</family>
                <arch>x86_64</arch>
            </os>
        </activation>
        <properties>
            <lwjgl.natives>natives-windows</lwjgl.natives>
        </properties>
    </profile>

    <profile>
        <id>lwjgl-natives-macos-x86_64</id>
        <activation>
            <os>
                <name>mac os x</name>
                <arch>x86_64</arch>
            </os>
        </activation>
        <properties>
            <lwjgl.natives>natives-macos</lwjgl.natives>
        </properties>
    </profile>

    <profile>
        <id>lwjgl-natives-macos-aarch64</id>
        <activation>
            <os>
                <name>mac os x</name>
                <arch>aarch64</arch>
            </os>
        </activation>
        <properties>
            <lwjgl.natives>natives-macos-arm64</lwjgl.natives>
        </properties>
    </profile>
</profiles>
```

## Additional Improvements

1. **Remove hardcoded default**: Change line 24 from:
   ```xml
   <lwjgl.natives>natives-macos-arm64</lwjgl.natives>
   ```
   To:
   ```xml
   <!-- LWJGL natives are set by platform detection profiles -->
   <!-- Default fallback if no profile matches -->
   <lwjgl.natives>natives-linux</lwjgl.natives>
   ```

2. **Add LWJGL debug for CI**: In the GitHub Actions workflow, add:
   ```yaml
   - name: Build with Maven
     run: ./mvnw -batch-mode --update-snapshots clean install --file pom.xml
     env:
       MAVEN_OPTS: -Dorg.lwjgl.util.Debug=true -Dorg.lwjgl.util.DebugLoader=true
   ```

3. **Consider adding to Surefire configuration**:
   ```xml
   <plugin>
       <artifactId>maven-surefire-plugin</artifactId>
       <configuration>
           <systemPropertyVariables>
               <java.library.path>${project.build.directory}/natives</java.library.path>
               <org.lwjgl.util.NoChecks>true</org.lwjgl.util.NoChecks>
           </systemPropertyVariables>
       </configuration>
   </plugin>
   ```

## Testing
After applying the fix:
1. Local macOS ARM64 build should still work
2. CI Linux x64 build should pass
3. Tests requiring LWJGL should run successfully

## Alternative Solution
If platform detection continues to cause issues, consider using the LWJGL BOM (Bill of Materials) approach:

```xml
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>org.lwjgl</groupId>
            <artifactId>lwjgl-bom</artifactId>
            <version>${lwjgl.version}</version>
            <scope>import</scope>
            <type>pom</type>
        </dependency>
    </dependencies>
</dependencyManagement>
```

This automatically manages all LWJGL dependencies and their natives.