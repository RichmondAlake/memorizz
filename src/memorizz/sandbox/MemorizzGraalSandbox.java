/*
 * Reference GraalVM Polyglot wrapper for MemoRizz's strong local sandbox mode.
 * Build this against the matching GraalVM SDK and package it as an executable
 * JAR whose Main-Class is memorizz.sandbox.MemorizzGraalSandbox.
 */
package memorizz.sandbox;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.graalvm.polyglot.Context;
import org.graalvm.polyglot.EnvironmentAccess;
import org.graalvm.polyglot.HostAccess;
import org.graalvm.polyglot.SandboxPolicy;
import org.graalvm.polyglot.Source;
import org.graalvm.polyglot.io.IOAccess;

public final class MemorizzGraalSandbox {
    private MemorizzGraalSandbox() {}

    public static void main(String[] args) throws Exception {
        String policy = "UNTRUSTED";
        String codeFile = null;
        int maxCpuSeconds = 30;
        int maxMemoryMb = 512;
        int maxThreads = 32;
        int maxOutputBytes = 10_000_000;
        for (int index = 0; index < args.length; index++) {
            if ("--sandbox-policy".equals(args[index]) && index + 1 < args.length) {
                policy = args[++index];
            } else if ("--code-file".equals(args[index]) && index + 1 < args.length) {
                codeFile = args[++index];
            } else if ("--max-cpu-seconds".equals(args[index]) && index + 1 < args.length) {
                maxCpuSeconds = Integer.parseInt(args[++index]);
            } else if ("--max-memory-mb".equals(args[index]) && index + 1 < args.length) {
                maxMemoryMb = Integer.parseInt(args[++index]);
            } else if ("--max-threads".equals(args[index]) && index + 1 < args.length) {
                maxThreads = Integer.parseInt(args[++index]);
            } else if ("--max-output-bytes".equals(args[index]) && index + 1 < args.length) {
                maxOutputBytes = Integer.parseInt(args[++index]);
            }
        }
        if (!"UNTRUSTED".equals(policy) || codeFile == null) {
            System.err.println("MemoRizz wrapper requires UNTRUSTED and --code-file");
            System.exit(2);
        }

        maxCpuSeconds = Math.max(1, Math.min(maxCpuSeconds, 86_400));
        maxMemoryMb = Math.max(128, Math.min(maxMemoryMb, 1_048_576));
        maxThreads = Math.max(1, Math.min(maxThreads, 1_024));
        maxOutputBytes = Math.max(1_024, Math.min(maxOutputBytes, 1_000_000_000));

        String code = Files.readString(Path.of(codeFile), StandardCharsets.UTF_8);
        Source source = Source.newBuilder("python", code, "memorizz_input.py").build();
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        ByteArrayOutputStream error = new ByteArrayOutputStream();
        try (Context context = Context.newBuilder("python")
                .sandbox(SandboxPolicy.UNTRUSTED)
                .allowAllAccess(false)
                .allowHostAccess(HostAccess.NONE)
                .allowHostClassLookup(name -> false)
                .allowHostClassLoading(false)
                .allowNativeAccess(false)
                .allowCreateProcess(false)
                .allowEnvironmentAccess(EnvironmentAccess.NONE)
                .allowIO(IOAccess.NONE)
                .in(InputStream.nullInputStream())
                .out(output)
                .err(error)
                .option("sandbox.MaxCPUTime", maxCpuSeconds + "s")
                .option("sandbox.MaxHeapMemory", maxMemoryMb + "MB")
                .option("sandbox.MaxASTDepth", "10000")
                .option("sandbox.MaxStackFrames", "1000")
                .option("sandbox.MaxThreads", Integer.toString(maxThreads))
                .option("sandbox.MaxOutputStreamSize", maxOutputBytes + "B")
                .option("sandbox.MaxErrorStreamSize", maxOutputBytes + "B")
                .build()) {
            context.eval(source);
        }
        System.out.print(output.toString(StandardCharsets.UTF_8));
        System.err.print(error.toString(StandardCharsets.UTF_8));
    }
}
