/*
 * Copyright (c) 2025 Hal Hildebrand. All rights reserved.
 * 
 * This file is part of HART-CQ System.
 * 
 * HART-CQ is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * HART-CQ is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with HART-CQ. If not, see <https://www.gnu.org/licenses/>.
 */
package com.hellblazer.art.hartcq.benchmark;

import com.hellblazer.art.hartcq.integration.HARTCQ;
import com.hellblazer.art.hartcq.integration.ProcessingResult;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

/**
 * JMH Benchmark for HART-CQ Sentence Processing Performance
 * 
 * Target: >100 sentences/second throughput
 * 
 * This benchmark measures:
 * - Single sentence processing throughput
 * - Batch sentence processing throughput
 * - Multi-channel coordination overhead
 * - Template-bounded generation performance
 * - End-to-end pipeline performance
 * 
 * @author Hal Hildebrand
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Benchmark)
@Fork(value = 2, jvmArgs = {"-Xms2G", "-Xmx2G", "--enable-preview"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 2)
public class HARTCQSentenceBenchmark {

    private HARTCQ hartcq;
    private List<String> sampleSentences;
    private List<String> shortSentences;
    private List<String> mediumSentences;
    private List<String> longSentences;
    private List<String> complexSentences;

    @Param({"10", "100", "1000"})
    private int sentenceCount;

    @Setup(Level.Trial)
    public void setup() {
        hartcq = new HARTCQ();
        generateSampleSentences();
    }

    @TearDown(Level.Trial)
    public void teardown() {
        if (hartcq != null) {
            // Clean up resources if needed
        }
    }

    private void generateSampleSentences() {
        var random = new Random(42); // Deterministic for reproducibility
        
        // Short sentences (5-10 tokens)
        shortSentences = List.of(
            "The cat sat on the mat.",
            "It was a beautiful morning.",
            "She walked to the store.",
            "The sun was shining brightly.",
            "He opened the door slowly."
        );

        // Medium sentences (15-25 tokens)
        mediumSentences = List.of(
            "The quick brown fox jumps over the lazy dog while the sun sets behind the mountains.",
            "In the heart of the city, where skyscrapers touch the clouds, life moves at an incredible pace.",
            "She carefully examined the ancient manuscript, noting every detail with her magnifying glass.",
            "The conference room was filled with executives discussing the quarterly financial results.",
            "As the storm approached, the villagers hurried to secure their homes and gather supplies."
        );

        // Long sentences (30-50 tokens)
        longSentences = List.of(
            "Despite the challenging economic conditions and unprecedented global events that had unfolded over the past year, the company managed to maintain profitability through innovative strategies, cost-cutting measures, and a dedicated workforce that adapted quickly to remote work environments.",
            "The archaeological team, led by Professor Johnson and comprising experts from various fields including geology, anthropology, and history, discovered what appeared to be an ancient civilization's trading post, complete with well-preserved artifacts, intricate wall paintings, and a complex system of underground tunnels.",
            "When the new legislation was finally passed after months of heated debate and numerous amendments, it fundamentally changed how businesses operate in the digital space, requiring them to implement stricter data protection measures, obtain explicit user consent for data collection, and provide transparent reports on their data handling practices."
        );

        // Complex sentences with nested structures
        complexSentences = List.of(
            "The committee, which had been formed to investigate the allegations that were raised by the whistleblower who had worked in the department for over ten years, concluded that while there were indeed some procedural violations, they did not constitute criminal activity.",
            "Having considered all the evidence presented, including the testimony of expert witnesses, the documentary proof submitted by both parties, and the circumstances surrounding the incident, the judge ruled that the defendant, despite having acted with good intentions, was nonetheless liable for damages.",
            "The research paper, published in the prestigious journal after undergoing rigorous peer review by leading experts in the field who had initially expressed skepticism about the methodology, demonstrated that the new approach could significantly improve outcomes in patients with chronic conditions."
        );

        // Build the full sample set
        sampleSentences = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            switch (i % 4) {
                case 0 -> sampleSentences.add(shortSentences.get(random.nextInt(shortSentences.size())));
                case 1 -> sampleSentences.add(mediumSentences.get(random.nextInt(mediumSentences.size())));
                case 2 -> sampleSentences.add(longSentences.get(random.nextInt(longSentences.size())));
                case 3 -> sampleSentences.add(complexSentences.get(random.nextInt(complexSentences.size())));
            }
        }
    }

    /**
     * Benchmark single sentence processing throughput
     * Target: >100 sentences/second
     */
    @Benchmark
    public void measureSingleSentenceThroughput(Blackhole blackhole) {
        var sentence = sampleSentences.get(0);
        var result = hartcq.process(sentence);
        blackhole.consume(result);
    }

    /**
     * Benchmark batch sentence processing
     * Tests efficiency of batch processing vs individual
     */
    @Benchmark
    public void measureBatchSentenceThroughput(Blackhole blackhole) {
        var batch = sampleSentences.subList(0, Math.min(sentenceCount, sampleSentences.size()));
        var results = hartcq.processBatch(batch);
        blackhole.consume(results);
    }

    /**
     * Benchmark short sentence processing
     * Should be fastest due to fewer tokens
     */
    @Benchmark
    public void measureShortSentenceThroughput(Blackhole blackhole) {
        for (var sentence : shortSentences) {
            var result = hartcq.process(sentence);
            blackhole.consume(result);
        }
    }

    /**
     * Benchmark medium sentence processing
     * Typical use case
     */
    @Benchmark
    public void measureMediumSentenceThroughput(Blackhole blackhole) {
        for (var sentence : mediumSentences) {
            var result = hartcq.process(sentence);
            blackhole.consume(result);
        }
    }

    /**
     * Benchmark long sentence processing
     * Tests performance with more tokens
     */
    @Benchmark
    public void measureLongSentenceThroughput(Blackhole blackhole) {
        for (var sentence : longSentences) {
            var result = hartcq.process(sentence);
            blackhole.consume(result);
        }
    }

    /**
     * Benchmark complex sentence processing
     * Tests performance with nested structures
     */
    @Benchmark
    public void measureComplexSentenceThroughput(Blackhole blackhole) {
        for (var sentence : complexSentences) {
            var result = hartcq.process(sentence);
            blackhole.consume(result);
        }
    }

    /**
     * Benchmark sliding window processing
     * Tests the 20-token window mechanism
     */
    @Benchmark
    public void measureSlidingWindowThroughput(Blackhole blackhole) {
        // Process a long text that requires multiple windows
        var longText = String.join(" ", mediumSentences) + " " + String.join(" ", longSentences);
        var result = hartcq.process(longText);
        blackhole.consume(result);
    }

    /**
     * Benchmark parallel processing capabilities
     * Tests concurrent sentence processing
     */
    @Benchmark
    @Threads(4)
    public void measureParallelSentenceThroughput(Blackhole blackhole) {
        var sentence = sampleSentences.get((int) (Thread.currentThread().getId() % sampleSentences.size()));
        var result = hartcq.process(sentence);
        blackhole.consume(result);
    }

    /**
     * Benchmark end-to-end pipeline performance
     * Includes all processing stages
     */
    @Benchmark
    public void measureEndToEndPipelineThroughput(Blackhole blackhole) {
        // Process varied sentence types to test full pipeline
        for (int i = 0; i < 10; i++) {
            var sentence = sampleSentences.get(i);
            var result = hartcq.process(sentence);
            
            // Validate result to ensure full processing
            if (result != null && result.isSuccessful()) {
                blackhole.consume(result.getOutput());
                blackhole.consume(result.getConfidence());
                blackhole.consume(result.getProcessingTime());
            }
        }
    }

    /**
     * Benchmark template-bounded generation
     * Tests template system performance
     */
    @Benchmark
    public void measureTemplateBoundedGeneration(Blackhole blackhole) {
        // Process sentences that should trigger template usage
        var templateTriggers = List.of(
            "Generate a greeting for the morning.",
            "Create a farewell message.",
            "Write a thank you note.",
            "Compose a brief apology.",
            "Draft a meeting invitation."
        );
        
        for (var trigger : templateTriggers) {
            var result = hartcq.process(trigger);
            blackhole.consume(result);
        }
    }

    /**
     * Main method to run benchmarks from IDE
     */
    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
            .include(HARTCQSentenceBenchmark.class.getSimpleName())
            .forks(1)
            .warmupIterations(2)
            .measurementIterations(3)
            .build();

        var runner = new Runner(opt);
        var results = runner.run();
        
        // Print summary
        System.out.println("\n=== HART-CQ Performance Benchmark Results ===");
        System.out.println("Target: >100 sentences/second");
        
        results.forEach(result -> {
            var benchmarkName = result.getPrimaryResult().getLabel();
            var throughput = result.getPrimaryResult().getScore();
            var unit = result.getPrimaryResult().getScoreUnit();
            
            System.out.printf("%s: %.2f %s%n", benchmarkName, throughput, unit);
            
            if (benchmarkName.contains("SingleSentence")) {
                if (throughput >= 100) {
                    System.out.println("✅ MEETS PERFORMANCE TARGET");
                } else {
                    System.out.println("❌ BELOW PERFORMANCE TARGET");
                }
            }
        });
    }
}