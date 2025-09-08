package com.hellblazer.art.nlp.util;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Utility class for detecting and removing duplicates in corpus collections
 * and other text data processing operations.
 */
public class DeduplicationUtil {
    
    /**
     * Removes duplicate strings from a collection based on exact matching.
     */
    public static <T extends Collection<String>> List<String> removeDuplicates(T collection) {
        return collection.stream()
                .distinct()
                .collect(Collectors.toList());
    }
    
    /**
     * Removes duplicate strings from a collection based on normalized content
     * (trimmed, lowercase).
     */
    public static <T extends Collection<String>> List<String> removeDuplicatesNormalized(T collection) {
        var seen = new HashSet<String>();
        return collection.stream()
                .filter(item -> {
                    var normalized = item.trim().toLowerCase();
                    return seen.add(normalized);
                })
                .collect(Collectors.toList());
    }
    
    /**
     * Removes near-duplicate strings based on similarity threshold.
     */
    public static List<String> removeNearDuplicates(Collection<String> collection, double similarityThreshold) {
        var result = new ArrayList<String>();
        
        for (var item : collection) {
            boolean isDuplicate = false;
            for (var existing : result) {
                if (calculateSimilarity(item, existing) > similarityThreshold) {
                    isDuplicate = true;
                    break;
                }
            }
            if (!isDuplicate) {
                result.add(item);
            }
        }
        
        return result;
    }
    
    /**
     * Calculates Jaccard similarity between two strings based on character n-grams.
     */
    public static double calculateSimilarity(String str1, String str2) {
        if (str1.equals(str2)) return 1.0;
        
        var ngrams1 = generateNGrams(str1, 2);
        var ngrams2 = generateNGrams(str2, 2);
        
        var intersection = new HashSet<>(ngrams1);
        intersection.retainAll(ngrams2);
        
        var union = new HashSet<>(ngrams1);
        union.addAll(ngrams2);
        
        return union.isEmpty() ? 0.0 : (double) intersection.size() / union.size();
    }
    
    /**
     * Generates character n-grams for a string.
     */
    private static Set<String> generateNGrams(String text, int n) {
        var ngrams = new HashSet<String>();
        if (text.length() < n) {
            ngrams.add(text);
            return ngrams;
        }
        
        for (int i = 0; i <= text.length() - n; i++) {
            ngrams.add(text.substring(i, i + n));
        }
        return ngrams;
    }
    
    /**
     * Detects duplicate files in a directory based on content hash.
     */
    public static Map<String, List<Path>> findDuplicateFiles(Path directory) throws IOException {
        var fileHashes = new ConcurrentHashMap<String, List<Path>>();
        
        Files.walk(directory)
                .filter(Files::isRegularFile)
                .filter(path -> !path.toString().contains("target")) // Skip Maven target
                .parallel()
                .forEach(path -> {
                    try {
                        var hash = calculateFileHash(path);
                        fileHashes.computeIfAbsent(hash, k -> new ArrayList<>()).add(path);
                    } catch (Exception e) {
                        System.err.println("Error processing file " + path + ": " + e.getMessage());
                    }
                });
        
        // Return only files that have duplicates
        return fileHashes.entrySet().stream()
                .filter(entry -> entry.getValue().size() > 1)
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
    }
    
    /**
     * Calculates SHA-256 hash of a file for duplicate detection.
     */
    private static String calculateFileHash(Path filePath) throws IOException, NoSuchAlgorithmException {
        var digest = MessageDigest.getInstance("SHA-256");
        var bytes = Files.readAllBytes(filePath);
        var hashBytes = digest.digest(bytes);
        
        var hexString = new StringBuilder();
        for (byte b : hashBytes) {
            var hex = Integer.toHexString(0xff & b);
            if (hex.length() == 1) {
                hexString.append('0');
            }
            hexString.append(hex);
        }
        return hexString.toString();
    }
    
    /**
     * Removes duplicate lines from a text file and writes result to a new file.
     */
    public static int deduplicateTextFile(Path inputFile, Path outputFile) throws IOException {
        var lines = Files.readAllLines(inputFile);
        var originalSize = lines.size();
        
        var uniqueLines = removeDuplicates(lines);
        Files.write(outputFile, uniqueLines);
        
        return originalSize - uniqueLines.size();
    }
    
    /**
     * Statistics about deduplication operation.
     */
    public record DeduplicationStats(
        int originalCount,
        int uniqueCount,
        int duplicatesRemoved,
        double deduplicationRatio
    ) {
        public static DeduplicationStats calculate(int original, int unique) {
            var removed = original - unique;
            var ratio = original > 0 ? (double) removed / original : 0.0;
            return new DeduplicationStats(original, unique, removed, ratio);
        }
        
        @Override
        public String toString() {
            return String.format("Original: %d, Unique: %d, Removed: %d (%.2f%%)", 
                originalCount, uniqueCount, duplicatesRemoved, deduplicationRatio * 100);
        }
    }
    
    /**
     * Memory-efficient duplicate detection for large collections.
     */
    public static class MemoryEfficientDeduplicator {
        private final Set<Integer> seenHashes = new HashSet<>();
        private int processedCount = 0;
        private int duplicateCount = 0;
        
        public boolean addIfUnique(String item) {
            processedCount++;
            var hash = item.hashCode();
            
            if (seenHashes.contains(hash)) {
                duplicateCount++;
                return false; // Duplicate
            } else {
                seenHashes.add(hash);
                return true; // Unique
            }
        }
        
        public DeduplicationStats getStats() {
            var unique = processedCount - duplicateCount;
            return DeduplicationStats.calculate(processedCount, unique);
        }
        
        public void reset() {
            seenHashes.clear();
            processedCount = 0;
            duplicateCount = 0;
        }
    }
}