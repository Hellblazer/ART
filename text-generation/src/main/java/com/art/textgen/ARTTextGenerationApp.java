package com.art.textgen;

import com.art.textgen.core.Vocabulary;
import com.art.textgen.generation.PatternGenerator;
import com.art.textgen.generation.EnhancedPatternGenerator;
import com.art.textgen.training.*;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Main ART Text Generation Application
 * Provides complete training and generation interface
 */
public class ARTTextGenerationApp {
    
    private final Vocabulary vocabulary;
    private final EnhancedPatternGenerator patternGenerator;
    private final TrainingPipeline trainingPipeline;
    private final GrossbergTextGenerator textGenerator;
    private boolean isTrained = false;
    
    public ARTTextGenerationApp() {
        this.vocabulary = new Vocabulary(64); // 64-dimensional embeddings
        this.patternGenerator = new EnhancedPatternGenerator(vocabulary, 0.7); // Enhanced with repetition penalty
        this.trainingPipeline = new TrainingPipeline(vocabulary, patternGenerator);
        this.textGenerator = new GrossbergTextGenerator();
    }
    
    public static void main(String[] args) {
        ARTTextGenerationApp app = new ARTTextGenerationApp();
        app.run();
    }
    
    /**
     * Run the application
     */
    public void run() {
        Scanner scanner = new Scanner(System.in);
        
        printWelcome();
        
        while (true) {
            printMenu();
            System.out.print("\nChoice: ");
            String choice = scanner.nextLine().trim();
            
            switch (choice) {
                case "1":
                    trainFromSamples();
                    break;
                case "2":
                    trainFromDirectory(scanner);
                    break;
                case "3":
                    downloadAndPrepareCorpus();
                    break;
                case "4":
                    if (checkTrained()) {
                        generateInteractive(scanner);
                    }
                    break;
                case "5":
                    if (checkTrained()) {
                        batchGenerate(scanner);
                    }
                    break;
                case "6":
                    if (checkTrained()) {
                        showStatistics();
                    }
                    break;
                case "7":
                    if (checkTrained()) {
                        tuneParameters(scanner);
                    }
                    break;
                case "8":
                    System.out.println("\nGoodbye!");
                    textGenerator.shutdown();
                    return;
                default:
                    System.out.println("\nInvalid choice. Please try again.");
            }
        }
    }
    
    /**
     * Print welcome message
     */
    private void printWelcome() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("     ART-Based Text Generation System");
        System.out.println("     Adaptive Resonance Theory for Language Modeling");
        System.out.println("=".repeat(60));
        System.out.println("\nThis system uses Grossberg's neural dynamics and ART");
        System.out.println("principles for text generation with hierarchical memory.");
    }
    
    /**
     * Print menu
     */
    private void printMenu() {
        System.out.println("\n" + "-".repeat(40));
        System.out.println("MAIN MENU");
        System.out.println("-".repeat(40));
        System.out.println("Training Options:");
        System.out.println("  1. Train from sample corpus");
        System.out.println("  2. Train from directory");
        System.out.println("  3. Download and prepare corpus");
        
        if (isTrained) {
            System.out.println("\nGeneration Options:");
            System.out.println("  4. Interactive generation");
            System.out.println("  5. Batch generation");
            System.out.println("  6. Show statistics");
            System.out.println("  7. Tune parameters");
        } else {
            System.out.println("\n(Train the model first to unlock generation options)");
        }
        
        System.out.println("\n  8. Exit");
    }
    
    /**
     * Train from samples
     */
    private void trainFromSamples() {
        System.out.println("\n" + "=".repeat(40));
        System.out.println("Training from Sample Corpus");
        System.out.println("=".repeat(40));
        
        trainingPipeline.trainFromSamples();
        isTrained = true;
        
        System.out.println("\n✓ Training complete!");
        pressEnterToContinue();
    }
    
    /**
     * Train from directory
     */
    private void trainFromDirectory(Scanner scanner) {
        System.out.println("\n" + "=".repeat(40));
        System.out.println("Train from Directory");
        System.out.println("=".repeat(40));
        
        System.out.print("\nEnter directory path (or 'default' for training-corpus): ");
        String path = scanner.nextLine().trim();
        
        if (path.equals("default")) {
            path = "training-corpus";
        }
        
        try {
            System.out.println("\nTraining from: " + path);
            trainingPipeline.trainFromDirectory(path);
            isTrained = true;
            System.out.println("\n✓ Training complete!");
        } catch (IOException e) {
            System.err.println("\n✗ Error: " + e.getMessage());
            System.out.println("Falling back to sample corpus...");
            trainFromSamples();
        }
        
        pressEnterToContinue();
    }
    
    /**
     * Download and prepare corpus
     */
    private void downloadAndPrepareCorpus() {
        System.out.println("\n" + "=".repeat(40));
        System.out.println("Download Training Corpus");
        System.out.println("=".repeat(40));
        
        System.out.println("\nThis will download public domain books from");
        System.out.println("Project Gutenberg and create training corpora.");
        System.out.println("\nNote: This requires internet connection and may");
        System.out.println("take several minutes.");
        
        System.out.print("\nProceed? (y/n): ");
        Scanner scanner = new Scanner(System.in);
        String choice = scanner.nextLine().trim().toLowerCase();
        
        if (choice.equals("y") || choice.equals("yes")) {
            try {
                EnhancedCorpusDownloader.main(new String[]{});
                System.out.println("\n✓ Corpus downloaded successfully!");
                System.out.println("You can now train from the 'training-corpus' directory.");
            } catch (Exception e) {
                System.err.println("\n✗ Error downloading corpus: " + e.getMessage());
            }
        }
        
        pressEnterToContinue();
    }
    
    /**
     * Interactive generation
     */
    private void generateInteractive(Scanner scanner) {
        System.out.println("\n" + "=".repeat(40));
        System.out.println("Interactive Text Generation");
        System.out.println("=".repeat(40));
        System.out.println("Enter a prompt and see the generated continuation.");
        System.out.println("Type 'quit' to return to main menu.\n");
        
        while (true) {
            System.out.print("Prompt: ");
            String prompt = scanner.nextLine().trim();
            
            if (prompt.equalsIgnoreCase("quit")) {
                break;
            }
            
            if (prompt.isEmpty()) {
                System.out.println("Please enter a prompt.\n");
                continue;
            }
            
            System.out.print("Length (words, default 30): ");
            String lengthStr = scanner.nextLine().trim();
            int length = lengthStr.isEmpty() ? 30 : Integer.parseInt(lengthStr);
            
            System.out.println("\nGenerating...\n");
            
            // Generate using the main generator
            List<String> generated = textGenerator.generate(prompt, length)
                .collect(Collectors.toList());
            
            // Display result
            System.out.println("Original: " + prompt);
            System.out.print("Generated: " + prompt);
            for (String token : generated) {
                System.out.print(" " + token);
            }
            System.out.println("\n");
            
            // Also try with pattern generator for comparison
            System.out.println("Pattern-based variant:");
            generateWithPatterns(prompt, length);
            System.out.println();
        }
    }
    
    /**
     * Generate with pattern generator
     */
    private void generateWithPatterns(String prompt, int length) {
        List<String> context = vocabulary.tokenize(prompt);
        System.out.print(prompt);
        
        for (int i = 0; i < length; i++) {
            String next = patternGenerator.generateNext(context);
            
            if (next.equals(Vocabulary.END_TOKEN)) {
                break;
            }
            
            System.out.print(" " + next);
            context.add(next);
            
            // Keep context window
            if (context.size() > 10) {
                context.remove(0);
            }
        }
        System.out.println();
    }
    
    /**
     * Batch generation
     */
    private void batchGenerate(Scanner scanner) {
        System.out.println("\n" + "=".repeat(40));
        System.out.println("Batch Generation");
        System.out.println("=".repeat(40));
        
        String[] prompts = {
            "The future of artificial intelligence",
            "Once upon a time",
            "The human brain",
            "In the beginning",
            "Science has shown that",
            "The key to understanding",
            "Machine learning algorithms",
            "Consciousness emerges from"
        };
        
        System.out.print("Length per prompt (default 30): ");
        String lengthStr = scanner.nextLine().trim();
        int length = lengthStr.isEmpty() ? 30 : Integer.parseInt(lengthStr);
        
        System.out.println("\nGenerating " + prompts.length + " samples...\n");
        
        for (String prompt : prompts) {
            System.out.println("Prompt: \"" + prompt + "\"");
            
            List<String> generated = textGenerator.generate(prompt, length)
                .collect(Collectors.toList());
            
            System.out.print("Output: " + prompt);
            for (String token : generated) {
                System.out.print(" " + token);
            }
            System.out.println("\n");
        }
        
        pressEnterToContinue();
    }
    
    /**
     * Show statistics
     */
    private void showStatistics() {
        System.out.println("\n" + "=".repeat(40));
        System.out.println("System Statistics");
        System.out.println("=".repeat(40));
        
        Map<String, Double> metrics = trainingPipeline.getMetrics();
        
        System.out.println("\nTraining Metrics:");
        System.out.printf("  Documents:        %.0f\n", metrics.get("total_documents"));
        System.out.printf("  Total Tokens:     %.0f\n", metrics.get("total_tokens"));
        System.out.printf("  Unique Tokens:    %.0f\n", metrics.get("unique_tokens"));
        System.out.printf("  Vocabulary Size:  %.0f\n", metrics.get("vocabulary_size"));
        System.out.printf("  N-gram Patterns:  %.0f\n", metrics.get("ngram_patterns"));
        
        System.out.println("\nVocabulary Statistics:");
        System.out.println("  Size: " + vocabulary.size());
        System.out.println("  Embedding Dim: " + vocabulary.getEmbeddingDim());
        
        System.out.println("\nPattern Generator Statistics:");
        Map<String, Object> patternStats = patternGenerator.getStatistics();
        System.out.println("  Total Patterns: " + patternStats.get("total_patterns"));
        System.out.println("  Max Pattern Length: " + patternStats.get("max_length"));
        
        pressEnterToContinue();
    }
    
    /**
     * Tune parameters
     */
    private void tuneParameters(Scanner scanner) {
        System.out.println("\n" + "=".repeat(40));
        System.out.println("Parameter Tuning");
        System.out.println("=".repeat(40));
        
        System.out.println("\nCurrent Parameters:");
        System.out.println("  Temperature: " + patternGenerator.getTemperature());
        
        System.out.print("\nNew temperature (0.1-1.0, current: " + 
            patternGenerator.getTemperature() + "): ");
        String tempStr = scanner.nextLine().trim();
        
        if (!tempStr.isEmpty()) {
            try {
                double temp = Double.parseDouble(tempStr);
                if (temp >= 0.1 && temp <= 1.0) {
                    patternGenerator.setTemperature(temp);
                    System.out.println("✓ Temperature updated to: " + temp);
                } else {
                    System.out.println("✗ Temperature must be between 0.1 and 1.0");
                }
            } catch (NumberFormatException e) {
                System.out.println("✗ Invalid number format");
            }
        }
        
        pressEnterToContinue();
    }
    
    /**
     * Check if model is trained
     */
    private boolean checkTrained() {
        if (!isTrained) {
            System.out.println("\n⚠ Please train the model first!");
            pressEnterToContinue();
            return false;
        }
        return true;
    }
    
    /**
     * Press enter to continue
     */
    private void pressEnterToContinue() {
        System.out.print("\nPress Enter to continue...");
        new Scanner(System.in).nextLine();
    }
}
