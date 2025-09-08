package com.art.textgen.monitoring;

import javax.swing.*;
import java.awt.*;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Real-time training dashboard for ART Text Generation
 * Provides visualization of metrics, patterns, and generation samples
 * Based on Phase 6.1 of EXECUTION_PLAN.md
 */
public class TrainingDashboard extends JFrame {
    
    // UI Components
    private JTextArea metricsDisplay;
    private JTextArea patternsDisplay;
    private JTextArea samplesDisplay;
    private JTextArea logArea;
    private JProgressBar trainingProgress;
    private JLabel statusLabel;
    
    // Data tracking
    private final Map<String, List<Double>> metricsHistory;
    private final Queue<String> recentSamples;
    private final Map<String, Integer> patternStatistics;
    
    // Update thread
    private final ScheduledExecutorService updateScheduler;
    private volatile boolean isRunning = false;
    
    // Dashboard configuration
    private final int updateIntervalMs = 1000;
    private final int maxHistorySize = 100;
    private final int maxSamplesShown = 5;
    
    public TrainingDashboard() {
        super("ART Text Generation - Training Dashboard");
        
        this.metricsHistory = new ConcurrentHashMap<>();
        this.recentSamples = new ConcurrentLinkedQueue<>();
        this.patternStatistics = new ConcurrentHashMap<>();
        this.updateScheduler = Executors.newScheduledThreadPool(1);
        
        initializeUI();
        startUpdateThread();
    }
    
    /**
     * Initialize the dashboard UI
     */
    private void initializeUI() {
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());
        
        // Create main panels
        JPanel mainPanel = new JPanel(new GridLayout(2, 2, 10, 10));
        mainPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        
        // Metrics Panel
        JPanel metricsPanel = createPanel("Training Metrics");
        metricsDisplay = new JTextArea(10, 30);
        metricsDisplay.setEditable(false);
        metricsDisplay.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        metricsPanel.add(new JScrollPane(metricsDisplay));
        
        // Patterns Panel
        JPanel patternsPanel = createPanel("Pattern Statistics");
        patternsDisplay = new JTextArea(10, 30);
        patternsDisplay.setEditable(false);
        patternsDisplay.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        patternsPanel.add(new JScrollPane(patternsDisplay));
        
        // Samples Panel
        JPanel samplesPanel = createPanel("Generation Samples");
        samplesDisplay = new JTextArea(10, 30);
        samplesDisplay.setEditable(false);
        samplesDisplay.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));
        samplesDisplay.setLineWrap(true);
        samplesDisplay.setWrapStyleWord(true);
        samplesPanel.add(new JScrollPane(samplesDisplay));
        
        // Progress Panel
        JPanel progressPanel = createPanel("Training Progress");
        JPanel progressContent = new JPanel(new GridLayout(3, 1, 5, 5));
        
        statusLabel = new JLabel("Status: Initializing...");
        statusLabel.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 12));
        
        trainingProgress = new JProgressBar(0, 100);
        trainingProgress.setStringPainted(true);
        
        JPanel buttonPanel = new JPanel(new FlowLayout());
        JButton startButton = new JButton("Start");
        JButton pauseButton = new JButton("Pause");
        JButton stopButton = new JButton("Stop");
        JButton exportButton = new JButton("Export");
        
        startButton.addActionListener(e -> startTraining());
        pauseButton.addActionListener(e -> pauseTraining());
        stopButton.addActionListener(e -> stopTraining());
        exportButton.addActionListener(e -> exportMetrics());
        
        buttonPanel.add(startButton);
        buttonPanel.add(pauseButton);
        buttonPanel.add(stopButton);
        buttonPanel.add(exportButton);
        
        progressContent.add(statusLabel);
        progressContent.add(trainingProgress);
        progressContent.add(buttonPanel);
        progressPanel.add(progressContent);
        
        // Add panels to main panel
        mainPanel.add(metricsPanel);
        mainPanel.add(patternsPanel);
        mainPanel.add(samplesPanel);
        mainPanel.add(progressPanel);
        
        // Log Panel at bottom
        JPanel logPanel = createPanel("Training Log");
        logArea = new JTextArea(5, 80);
        logArea.setEditable(false);
        logArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));
        JScrollPane logScroll = new JScrollPane(logArea);
        logScroll.setPreferredSize(new Dimension(800, 100));
        logPanel.add(logScroll);
        
        // Add to frame
        add(mainPanel, BorderLayout.CENTER);
        add(logPanel, BorderLayout.SOUTH);
        
        // Set size and center
        setSize(1000, 700);
        setLocationRelativeTo(null);
        
        // Initial update
        updateDisplay();
    }
    
    /**
     * Create a titled panel
     */
    private JPanel createPanel(String title) {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBorder(BorderFactory.createTitledBorder(
            BorderFactory.createEtchedBorder(), 
            title,
            javax.swing.border.TitledBorder.LEFT,
            javax.swing.border.TitledBorder.TOP,
            new Font(Font.SANS_SERIF, Font.BOLD, 14)
        ));
        return panel;
    }
    
    /**
     * Start the update thread
     */
    private void startUpdateThread() {
        updateScheduler.scheduleAtFixedRate(() -> {
            if (isRunning) {
                SwingUtilities.invokeLater(this::updateDisplay);
            }
        }, 0, updateIntervalMs, TimeUnit.MILLISECONDS);
    }
    
    /**
     * Update all display panels
     */
    private void updateDisplay() {
        updateMetricsDisplay();
        updatePatternsDisplay();
        updateSamplesDisplay();
    }
    
    /**
     * Update metrics display
     */
    private void updateMetricsDisplay() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Current Metrics ===\n\n");
        
        for (Map.Entry<String, List<Double>> entry : metricsHistory.entrySet()) {
            String metric = entry.getKey();
            List<Double> history = entry.getValue();
            
            if (!history.isEmpty()) {
                double current = history.get(history.size() - 1);
                double avg = history.stream().mapToDouble(Double::doubleValue).average().orElse(0);
                
                sb.append(String.format("%-20s: %.4f (avg: %.4f)\n", metric, current, avg));
                sb.append("  ").append(createMiniChart(history, 30)).append("\n");
            }
        }
        
        metricsDisplay.setText(sb.toString());
    }
    
    /**
     * Update patterns display
     */
    private void updatePatternsDisplay() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Pattern Statistics ===\n\n");
        
        List<Map.Entry<String, Integer>> sorted = new ArrayList<>(patternStatistics.entrySet());
        sorted.sort((a, b) -> b.getValue().compareTo(a.getValue()));
        
        sb.append(String.format("Total Patterns: %d\n\n", patternStatistics.size()));
        sb.append("Top Patterns:\n");
        
        int count = 0;
        for (Map.Entry<String, Integer> entry : sorted) {
            if (count++ >= 10) break;
            sb.append(String.format("  %-30s: %d\n", 
                truncate(entry.getKey(), 30), entry.getValue()));
        }
        
        patternsDisplay.setText(sb.toString());
    }
    
    /**
     * Update samples display
     */
    private void updateSamplesDisplay() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Recent Generation Samples ===\n\n");
        
        int sampleNum = 1;
        for (String sample : recentSamples) {
            sb.append(String.format("Sample %d:\n", sampleNum++));
            sb.append(wrapText(sample, 60));
            sb.append("\n\n");
            
            if (sampleNum > maxSamplesShown) break;
        }
        
        samplesDisplay.setText(sb.toString());
    }
    
    /**
     * Create mini ASCII chart
     */
    private String createMiniChart(List<Double> values, int width) {
        if (values.isEmpty()) return "";
        
        double min = values.stream().min(Double::compare).orElse(0.0);
        double max = values.stream().max(Double::compare).orElse(1.0);
        double range = max - min;
        
        if (range == 0) range = 1;
        
        StringBuilder chart = new StringBuilder();
        int step = Math.max(1, values.size() / width);
        
        for (int i = 0; i < values.size(); i += step) {
            double val = values.get(i);
            double normalized = (val - min) / range;
            
            if (normalized < 0.2) chart.append("_");
            else if (normalized < 0.4) chart.append("-");
            else if (normalized < 0.6) chart.append("=");
            else if (normalized < 0.8) chart.append("+");
            else chart.append("#");
        }
        
        return chart.toString();
    }
    
    /**
     * Public methods for updating dashboard from training pipeline
     */
    
    public void updateMetric(String name, double value) {
        metricsHistory.computeIfAbsent(name, k -> new ArrayList<>()).add(value);
        
        List<Double> history = metricsHistory.get(name);
        if (history.size() > maxHistorySize) {
            history.remove(0);
        }
    }
    
    public void addSample(String sample) {
        recentSamples.offer(sample);
        while (recentSamples.size() > maxSamplesShown * 2) {
            recentSamples.poll();
        }
    }
    
    public void updatePattern(String pattern, int count) {
        patternStatistics.put(pattern, count);
    }
    
    public void updateProgress(int percent) {
        SwingUtilities.invokeLater(() -> {
            trainingProgress.setValue(percent);
        });
    }
    
    public void updateStatus(String status) {
        SwingUtilities.invokeLater(() -> {
            statusLabel.setText("Status: " + status);
            log(status);
        });
    }
    
    public void log(String message) {
        SwingUtilities.invokeLater(() -> {
            String timestamp = LocalDateTime.now()
                .format(DateTimeFormatter.ofPattern("HH:mm:ss"));
            logArea.append("[" + timestamp + "] " + message + "\n");
            logArea.setCaretPosition(logArea.getDocument().getLength());
        });
    }
    
    /**
     * Control methods
     */
    
    private void startTraining() {
        isRunning = true;
        updateStatus("Training started");
        log("Training pipeline initiated");
    }
    
    private void pauseTraining() {
        isRunning = false;
        updateStatus("Training paused");
        log("Training pipeline paused");
    }
    
    private void stopTraining() {
        isRunning = false;
        updateStatus("Training stopped");
        log("Training pipeline stopped");
        trainingProgress.setValue(0);
    }
    
    /**
     * Export metrics to file
     */
    private void exportMetrics() {
        JFileChooser chooser = new JFileChooser();
        chooser.setSelectedFile(new java.io.File("metrics_export.csv"));
        
        if (chooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                java.io.File file = chooser.getSelectedFile();
                exportMetricsToFile(file);
                log("Metrics exported to: " + file.getAbsolutePath());
            } catch (Exception e) {
                log("Error exporting metrics: " + e.getMessage());
            }
        }
    }
    
    private void exportMetricsToFile(java.io.File file) throws java.io.IOException {
        try (java.io.PrintWriter writer = new java.io.PrintWriter(file)) {
            writer.println("Metric,Values");
            
            for (Map.Entry<String, List<Double>> entry : metricsHistory.entrySet()) {
                writer.print(entry.getKey() + ",");
                writer.println(entry.getValue().stream()
                    .map(String::valueOf)
                    .reduce((a, b) -> a + "," + b)
                    .orElse(""));
            }
        }
    }
    
    // Helper methods
    
    private String truncate(String s, int maxLength) {
        if (s.length() <= maxLength) return s;
        return s.substring(0, maxLength - 3) + "...";
    }
    
    private String wrapText(String text, int width) {
        StringBuilder wrapped = new StringBuilder();
        String[] words = text.split("\\s+");
        int lineLength = 0;
        
        for (String word : words) {
            if (lineLength + word.length() + 1 > width) {
                wrapped.append("\n");
                lineLength = 0;
            }
            if (lineLength > 0) {
                wrapped.append(" ");
                lineLength++;
            }
            wrapped.append(word);
            lineLength += word.length();
        }
        
        return wrapped.toString();
    }
    
    /**
     * Shutdown the dashboard
     */
    public void shutdown() {
        isRunning = false;
        updateScheduler.shutdown();
        dispose();
    }
    
    /**
     * Main method for testing
     */
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            TrainingDashboard dashboard = new TrainingDashboard();
            dashboard.setVisible(true);
            
            // Simulate some data for testing
            dashboard.isRunning = true;
            Random random = new Random();
            
            javax.swing.Timer timer = new javax.swing.Timer(500, e -> {
                dashboard.updateMetric("perplexity", 50 + random.nextGaussian() * 10);
                dashboard.updateMetric("loss", 2.5 + random.nextGaussian() * 0.5);
                dashboard.updateMetric("accuracy", 0.7 + random.nextGaussian() * 0.1);
                
                dashboard.updatePattern("pattern_" + random.nextInt(20), random.nextInt(100));
                
                if (random.nextDouble() < 0.2) {
                    dashboard.addSample("This is a generated sample text that demonstrates " +
                        "the generation capabilities of the ART system. " +
                        "Random value: " + random.nextInt(1000));
                }
                
                dashboard.updateProgress(random.nextInt(100));
            });
            timer.start();
        });
    }
}
