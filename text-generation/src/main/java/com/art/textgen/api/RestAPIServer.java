package com.art.textgen.api;

import com.art.textgen.generation.EnhancedPatternGenerator;
import com.art.textgen.generation.PatternGenerator;
import com.art.textgen.generation.ContextAwareGenerator;
import com.art.textgen.evaluation.TextGenerationMetrics;
import com.art.textgen.training.IncrementalTrainer;
import com.art.textgen.infrastructure.ModelCheckpoint;
import com.art.textgen.core.Vocabulary;
import com.sun.net.httpserver.*;
import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.io.*;
import java.net.InetSocketAddress;
import java.util.*;
import java.util.concurrent.Executors;

/**
 * REST API Server for ART Text Generation
 * Provides endpoints for text generation, training, and metrics
 */
public class RestAPIServer {
    private static final int PORT = 8080;
    private final HttpServer server;
    private final Gson gson;
    
    // Model components
    private final EnhancedPatternGenerator generator;
    private final IncrementalTrainer trainer;
    private final TextGenerationMetrics metrics;
    private final Vocabulary vocabulary;
    private ModelCheckpoint checkpoint;
    
    // API Statistics
    private long requestCount = 0;
    private long totalGenerationTime = 0;
    private final Map<String, Long> endpointCounts;
    
    public RestAPIServer() throws IOException {
        this.server = HttpServer.create(new InetSocketAddress(PORT), 0);
        this.gson = new Gson();
        this.endpointCounts = new HashMap<>();
        
        // Initialize model components
        this.vocabulary = new Vocabulary(10000);
        this.generator = new EnhancedPatternGenerator(vocabulary);
        PatternGenerator patternGen = new PatternGenerator(vocabulary, 1.0);
        this.trainer = new IncrementalTrainer(vocabulary, patternGen);
        this.metrics = new TextGenerationMetrics();
        
        // Load latest checkpoint if available
        loadLatestCheckpoint();
        
        // Configure endpoints
        setupEndpoints();
        
        // Configure thread pool
        server.setExecutor(Executors.newFixedThreadPool(10));
    }
    
    /**
     * Setup all REST endpoints
     */
    private void setupEndpoints() {
        // Text generation endpoint
        server.createContext("/api/generate", new GenerateHandler());
        
        // Training endpoint
        server.createContext("/api/train", new TrainHandler());
        
        // Metrics endpoint
        server.createContext("/api/metrics", new MetricsHandler());
        
        // Model management endpoints
        server.createContext("/api/model/save", new SaveModelHandler());
        server.createContext("/api/model/load", new LoadModelHandler());
        server.createContext("/api/model/reset", new ResetModelHandler());
        
        // Configuration endpoint
        server.createContext("/api/config", new ConfigHandler());
        
        // Health check endpoint
        server.createContext("/api/health", new HealthHandler());
        
        // Statistics endpoint
        server.createContext("/api/stats", new StatsHandler());
        
        // Documentation endpoint
        server.createContext("/api/docs", new DocsHandler());
    }
    
    /**
     * Generate text handler
     */
    class GenerateHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"POST".equals(exchange.getRequestMethod())) {
                sendResponse(exchange, 405, "{\"error\": \"Method not allowed\"}");
                return;
            }
            
            long startTime = System.currentTimeMillis();
            incrementEndpointCount("/api/generate");
            
            try {
                // Parse request
                String requestBody = readRequestBody(exchange);
                JsonObject request = gson.fromJson(requestBody, JsonObject.class);
                
                String prompt = request.get("prompt").getAsString();
                int maxLength = request.has("maxLength") ? 
                    request.get("maxLength").getAsInt() : 100;
                double temperature = request.has("temperature") ? 
                    request.get("temperature").getAsDouble() : 0.9;
                int topK = request.has("topK") ? 
                    request.get("topK").getAsInt() : 40;
                double topP = request.has("topP") ? 
                    request.get("topP").getAsDouble() : 0.9;
                
                // Configure generator
                generator.setTemperature(temperature);
                generator.setTopK(topK);
                generator.setTopP(topP);
                
                // Generate text
                String generated = generator.generate(prompt, maxLength);
                
                // Calculate metrics
                double diversity = metrics.calculateDiversity(Arrays.asList(generated), 2);
                double readability = metrics.calculateReadability(generated);
                
                // Build response
                JsonObject response = new JsonObject();
                response.addProperty("prompt", prompt);
                response.addProperty("generated", generated);
                response.addProperty("diversity", diversity);
                response.addProperty("readability", readability);
                response.addProperty("generationTime", System.currentTimeMillis() - startTime);
                
                totalGenerationTime += (System.currentTimeMillis() - startTime);
                sendResponse(exchange, 200, gson.toJson(response));
                
            } catch (Exception e) {
                JsonObject error = new JsonObject();
                error.addProperty("error", e.getMessage());
                sendResponse(exchange, 500, gson.toJson(error));
            }
        }
    }
    
    /**
     * Training handler
     */
    class TrainHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"POST".equals(exchange.getRequestMethod())) {
                sendResponse(exchange, 405, "{\"error\": \"Method not allowed\"}");
                return;
            }
            
            incrementEndpointCount("/api/train");
            
            try {
                String requestBody = readRequestBody(exchange);
                JsonObject request = gson.fromJson(requestBody, JsonObject.class);
                
                String text = request.get("text").getAsString();
                boolean incremental = request.has("incremental") ? 
                    request.get("incremental").getAsBoolean() : true;
                
                // Train the model
                List<String> documents = Arrays.asList(text.split("\n\n"));
                trainer.trainBatch(documents);
                
                JsonObject response = new JsonObject();
                response.addProperty("success", true);
                response.addProperty("documentsProcessed", documents.size());
                response.addProperty("incremental", incremental);
                
                sendResponse(exchange, 200, gson.toJson(response));
                
            } catch (Exception e) {
                JsonObject error = new JsonObject();
                error.addProperty("error", e.getMessage());
                sendResponse(exchange, 500, gson.toJson(error));
            }
        }
    }
    
    /**
     * Metrics handler
     */
    class MetricsHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"POST".equals(exchange.getRequestMethod())) {
                sendResponse(exchange, 405, "{\"error\": \"Method not allowed\"}");
                return;
            }
            
            incrementEndpointCount("/api/metrics");
            
            try {
                String requestBody = readRequestBody(exchange);
                JsonObject request = gson.fromJson(requestBody, JsonObject.class);
                
                String text = request.get("text").getAsString();
                
                // Calculate all metrics
                List<String> tokens = Arrays.asList(text.split("\\s+"));
                double perplexity = 50.0; // Placeholder - needs proper implementation
                double bleu = metrics.calculateBLEU(text, "reference text here", 4);
                double diversity = metrics.calculateDiversity(Arrays.asList(text), 2);
                double coherence = metrics.calculateCoherence(text, 3);
                double fluency = metrics.calculateFluency(text);
                double readability = metrics.calculateReadability(text);
                
                JsonObject response = new JsonObject();
                response.addProperty("perplexity", perplexity);
                response.addProperty("bleu", bleu);
                response.addProperty("diversity", diversity);
                response.addProperty("coherence", coherence);
                response.addProperty("fluency", fluency);
                response.addProperty("readability", readability);
                
                sendResponse(exchange, 200, gson.toJson(response));
                
            } catch (Exception e) {
                JsonObject error = new JsonObject();
                error.addProperty("error", e.getMessage());
                sendResponse(exchange, 500, gson.toJson(error));
            }
        }
    }
    
    /**
     * Save model handler
     */
    class SaveModelHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"POST".equals(exchange.getRequestMethod())) {
                sendResponse(exchange, 405, "{\"error\": \"Method not allowed\"}");
                return;
            }
            
            incrementEndpointCount("/api/model/save");
            
            try {
                String requestBody = readRequestBody(exchange);
                JsonObject request = gson.fromJson(requestBody, JsonObject.class);
                
                String modelName = request.has("name") ? 
                    request.get("name").getAsString() : "art_model_" + System.currentTimeMillis();
                
                // Save checkpoint
                checkpoint = new ModelCheckpoint.Builder(modelName)
                    .withEpoch(1)
                    .withTotalSteps(trainer.getTotalSteps())
                    .withTrainingMetric("requests", requestCount)
                    .build();
                checkpoint.save();
                
                JsonObject response = new JsonObject();
                response.addProperty("success", true);
                response.addProperty("modelName", modelName);
                response.addProperty("savedAt", new Date().toString());
                
                sendResponse(exchange, 200, gson.toJson(response));
                
            } catch (Exception e) {
                JsonObject error = new JsonObject();
                error.addProperty("error", e.getMessage());
                sendResponse(exchange, 500, gson.toJson(error));
            }
        }
    }
    
    /**
     * Load model handler
     */
    class LoadModelHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"POST".equals(exchange.getRequestMethod())) {
                sendResponse(exchange, 405, "{\"error\": \"Method not allowed\"}");
                return;
            }
            
            incrementEndpointCount("/api/model/load");
            
            try {
                String requestBody = readRequestBody(exchange);
                JsonObject request = gson.fromJson(requestBody, JsonObject.class);
                
                String modelName = request.get("name").getAsString();
                
                // Load checkpoint
                checkpoint = ModelCheckpoint.load(modelName);
                // Restore model state (implement restoration logic)
                
                JsonObject response = new JsonObject();
                response.addProperty("success", true);
                response.addProperty("modelName", modelName);
                response.addProperty("loadedAt", new Date().toString());
                
                sendResponse(exchange, 200, gson.toJson(response));
                
            } catch (Exception e) {
                JsonObject error = new JsonObject();
                error.addProperty("error", e.getMessage());
                sendResponse(exchange, 500, gson.toJson(error));
            }
        }
    }
    
    /**
     * Reset model handler
     */
    class ResetModelHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"POST".equals(exchange.getRequestMethod())) {
                sendResponse(exchange, 405, "{\"error\": \"Method not allowed\"}");
                return;
            }
            
            incrementEndpointCount("/api/model/reset");
            
            // Reset model components
            generator.reset();
            // trainer.reset(); // IncrementalTrainer doesn't have reset
            
            JsonObject response = new JsonObject();
            response.addProperty("success", true);
            response.addProperty("resetAt", new Date().toString());
            
            sendResponse(exchange, 200, gson.toJson(response));
        }
    }
    
    /**
     * Configuration handler
     */
    class ConfigHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            incrementEndpointCount("/api/config");
            
            if ("GET".equals(exchange.getRequestMethod())) {
                // Return current configuration
                JsonObject config = new JsonObject();
                config.addProperty("temperature", generator.getTemperature());
                config.addProperty("topK", generator.getTopK());
                config.addProperty("topP", generator.getTopP());
                config.addProperty("repetitionPenalty", generator.getRepetitionPenalty());
                
                sendResponse(exchange, 200, gson.toJson(config));
                
            } else if ("POST".equals(exchange.getRequestMethod())) {
                // Update configuration
                String requestBody = readRequestBody(exchange);
                JsonObject request = gson.fromJson(requestBody, JsonObject.class);
                
                if (request.has("temperature")) {
                    generator.setTemperature(request.get("temperature").getAsDouble());
                }
                if (request.has("topK")) {
                    generator.setTopK(request.get("topK").getAsInt());
                }
                if (request.has("topP")) {
                    generator.setTopP(request.get("topP").getAsDouble());
                }
                if (request.has("repetitionPenalty")) {
                    generator.setRepetitionPenalty(request.get("repetitionPenalty").getAsDouble());
                }
                
                JsonObject response = new JsonObject();
                response.addProperty("success", true);
                sendResponse(exchange, 200, gson.toJson(response));
                
            } else {
                sendResponse(exchange, 405, "{\"error\": \"Method not allowed\"}");
            }
        }
    }
    
    /**
     * Health check handler
     */
    class HealthHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            incrementEndpointCount("/api/health");
            
            JsonObject health = new JsonObject();
            health.addProperty("status", "healthy");
            health.addProperty("uptime", getUptime());
            health.addProperty("requestCount", requestCount);
            health.addProperty("averageGenerationTime", 
                requestCount > 0 ? totalGenerationTime / requestCount : 0);
            
            sendResponse(exchange, 200, gson.toJson(health));
        }
    }
    
    /**
     * Statistics handler
     */
    class StatsHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            incrementEndpointCount("/api/stats");
            
            JsonObject stats = new JsonObject();
            stats.addProperty("totalRequests", requestCount);
            stats.addProperty("totalGenerationTime", totalGenerationTime);
            stats.addProperty("averageGenerationTime", 
                requestCount > 0 ? totalGenerationTime / requestCount : 0);
            
            JsonObject endpoints = new JsonObject();
            for (Map.Entry<String, Long> entry : endpointCounts.entrySet()) {
                endpoints.addProperty(entry.getKey(), entry.getValue());
            }
            stats.add("endpointCounts", endpoints);
            
            sendResponse(exchange, 200, gson.toJson(stats));
        }
    }
    
    /**
     * API Documentation handler
     */
    class DocsHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            incrementEndpointCount("/api/docs");
            
            String html = generateAPIDocumentation();
            exchange.getResponseHeaders().set("Content-Type", "text/html");
            sendResponse(exchange, 200, html);
        }
    }
    
    /**
     * Helper method to read request body
     */
    private String readRequestBody(HttpExchange exchange) throws IOException {
        InputStreamReader isr = new InputStreamReader(exchange.getRequestBody(), "utf-8");
        BufferedReader br = new BufferedReader(isr);
        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = br.readLine()) != null) {
            sb.append(line);
        }
        br.close();
        return sb.toString();
    }
    
    /**
     * Helper method to send response
     */
    private void sendResponse(HttpExchange exchange, int statusCode, String response) throws IOException {
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(statusCode, response.getBytes().length);
        OutputStream os = exchange.getResponseBody();
        os.write(response.getBytes());
        os.close();
        requestCount++;
    }
    
    /**
     * Load latest checkpoint on startup
     */
    private void loadLatestCheckpoint() {
        // Implementation to load most recent checkpoint
        System.out.println("Loading latest checkpoint...");
        // checkpoint = ModelCheckpoint.loadLatest();
    }
    
    /**
     * Increment endpoint counter
     */
    private void incrementEndpointCount(String endpoint) {
        endpointCounts.put(endpoint, endpointCounts.getOrDefault(endpoint, 0L) + 1);
    }
    
    /**
     * Get server uptime
     */
    private String getUptime() {
        long uptime = System.currentTimeMillis() - startTime;
        long hours = uptime / 3600000;
        long minutes = (uptime % 3600000) / 60000;
        return String.format("%d hours, %d minutes", hours, minutes);
    }
    
    private final long startTime = System.currentTimeMillis();
    
    /**
     * Generate API documentation HTML
     */
    private String generateAPIDocumentation() {
        return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>ART Text Generation API</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    h1 { color: #333; }
                    .endpoint { background: #f5f5f5; padding: 15px; margin: 20px 0; border-radius: 5px; }
                    .method { color: #007bff; font-weight: bold; }
                    .path { color: #28a745; font-weight: bold; }
                    code { background: #e9ecef; padding: 2px 5px; border-radius: 3px; }
                </style>
            </head>
            <body>
                <h1>ART Text Generation REST API</h1>
                <p>Neuroscience-inspired text generation using Adaptive Resonance Theory</p>
                
                <div class="endpoint">
                    <span class="method">POST</span> <span class="path">/api/generate</span>
                    <p>Generate text from a prompt</p>
                    <p>Request body:</p>
                    <code>{
                        "prompt": "string",
                        "maxLength": 100,
                        "temperature": 0.9,
                        "topK": 40,
                        "topP": 0.9
                    }</code>
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> <span class="path">/api/train</span>
                    <p>Train the model on new text</p>
                    <p>Request body:</p>
                    <code>{
                        "text": "string",
                        "incremental": true
                    }</code>
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> <span class="path">/api/metrics</span>
                    <p>Calculate text quality metrics</p>
                    <p>Request body:</p>
                    <code>{
                        "text": "string"
                    }</code>
                </div>
                
                <div class="endpoint">
                    <span class="method">GET/POST</span> <span class="path">/api/config</span>
                    <p>Get or update model configuration</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <span class="path">/api/health</span>
                    <p>Health check endpoint</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> <span class="path">/api/stats</span>
                    <p>API usage statistics</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> <span class="path">/api/model/save</span>
                    <p>Save current model state</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> <span class="path">/api/model/load</span>
                    <p>Load a saved model</p>
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> <span class="path">/api/model/reset</span>
                    <p>Reset model to initial state</p>
                </div>
            </body>
            </html>
            """;
    }
    
    /**
     * Start the server
     */
    public void start() {
        server.start();
        System.out.println("=".repeat(60));
        System.out.println("ART Text Generation REST API Server");
        System.out.println("=".repeat(60));
        System.out.println("Server started on port " + PORT);
        System.out.println("API Documentation: http://localhost:" + PORT + "/api/docs");
        System.out.println("Health Check: http://localhost:" + PORT + "/api/health");
        System.out.println("=".repeat(60));
        System.out.println("Ready to accept requests...");
    }
    
    /**
     * Stop the server
     */
    public void stop() {
        server.stop(0);
        System.out.println("Server stopped");
    }
    
    public static void main(String[] args) {
        try {
            RestAPIServer server = new RestAPIServer();
            server.start();
            
            // Add shutdown hook
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                System.out.println("\nShutting down server...");
                server.stop();
            }));
            
            // Keep server running
            Thread.currentThread().join();
            
        } catch (Exception e) {
            System.err.println("Failed to start server: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
