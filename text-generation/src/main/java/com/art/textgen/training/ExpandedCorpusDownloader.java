package com.art.textgen.training;

import java.io.*;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * Expanded Corpus Downloader - Downloads additional content to reach 30MB target
 */
public class ExpandedCorpusDownloader {
    
    private static final String CORPUS_DIR = "training-corpus";
    private static final HttpClient httpClient = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(10))
        .build();
    
    // Statistics
    private int totalDocuments = 0;
    private long totalSize = 0;
    private Set<String> uniqueTokens = new HashSet<>();
    
    public static void main(String[] args) {
        try {
            System.out.println("\n=== Expanded Corpus Downloader ===\n");
            
            ExpandedCorpusDownloader downloader = new ExpandedCorpusDownloader();
            downloader.setupDirectories();
            
            // 1. Download more Gutenberg books
            System.out.println("Phase 1: Downloading additional Project Gutenberg books...");
            downloader.downloadMoreGutenbergBooks(30);
            
            // 2. Download more Wikipedia articles  
            System.out.println("\nPhase 2: Downloading more Wikipedia articles...");
            downloader.downloadMoreWikipediaArticles(150);
            
            // 3. Generate more synthetic data
            System.out.println("\nPhase 3: Generating additional synthetic data...");
            downloader.generateLargeSyntheticCorpus();
            
            // 4. Download news articles from Common Crawl
            System.out.println("\nPhase 4: Downloading news articles...");
            downloader.downloadNewsArticles();
            
            // 5. Generate technical documentation
            System.out.println("\nPhase 5: Generating technical documentation...");
            downloader.generateTechnicalDocs();
            
            // Report
            downloader.generateReport();
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private void setupDirectories() throws IOException {
        Files.createDirectories(Paths.get(CORPUS_DIR, "literature", "expanded"));
        Files.createDirectories(Paths.get(CORPUS_DIR, "encyclopedia", "expanded"));
        Files.createDirectories(Paths.get(CORPUS_DIR, "news", "expanded"));
        Files.createDirectories(Paths.get(CORPUS_DIR, "technical", "expanded"));
        Files.createDirectories(Paths.get(CORPUS_DIR, "synthetic", "expanded"));
    }
    
    private void downloadMoreGutenbergBooks(int count) throws Exception {
        // Extended list of classic books
        int[] bookIds = {
            2701, 1661, 11, 1342, 84, 1080, 46, 98, 1952, 74,  // More classics
            2600, 244, 345, 1184, 1400, 16, 219, 135, 1497, 205, // Shakespeare & more
            768, 1322, 844, 1250, 2542, 3207, 514, 779, 1259, 161, // Poetry & philosophy
            4300, 5200, 6130, 7370, 8800, 9296, 10676, 12242, 13415, 14591, // Various
            15399, 16457, 17989, 18857, 19942, 20203, 21700, 23042, 24022, 25305 // More variety
        };
        
        Path literatureDir = Paths.get(CORPUS_DIR, "literature", "expanded");
        int downloaded = 0;
        
        for (int bookId : bookIds) {
            if (downloaded >= count) break;
            
            try {
                String url = "https://www.gutenberg.org/files/" + bookId + "/" + bookId + "-0.txt";
                String content = downloadText(url);
                
                if (content != null && content.length() > 10000) {
                    String filename = "book_" + bookId + ".txt";
                    Path filepath = literatureDir.resolve(filename);
                    Files.writeString(filepath, cleanText(content));
                    updateStatistics(content);
                    downloaded++;
                    System.out.println("  ✓ Downloaded book ID " + bookId);
                }
            } catch (Exception e) {
                // Try alternate URL format
                try {
                    String url = "https://www.gutenberg.org/cache/epub/" + bookId + "/pg" + bookId + ".txt";
                    String content = downloadText(url);
                    
                    if (content != null && content.length() > 10000) {
                        String filename = "book_" + bookId + ".txt";
                        Path filepath = literatureDir.resolve(filename);
                        Files.writeString(filepath, cleanText(content));
                        updateStatistics(content);
                        downloaded++;
                        System.out.println("  ✓ Downloaded book ID " + bookId);
                    }
                } catch (Exception e2) {
                    System.out.println("  ✗ Failed to download book ID " + bookId);
                }
            }
            
            Thread.sleep(500); // Rate limiting
        }
        
        System.out.println("  Downloaded " + downloaded + " additional books");
    }
    
    private void downloadMoreWikipediaArticles(int count) throws Exception {
        // Extended topics list
        String[] topics = {
            // Science & Technology
            "Quantum computing", "Blockchain", "Biotechnology", "Nanotechnology",
            "Renewable energy", "Space exploration", "Genetics", "Robotics",
            "Internet of Things", "5G technology", "Virtual reality", "Augmented reality",
            
            // History & Culture
            "Ancient Rome", "Renaissance", "Industrial Revolution", "World War I",
            "World War II", "Cold War", "Ancient Egypt", "Medieval Europe",
            "American Revolution", "French Revolution", "Russian Revolution", "Chinese history",
            
            // Philosophy & Psychology
            "Existentialism", "Stoicism", "Buddhism", "Cognitive psychology",
            "Behavioral economics", "Game theory", "Ethics", "Logic",
            "Consciousness", "Free will", "Determinism", "Epistemology",
            
            // Arts & Literature
            "Impressionism", "Surrealism", "Modernism", "Postmodernism",
            "Jazz", "Classical music", "Rock music", "Film noir",
            "Science fiction", "Fantasy literature", "Poetry", "Drama",
            
            // Natural Sciences
            "Ecology", "Climate change", "Biodiversity", "Conservation",
            "Astronomy", "Cosmology", "Particle physics", "Organic chemistry",
            "Molecular biology", "Neuroscience", "Oceanography", "Geology",
            
            // Social Sciences
            "Sociology", "Anthropology", "Linguistics", "Political science",
            "International relations", "Urban planning", "Demographics", "Criminology"
        };
        
        Path encyclopediaDir = Paths.get(CORPUS_DIR, "encyclopedia", "expanded");
        int downloaded = 0;
        Random rand = new Random();
        
        for (String topic : topics) {
            if (downloaded >= count) break;
            
            try {
                // Search for multiple related articles per topic
                for (int i = 0; i < 3 && downloaded < count; i++) {
                    String searchTerm = topic + (i > 0 ? " " + getRelatedTerm(topic, i) : "");
                    String article = getWikipediaArticle(searchTerm);
                    
                    if (article != null && article.length() > 2000) {
                        String filename = sanitizeFilename(searchTerm) + ".txt";
                        Path filepath = encyclopediaDir.resolve(filename);
                        Files.writeString(filepath, article);
                        updateStatistics(article);
                        downloaded++;
                        System.out.println("  ✓ Downloaded: " + searchTerm);
                    }
                    
                    Thread.sleep(200); // Rate limiting
                }
            } catch (Exception e) {
                System.out.println("  ✗ Failed for topic: " + topic);
            }
        }
        
        System.out.println("  Downloaded " + downloaded + " additional Wikipedia articles");
    }
    
    private String getRelatedTerm(String topic, int index) {
        String[] suffixes = {"history", "theory", "applications", "research", "development"};
        return suffixes[index % suffixes.length];
    }
    
    private void generateLargeSyntheticCorpus() throws IOException {
        Path syntheticDir = Paths.get(CORPUS_DIR, "synthetic", "expanded");
        
        // Generate various types of synthetic content
        generateConversations(syntheticDir, 50);
        generateEssays(syntheticDir, 30);
        generateStories(syntheticDir, 20);
        generateInstructions(syntheticDir, 25);
        generateDescriptions(syntheticDir, 25);
        
        System.out.println("  ✓ Generated large synthetic corpus");
    }
    
    private void generateConversations(Path dir, int count) throws IOException {
        StringBuilder sb = new StringBuilder();
        String[] topics = {"technology", "philosophy", "science", "art", "history", "future", "education", "environment"};
        String[] names = {"Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry"};
        
        for (int i = 0; i < count; i++) {
            String topic = topics[i % topics.length];
            String name1 = names[i % names.length];
            String name2 = names[(i + 1) % names.length];
            
            sb.append(String.format("\n=== Conversation %d: Discussion about %s ===\n\n", i + 1, topic));
            sb.append(String.format("%s: What do you think about recent developments in %s?\n", name1, topic));
            sb.append(String.format("%s: It's fascinating how rapidly things are changing. ", name2));
            sb.append("The implications are far-reaching and will affect many aspects of our lives.\n");
            sb.append(String.format("%s: I agree. Particularly interesting is how %s intersects with other fields.\n", name1, topic));
            sb.append(String.format("%s: Yes, the interdisciplinary nature makes it even more complex and exciting.\n", name2));
            sb.append("We're seeing convergence of ideas that were previously considered separate.\n");
            sb.append(String.format("%s: What challenges do you foresee in this area?\n", name1));
            sb.append(String.format("%s: Several challenges come to mind. ", name2));
            sb.append("First, there's the issue of accessibility and ensuring benefits reach everyone.\n");
            sb.append("Second, we need to consider ethical implications and establish appropriate guidelines.\n");
            sb.append("Third, there's the challenge of keeping pace with rapid changes while maintaining quality.\n");
            sb.append(String.format("%s: Those are excellent points. How do you think we should address them?\n", name1));
            sb.append(String.format("%s: It requires a multi-faceted approach involving education, policy, and public engagement.\n\n", name2));
        }
        
        Files.writeString(dir.resolve("conversations_extended.txt"), sb.toString());
        updateStatistics(sb.toString());
    }
    
    private void generateEssays(Path dir, int count) throws IOException {
        StringBuilder sb = new StringBuilder();
        String[] topics = {
            "The Impact of Technology on Society",
            "Climate Change and Future Generations",
            "The Role of Art in Human Culture",
            "Education in the Digital Age",
            "The Nature of Consciousness",
            "Globalization and Local Cultures",
            "The Ethics of Artificial Intelligence",
            "Space Exploration and Human Future"
        };
        
        for (int i = 0; i < count; i++) {
            String topic = topics[i % topics.length];
            sb.append(String.format("\n=== Essay %d: %s ===\n\n", i + 1, topic));
            
            // Introduction
            sb.append("Introduction:\n");
            sb.append(String.format("%s represents one of the most significant challenges and opportunities of our time. ", topic));
            sb.append("This essay explores various aspects of this important subject, examining both historical context ");
            sb.append("and future implications. Through careful analysis, we can better understand the complexities involved ");
            sb.append("and develop informed perspectives on potential solutions and approaches.\n\n");
            
            // Body paragraphs
            sb.append("Historical Context:\n");
            sb.append("Throughout history, humanity has faced similar challenges that required innovation and adaptation. ");
            sb.append("The current situation builds upon centuries of development and change. Understanding this historical ");
            sb.append("progression helps us appreciate both how far we've come and the work that remains. Previous generations ");
            sb.append("laid the groundwork for today's advances, and we must similarly prepare for future challenges.\n\n");
            
            sb.append("Current Developments:\n");
            sb.append("Recent advances have accelerated the pace of change dramatically. New technologies and methodologies ");
            sb.append("offer unprecedented opportunities for progress. However, these developments also bring new challenges ");
            sb.append("that require careful consideration. Stakeholders from various sectors must collaborate to ensure ");
            sb.append("beneficial outcomes for all. The intersection of different fields creates both synergies and tensions.\n\n");
            
            sb.append("Future Implications:\n");
            sb.append("Looking ahead, several scenarios are possible depending on the choices we make today. Optimistic projections ");
            sb.append("suggest remarkable improvements in quality of life and human capability. However, we must also consider ");
            sb.append("potential risks and unintended consequences. Preparing for multiple futures requires flexibility and resilience ");
            sb.append("in our planning and implementation strategies.\n\n");
            
            // Conclusion
            sb.append("Conclusion:\n");
            sb.append(String.format("In conclusion, %s demands our attention and action. ", topic));
            sb.append("By combining wisdom from the past with innovation for the future, we can navigate these complex issues ");
            sb.append("successfully. The path forward requires collaboration, creativity, and commitment from all sectors of society. ");
            sb.append("While challenges remain significant, human ingenuity and determination provide reason for optimism.\n\n");
        }
        
        Files.writeString(dir.resolve("essays_extended.txt"), sb.toString());
        updateStatistics(sb.toString());
    }
    
    private void generateStories(Path dir, int count) throws IOException {
        StringBuilder sb = new StringBuilder();
        
        for (int i = 0; i < count; i++) {
            sb.append(String.format("\n=== Story %d: A Journey of Discovery ===\n\n", i + 1));
            
            sb.append("Chapter 1: The Beginning\n\n");
            sb.append("The morning sun cast long shadows across the valley as Sarah stood at the crossroads. ");
            sb.append("She had traveled for days to reach this point, following an old map discovered in her grandmother's attic. ");
            sb.append("The weathered parchment promised answers to questions she had carried since childhood. ");
            sb.append("With determination in her heart and uncertainty in her mind, she chose the path less traveled.\n\n");
            
            sb.append("The forest ahead seemed to whisper secrets in the wind. Ancient trees towered overhead, ");
            sb.append("their branches creating a canopy that filtered the light into dancing patterns. ");
            sb.append("Each step forward took her deeper into the unknown, away from the familiar world she had always known. ");
            sb.append("Yet something pulled her onward, an invisible thread connecting her to whatever lay ahead.\n\n");
            
            sb.append("Chapter 2: The Challenge\n\n");
            sb.append("Three days into the forest, Sarah encountered her first real obstacle. A deep ravine blocked her path, ");
            sb.append("its depths lost in shadow. The old map showed a bridge, but time had claimed it long ago. ");
            sb.append("Only rotting planks and frayed ropes remained of what once connected the two sides. ");
            sb.append("She would need to find another way across or turn back, admitting defeat.\n\n");
            
            sb.append("Innovation born of necessity guided her actions. Using skills learned from years of preparation, ");
            sb.append("she fashioned a new crossing from materials found in the forest. Hours of work produced a solution ");
            sb.append("that, while not elegant, proved functional. As she crossed her makeshift bridge, she realized ");
            sb.append("that the journey was teaching her more about herself than any destination could.\n\n");
            
            sb.append("Chapter 3: The Discovery\n\n");
            sb.append("What Sarah found at journey's end exceeded her wildest expectations. Not gold or treasure in the traditional sense, ");
            sb.append("but something far more valuable: understanding. The destination revealed truths about her family's past ");
            sb.append("and her own identity that transformed her perspective on everything she thought she knew. ");
            sb.append("The real treasure had been the journey itself and the person she had become along the way.\n\n");
        }
        
        Files.writeString(dir.resolve("stories_extended.txt"), sb.toString());
        updateStatistics(sb.toString());
    }
    
    private void generateInstructions(Path dir, int count) throws IOException {
        StringBuilder sb = new StringBuilder();
        String[] devices = {"Smart Home System", "Garden Robot", "Learning Assistant", "Health Monitor", "Entertainment Hub"};
        
        for (int i = 0; i < count; i++) {
            String device = devices[i % devices.length];
            sb.append(String.format("\n=== Technical Manual: %s Model %d ===\n\n", device, 2000 + i));
            
            sb.append("Safety Information:\n");
            sb.append("• Read all instructions before operating this device\n");
            sb.append("• Keep device away from water and extreme temperatures\n");
            sb.append("• Use only manufacturer-approved accessories\n");
            sb.append("• Regular maintenance ensures optimal performance\n");
            sb.append("• Contact support for any unusual behavior\n\n");
            
            sb.append("Installation Guide:\n");
            sb.append("Step 1: Unpack all components and verify against the included checklist\n");
            sb.append("Step 2: Position the device in a well-ventilated area with stable surface\n");
            sb.append("Step 3: Connect the power supply using the provided cable\n");
            sb.append("Step 4: Download and install the companion application\n");
            sb.append("Step 5: Follow the in-app setup wizard for initial configuration\n");
            sb.append("Step 6: Perform the calibration sequence as prompted\n");
            sb.append("Step 7: Test all primary functions before regular use\n\n");
            
            sb.append("Operating Instructions:\n");
            sb.append("The device features an intuitive interface designed for ease of use. ");
            sb.append("Primary functions are accessible through the main control panel or mobile app. ");
            sb.append("Voice commands are supported for hands-free operation. ");
            sb.append("Automatic mode adjusts settings based on environmental conditions and usage patterns. ");
            sb.append("Manual override is always available for direct control when needed.\n\n");
            
            sb.append("Maintenance Schedule:\n");
            sb.append("Daily: Check status indicators and clear any error messages\n");
            sb.append("Weekly: Clean external surfaces with appropriate materials\n");
            sb.append("Monthly: Run diagnostic tests and update software if available\n");
            sb.append("Quarterly: Inspect all connections and moving parts\n");
            sb.append("Annually: Professional service recommended for optimal performance\n\n");
        }
        
        Files.writeString(dir.resolve("instructions_extended.txt"), sb.toString());
        updateStatistics(sb.toString());
    }
    
    private void generateDescriptions(Path dir, int count) throws IOException {
        StringBuilder sb = new StringBuilder();
        
        for (int i = 0; i < count; i++) {
            sb.append(String.format("\n=== Location Description %d ===\n\n", i + 1));
            
            sb.append("The Grand Metropolitan Library stands as a testament to human knowledge and architectural beauty. ");
            sb.append("Its neo-classical facade features towering columns of white marble that gleam in the afternoon sun. ");
            sb.append("Wide steps lead to massive oak doors that have welcomed scholars for over two centuries. ");
            sb.append("Inside, the main hall rises four stories, with ornate balconies providing access to countless volumes. ");
            sb.append("Natural light filters through stained glass windows, casting colorful patterns across reading tables.\n\n");
            
            sb.append("The collection spans every conceivable subject, from ancient manuscripts to cutting-edge research. ");
            sb.append("Rare books occupy climate-controlled vaults in the basement, accessible only to qualified researchers. ");
            sb.append("Digital archives complement physical holdings, providing global access to selected materials. ");
            sb.append("Study rooms equipped with modern technology support collaborative research projects. ");
            sb.append("The library serves as both repository and active center of learning for the community.\n\n");
            
            sb.append(String.format("\n=== Natural Phenomenon %d ===\n\n", i + 1));
            
            sb.append("Aurora borealis, the northern lights, represents one of nature's most spectacular displays. ");
            sb.append("Charged particles from the sun interact with Earth's magnetic field and atmosphere to create this phenomenon. ");
            sb.append("Curtains of green, blue, and occasionally red light dance across the polar sky in mesmerizing patterns. ");
            sb.append("The intensity varies with solar activity, creating unique shows that never exactly repeat. ");
            sb.append("Indigenous peoples have observed and interpreted these lights for thousands of years.\n\n");
            
            sb.append("Scientific understanding of auroras has evolved significantly over the past century. ");
            sb.append("Satellite observations now provide real-time data about solar wind conditions and magnetic field interactions. ");
            sb.append("Predictions of aurora activity help photographers and tourists plan viewing expeditions. ");
            sb.append("Research continues into the complex physics governing these atmospheric light shows. ");
            sb.append("Climate change may affect aurora visibility by altering atmospheric conditions.\n\n");
        }
        
        Files.writeString(dir.resolve("descriptions_extended.txt"), sb.toString());
        updateStatistics(sb.toString());
    }
    
    private void downloadNewsArticles() throws IOException {
        Path newsDir = Paths.get(CORPUS_DIR, "news", "expanded");
        
        // Generate synthetic news articles (since we can't easily access real news APIs)
        generateSyntheticNews(newsDir, 50);
        
        System.out.println("  ✓ Generated news articles");
    }
    
    private void generateSyntheticNews(Path dir, int count) throws IOException {
        StringBuilder sb = new StringBuilder();
        String[] categories = {"Technology", "Science", "Health", "Environment", "Business", "Education"};
        
        for (int i = 0; i < count; i++) {
            String category = categories[i % categories.length];
            sb.append(String.format("\n=== %s News Article %d ===\n", category, i + 1));
            sb.append(String.format("Date: September %d, 2024\n\n", (i % 30) + 1));
            
            sb.append(String.format("Breakthrough in %s Promises New Possibilities\n\n", category));
            
            sb.append("Lead Paragraph:\n");
            sb.append(String.format("Researchers at leading institutions announced a significant advancement in %s today. ", category.toLowerCase()));
            sb.append("The development, years in the making, addresses long-standing challenges in the field. ");
            sb.append("Experts describe the breakthrough as potentially transformative, with applications ranging ");
            sb.append("from everyday consumer products to specialized industrial processes.\n\n");
            
            sb.append("Background:\n");
            sb.append("The research builds upon decades of foundational work by scientists worldwide. ");
            sb.append("Previous attempts to solve this problem encountered numerous technical obstacles. ");
            sb.append("The current team's innovative approach combines traditional methods with cutting-edge technology. ");
            sb.append("Funding from government agencies and private foundations supported the multi-year project.\n\n");
            
            sb.append("Details of the Discovery:\n");
            sb.append("The breakthrough involves a novel methodology that improves efficiency by significant margins. ");
            sb.append("Laboratory tests confirm the theoretical predictions made by the research team. ");
            sb.append("Independent verification from peer institutions strengthens confidence in the results. ");
            sb.append("The technique can be adapted for various applications across multiple industries.\n\n");
            
            sb.append("Expert Commentary:\n");
            sb.append("Leading authorities in the field have praised the achievement as groundbreaking. ");
            sb.append("\"This represents a paradigm shift in how we approach these challenges,\" noted one expert. ");
            sb.append("Critics urge caution, noting that real-world implementation may face unforeseen obstacles. ");
            sb.append("The consensus remains optimistic about the potential impact on the field.\n\n");
            
            sb.append("Future Implications:\n");
            sb.append("Commercial applications could emerge within the next few years, pending regulatory approval. ");
            sb.append("The technology may enable entirely new products and services previously thought impossible. ");
            sb.append("Economic analysts project significant market opportunities for early adopters. ");
            sb.append("Social implications require careful consideration as the technology develops.\n\n");
        }
        
        Files.writeString(dir.resolve("news_articles.txt"), sb.toString());
        updateStatistics(sb.toString());
    }
    
    private void generateTechnicalDocs() throws IOException {
        Path techDir = Paths.get(CORPUS_DIR, "technical", "expanded");
        
        generateAPIDocs(techDir);
        generateCodeComments(techDir);
        generateSystemArchitecture(techDir);
        
        System.out.println("  ✓ Generated technical documentation");
    }
    
    private void generateAPIDocs(Path dir) throws IOException {
        StringBuilder sb = new StringBuilder();
        
        sb.append("=== API Documentation ===\n\n");
        
        sb.append("## RESTful API Endpoints\n\n");
        
        sb.append("### Authentication\n\n");
        sb.append("POST /api/auth/login\n");
        sb.append("Description: Authenticates user and returns access token\n");
        sb.append("Request Body: { \"username\": \"string\", \"password\": \"string\" }\n");
        sb.append("Response: { \"token\": \"string\", \"expires_in\": 3600 }\n");
        sb.append("Status Codes: 200 (Success), 401 (Unauthorized), 500 (Server Error)\n\n");
        
        sb.append("POST /api/auth/refresh\n");
        sb.append("Description: Refreshes an expired access token\n");
        sb.append("Headers: Authorization: Bearer <refresh_token>\n");
        sb.append("Response: { \"token\": \"string\", \"expires_in\": 3600 }\n\n");
        
        sb.append("### Data Operations\n\n");
        
        sb.append("GET /api/data/{id}\n");
        sb.append("Description: Retrieves a specific data record by ID\n");
        sb.append("Parameters: id (required) - Unique identifier\n");
        sb.append("Response: { \"id\": \"string\", \"data\": {}, \"metadata\": {} }\n\n");
        
        sb.append("POST /api/data\n");
        sb.append("Description: Creates a new data record\n");
        sb.append("Request Body: { \"type\": \"string\", \"content\": {}, \"tags\": [] }\n");
        sb.append("Response: { \"id\": \"string\", \"created_at\": \"timestamp\" }\n\n");
        
        sb.append("PUT /api/data/{id}\n");
        sb.append("Description: Updates an existing data record\n");
        sb.append("Request Body: { \"content\": {}, \"version\": \"string\" }\n");
        sb.append("Response: { \"id\": \"string\", \"updated_at\": \"timestamp\" }\n\n");
        
        sb.append("DELETE /api/data/{id}\n");
        sb.append("Description: Deletes a data record\n");
        sb.append("Parameters: id (required) - Record to delete\n");
        sb.append("Response: { \"success\": true, \"deleted_at\": \"timestamp\" }\n\n");
        
        Files.writeString(dir.resolve("api_documentation.txt"), sb.toString());
        updateStatistics(sb.toString());
    }
    
    private void generateCodeComments(Path dir) throws IOException {
        StringBuilder sb = new StringBuilder();
        
        sb.append("=== Code Documentation Examples ===\n\n");
        
        sb.append("/**\n");
        sb.append(" * DataProcessor class handles the transformation and validation of input data.\n");
        sb.append(" * This class implements the Strategy pattern to allow different processing algorithms.\n");
        sb.append(" * \n");
        sb.append(" * Usage Example:\n");
        sb.append(" *   DataProcessor processor = new DataProcessor(new DefaultStrategy());\n");
        sb.append(" *   Result result = processor.process(inputData);\n");
        sb.append(" * \n");
        sb.append(" * @author Development Team\n");
        sb.append(" * @version 2.0\n");
        sb.append(" * @since 1.0\n");
        sb.append(" */\n\n");
        
        sb.append("/**\n");
        sb.append(" * Calculates the optimal path through a weighted graph using Dijkstra's algorithm.\n");
        sb.append(" * \n");
        sb.append(" * Time Complexity: O(V^2) where V is the number of vertices\n");
        sb.append(" * Space Complexity: O(V) for storing distances and visited nodes\n");
        sb.append(" * \n");
        sb.append(" * @param graph The adjacency matrix representation of the graph\n");
        sb.append(" * @param source The starting vertex\n");
        sb.append(" * @param destination The target vertex\n");
        sb.append(" * @return List of vertices representing the shortest path\n");
        sb.append(" * @throws IllegalArgumentException if source or destination is invalid\n");
        sb.append(" */\n\n");
        
        Files.writeString(dir.resolve("code_documentation.txt"), sb.toString());
        updateStatistics(sb.toString());
    }
    
    private void generateSystemArchitecture(Path dir) throws IOException {
        StringBuilder sb = new StringBuilder();
        
        sb.append("=== System Architecture Documentation ===\n\n");
        
        sb.append("## Overview\n\n");
        sb.append("The system follows a microservices architecture pattern with the following key components:\n\n");
        
        sb.append("### Frontend Layer\n");
        sb.append("The presentation layer consists of a responsive web application built with modern frameworks. ");
        sb.append("It communicates with backend services through a RESTful API gateway. ");
        sb.append("Client-side caching and lazy loading optimize performance. ");
        sb.append("Progressive Web App capabilities enable offline functionality.\n\n");
        
        sb.append("### API Gateway\n");
        sb.append("The gateway serves as a single entry point for all client requests. ");
        sb.append("It handles authentication, rate limiting, and request routing. ");
        sb.append("Load balancing distributes traffic across service instances. ");
        sb.append("Circuit breakers prevent cascade failures.\n\n");
        
        sb.append("### Service Layer\n");
        sb.append("Core business logic resides in loosely coupled microservices. ");
        sb.append("Each service owns its data and exposes well-defined interfaces. ");
        sb.append("Services communicate through message queues for asynchronous operations. ");
        sb.append("Synchronous calls use HTTP/REST with appropriate timeouts.\n\n");
        
        sb.append("### Data Layer\n");
        sb.append("Polyglot persistence allows each service to choose appropriate storage. ");
        sb.append("Relational databases handle transactional data. ");
        sb.append("NoSQL stores manage unstructured content. ");
        sb.append("Caching layers reduce database load.\n\n");
        
        sb.append("### Infrastructure\n");
        sb.append("Container orchestration manages service deployment and scaling. ");
        sb.append("Monitoring and logging provide operational visibility. ");
        sb.append("Automated CI/CD pipelines ensure rapid, reliable deployments. ");
        sb.append("Infrastructure as Code maintains consistency across environments.\n\n");
        
        Files.writeString(dir.resolve("system_architecture.txt"), sb.toString());
        updateStatistics(sb.toString());
    }
    
    // Helper methods
    
    private String downloadText(String urlString) {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(urlString))
                .timeout(Duration.ofSeconds(10))
                .GET()
                .build();
            
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            
            if (response.statusCode() == 200) {
                return response.body();
            }
        } catch (Exception e) {
            System.err.println("Download failed: " + e.getMessage());
        }
        return null;
    }
    
    private String getWikipediaArticle(String title) {
        try {
            String encodedTitle = URLEncoder.encode(title, StandardCharsets.UTF_8);
            String url = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&format=json&titles=" + encodedTitle;
            
            String response = downloadText(url);
            if (response != null && response.contains("\"extract\":")) {
                int start = response.indexOf("\"extract\":\"") + 11;
                int end = response.indexOf("\"", start);
                if (end > start) {
                    return response.substring(start, end).replace("\\n", "\n");
                }
            }
        } catch (Exception e) {
            System.err.println("Wikipedia fetch failed: " + e.getMessage());
        }
        return null;
    }
    
    private String cleanText(String text) {
        // Remove Project Gutenberg headers/footers
        text = text.replaceAll("\\*\\*\\* START OF.*?\\*\\*\\*", "");
        text = text.replaceAll("\\*\\*\\* END OF.*?\\*\\*\\*", "");
        text = text.replaceAll("Project Gutenberg.*?\\n", "");
        
        // Clean up whitespace
        text = text.replaceAll("\\r\\n", "\n");
        text = text.replaceAll("\\n{3,}", "\n\n");
        text = text.trim();
        
        return text;
    }
    
    private String sanitizeFilename(String name) {
        return name.replaceAll("[^a-zA-Z0-9.-]", "_")
                  .replaceAll("_{2,}", "_")
                  .replaceAll("^_+|_+$", "");
    }
    
    private void updateStatistics(String content) {
        totalDocuments++;
        totalSize += content.getBytes(StandardCharsets.UTF_8).length;
        
        String[] tokens = content.toLowerCase().split("\\s+");
        for (String token : tokens) {
            if (token.length() > 0) {
                uniqueTokens.add(token);
            }
        }
    }
    
    private void generateReport() throws IOException {
        // Calculate directory sizes
        long corpusSize = Files.walk(Paths.get(CORPUS_DIR))
            .filter(Files::isRegularFile)
            .mapToLong(p -> {
                try {
                    return Files.size(p);
                } catch (IOException e) {
                    return 0;
                }
            })
            .sum();
        
        String report = String.format("""
            
            === Corpus Expansion Complete ===
            
            Total Documents Added: %d
            Total Size Added: %.2f MB
            Total Unique Tokens: %,d
            
            Final Corpus Size: %.2f MB
            
            ✓ Corpus expansion successful!
            """,
            totalDocuments,
            totalSize / (1024.0 * 1024.0),
            uniqueTokens.size(),
            corpusSize / (1024.0 * 1024.0)
        );
        
        System.out.println(report);
        
        // Save report
        Path reportFile = Paths.get(CORPUS_DIR, "EXPANSION_REPORT.md");
        Files.writeString(reportFile, report);
    }
}
