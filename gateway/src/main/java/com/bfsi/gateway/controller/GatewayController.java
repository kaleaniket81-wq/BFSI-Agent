// gateway/src/main/java/com/bfsi/gateway/controller/GatewayController.java
package com.bfsi.gateway.controller;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;

import java.util.Map;

/**
 * API Gateway for BFSI Document Intelligence.
 * Proxies all requests to FastAPI backend.
 * Central point for: JWT auth, rate limiting, request logging, CORS.
 */
@RestController
@CrossOrigin(origins = "*")
public class GatewayController {

    private static final Logger log = LoggerFactory.getLogger(GatewayController.class);

    @Value("${fastapi.url:http://localhost:8000}")
    private String fastapiUrl;

    private final RestTemplate restTemplate = new RestTemplate();

    // ── Health ────────────────────────────────────────────────────────────────
    @GetMapping("/health")
    public Map<String, String> health() {
        return Map.of(
            "status",  "ok",
            "service", "BFSI Gateway",
            "layer",   "Spring Boot 3 / Java 17"
        );
    }

    // ── Ingest ────────────────────────────────────────────────────────────────
    @PostMapping(value = "/api/ingest", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<String> ingest(
            @RequestParam("file")     MultipartFile file,
            @RequestParam(value = "doc_type", defaultValue = "general") String docType) {

        log.info("[Gateway] Ingest: {} ({})", file.getOriginalFilename(), docType);

        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.MULTIPART_FORM_DATA);

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file",     new org.springframework.core.io.ByteArrayResource(file.getBytes()) {
                @Override public String getFilename() { return file.getOriginalFilename(); }
            });
            body.add("doc_type", docType);

            HttpEntity<MultiValueMap<String, Object>> entity = new HttpEntity<>(body, headers);
            return restTemplate.exchange(fastapiUrl + "/api/ingest", HttpMethod.POST, entity, String.class);
        } catch (Exception e) {
            log.error("[Gateway] Ingest error", e);
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                    .body("{\"error\":\"" + e.getMessage() + "\"}");
        }
    }

    // ── RAG Query ─────────────────────────────────────────────────────────────
    @PostMapping("/api/query")
    public ResponseEntity<String> query(@RequestBody String body) {
        log.info("[Gateway] Query request");
        return proxyJson("/api/query", body);
    }

    // ── Analytics ─────────────────────────────────────────────────────────────
    @PostMapping("/api/analytics/query")
    public ResponseEntity<String> analyticsQuery(@RequestBody String body) {
        log.info("[Gateway] Analytics query");
        return proxyJson("/api/analytics/query", body);
    }

    @GetMapping("/api/analytics/portfolio-summary")
    public ResponseEntity<String> portfolioSummary() {
        return proxyGet("/api/analytics/portfolio-summary");
    }

    @GetMapping("/api/analytics/overdue-emi")
    public ResponseEntity<String> overdueEmi(@RequestParam(defaultValue = "30") int days) {
        return proxyGet("/api/analytics/overdue-emi?days=" + days);
    }

    // ── Backend health passthrough ────────────────────────────────────────────
    @GetMapping("/api/health")
    public ResponseEntity<String> backendHealth() {
        return proxyGet("/api/health");
    }

    // ── Helpers ───────────────────────────────────────────────────────────────
    private ResponseEntity<String> proxyJson(String path, String body) {
        HttpHeaders h = new HttpHeaders();
        h.setContentType(MediaType.APPLICATION_JSON);
        try {
            return restTemplate.exchange(
                fastapiUrl + path, HttpMethod.POST,
                new HttpEntity<>(body, h), String.class);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                    .body("{\"error\":\"" + e.getMessage() + "\"}");
        }
    }

    private ResponseEntity<String> proxyGet(String path) {
        try {
            return restTemplate.exchange(
                fastapiUrl + path, HttpMethod.GET,
                new HttpEntity<>(new HttpHeaders()), String.class);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                    .body("{\"error\":\"" + e.getMessage() + "\"}");
        }
    }
}
