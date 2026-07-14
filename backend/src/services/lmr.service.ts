import { ollamaChatService } from "./ollamaChat.service";
import { vectorDBService } from "./vectordb.service";
import { documentService } from "./document.service";
import { nllbService } from "./nllb.service";
import { languageService, SupportedLanguageCode } from "./language.service";
import { LanguageCode, SUPPORTED_LANGUAGES } from "../config/constants";
import { env } from "../config/env";

// Enhanced key topic with description for quick recall
export interface KeyTopic {
  name: string;
  description: string;
}

// Enhanced important concept with 5 bullet point descriptions
export interface ImportantConcept {
  name: string;
  points: string[];
}

export interface LMRSummary {
  // Structured summary format for last-minute revision
  introduction: string; // Short intro paragraph (2-3 sentences)
  summaryPoints: string[]; // Bullet points with descriptions
  conclusion: string; // Conclusion paragraph (1-2 sentences)
  // Legacy field for backward compatibility
  summary?: string;
  // Enhanced fields with descriptions
  keyTopics: KeyTopic[];
  importantConcepts: ImportantConcept[];
  language: string;
}

export interface LMRQuestion {
  id: number;
  question: string;
  answer: string;
  subject: string;
  difficulty: "Easy" | "Medium" | "Hard";
  pageReference?: number;
}

export interface LMRQuiz {
  id: number;
  question: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
  difficulty: "Easy" | "Medium" | "Hard";
  subject: string;
}

export interface LMRRecallNote {
  topic: string;
  keyPoints: string[];
  quickFacts: string[];
  mnemonics?: string[];
}

/**
 * Task types for the two-layer AI approach
 */
type LMRTaskType = "summary" | "questions" | "quiz" | "recallNotes";

/**
 * Compressed context from Layer 1
 * Updated to be UNIVERSAL for all task types
 */
interface CompressedContext {
  mainTopics: string[];
  keyFacts: string[];
  importantConcepts: string[];
  relevantExamples?: string[];
  pageReferences?: { topic: string; page: number }[];
  // New fields for Universal Context optimization
  definitions?: string[];
  numericalData?: string[];
}

/**
 * Document metrics for dynamic content quantity calculation
 */
interface DocumentMetrics {
  wordCount: number;
  charCount: number;
  pageCount: number;
  paragraphCount: number;
  estimatedTopics: number;
  contentDensity: "light" | "medium" | "dense";
  // Dynamic quantities based on document analysis
  recommendedConceptCount: number; // 5-15 based on content
  recommendedRecallTopicCount: number; // 6-15 based on content
  recommendedQuestionCount: number; // 10-15 based on content
  recommendedQuizCount: number; // 10-15 based on content
}

export class LMRService {
  // 1. CACHE: Store processed context by fileId to prevent redundant Layer 1 runs
  private contextCache: Map<string, CompressedContext> = new Map();

  /**
   * Helper: Sanitize JSON string from AI responses
   * Handles Python-style syntax (None, True, False) and malformed JSON
   */
  private sanitizeJSON(jsonString: string): string {
    let cleaned = jsonString;

    // Remove markdown code blocks
    cleaned = cleaned.replace(/```json\s*/g, "");
    cleaned = cleaned.replace(/```\s*/g, "");

    // Fix 1: Aggressively insert missing commas between objects/arrays
    // Matches "}" followed by newline/space followed by "{" -> "}, {"
    cleaned = cleaned.replace(/}(\s*){/g, "},$1{");
    // Matches "]" followed by newline/space followed by "[" -> "], ["
    cleaned = cleaned.replace(/](\s*)\[/g, "],$1[");

    // Fix 2: Handle python-style constants
    cleaned = cleaned.replace(/:\s*None\b/g, ": null");
    cleaned = cleaned.replace(/:\s*True\b/g, ": true");
    cleaned = cleaned.replace(/:\s*False\b/g, ": false");

    // Fix 3: Remove trailing commas (run multiple times for nested structures)
    // Replaces ",}" with "}" and ",]" with "]"
    cleaned = cleaned.replace(/,(\s*[}\]])/g, "$1");
    cleaned = cleaned.replace(/,(\s*[}\]])/g, "$1");

    // Fix 4: Fix common malformed closures like "],}" -> "}]"
    cleaned = cleaned.replace(/],\s*}/g, "]}");
    cleaned = cleaned.replace(/},\s*]/g, "}]");

    // Fix 5: Ensure property keys are quoted (fixes { key: "val" } -> { "key": "val" })
    cleaned = cleaned.replace(/([{,]\s*)([a-zA-Z0-9_]+?)\s*:/g, '$1"$2":');

    return cleaned;
  }

  /**
   * Helper: Extract and parse JSON from AI response
   * Handles truncated responses, trailing text, and malformed structures
   */
  private extractAndParseJSON(response: string, isArray: boolean = false): any {
    try {
      // Remove DeepSeek thinking tags if present
      let cleanedResponse = response.replace(/<think>[\s\S]*?<\/think>/g, "");

      // Remove any text before the JSON starts
      const jsonStartChar = isArray ? "[" : "{";
      const jsonStartIndex = cleanedResponse.indexOf(jsonStartChar);
      if (jsonStartIndex === -1) {
        throw new Error(
          `No JSON ${isArray ? "array" : "object"} found in response`,
        );
      }
      cleanedResponse = cleanedResponse.substring(jsonStartIndex);

      // Find the proper end of JSON by counting brackets
      const openChar = isArray ? "[" : "{";
      const closeChar = isArray ? "]" : "}";
      let depth = 0;
      let inString = false;
      let escapeNext = false;
      let jsonEndIndex = -1;

      for (let i = 0; i < cleanedResponse.length; i++) {
        const char = cleanedResponse[i];

        if (escapeNext) {
          escapeNext = false;
          continue;
        }
        if (char === "\\") {
          escapeNext = true;
          continue;
        }
        if (char === '"') {
          inString = !inString;
          continue;
        }
        if (!inString) {
          if (char === openChar) {
            depth++;
          } else if (char === closeChar) {
            depth--;
            if (depth === 0) {
              jsonEndIndex = i + 1;
              break;
            }
          }
        }
      }

      let jsonStr: string;

      if (jsonEndIndex === -1) {
        // JSON is truncated - try to repair by closing open brackets
        console.warn("⚠️ JSON appears truncated, attempting repair...");
        jsonStr = cleanedResponse;

        // Count unclosed brackets
        let openBraces = 0;
        let openBrackets = 0;
        inString = false;
        escapeNext = false;

        for (const char of jsonStr) {
          if (escapeNext) {
            escapeNext = false;
            continue;
          }
          if (char === "\\") {
            escapeNext = true;
            continue;
          }
          if (char === '"') {
            inString = !inString;
            continue;
          }
          if (!inString) {
            if (char === "{") openBraces++;
            else if (char === "}") openBraces--;
            else if (char === "[") openBrackets++;
            else if (char === "]") openBrackets--;
          }
        }

        // Close any unclosed strings
        if (inString) {
          jsonStr += '"';
        }

        // Remove any trailing incomplete values
        jsonStr = jsonStr.replace(/,\s*$/, "");
        jsonStr = jsonStr.replace(/:\s*$/, ": null");
        jsonStr = jsonStr.replace(/:\s*"[^"]*$/, ': ""');

        // Close unclosed brackets
        for (let i = 0; i < openBrackets; i++) jsonStr += "]";
        for (let i = 0; i < openBraces; i++) jsonStr += "}";

        console.log(
          `🔧 Repaired JSON: closed ${openBrackets} brackets, ${openBraces} braces`,
        );
      } else {
        jsonStr = cleanedResponse.substring(0, jsonEndIndex);
      }

      const sanitized = this.sanitizeJSON(jsonStr);

      if (jsonStr !== sanitized) {
        console.log("🔧 JSON sanitized.");
      }

      return JSON.parse(sanitized);
    } catch (error) {
      console.error("❌ JSON extraction/parsing failed:", error);
      throw new Error(
        `Failed to parse AI response: ${
          error instanceof Error ? error.message : "Unknown error"
        }`,
      );
    }
  }

  /**
   * DOCUMENT METRICS CALCULATOR
   */
  private calculateDocumentMetrics(
    content: string,
    pageCount: number = 1,
  ): DocumentMetrics {
    const wordCount = content.split(/\s+/).filter((w) => w.length > 0).length;
    const charCount = content.length;
    const paragraphCount = content
      .split(/\n\n+/)
      .filter((p) => p.trim().length > 0).length;

    const headingMatches = content.match(/^#+\s+.+$|^[A-Z][^.!?]*:$/gm) || [];
    const estimatedTopics = Math.max(
      3,
      Math.min(12, headingMatches.length || Math.ceil(paragraphCount / 3)),
    );

    let contentDensity: "light" | "medium" | "dense";
    const wordsPerPage = pageCount > 0 ? wordCount / pageCount : wordCount;

    if (wordsPerPage < 300 || wordCount < 500) {
      contentDensity = "light";
    } else if (wordsPerPage < 600 || wordCount < 2000) {
      contentDensity = "medium";
    } else {
      contentDensity = "dense";
    }

    let recommendedConceptCount: number;
    let recommendedQuestionCount: number;
    let recommendedQuizCount: number;
    let recommendedRecallTopicCount: number;

    switch (contentDensity) {
      case "light":
        recommendedConceptCount = 8;
        recommendedRecallTopicCount = 10;
        recommendedQuestionCount = 10;
        recommendedQuizCount = 10;
        break;
      case "medium":
        recommendedConceptCount = 8;
        recommendedRecallTopicCount = 12;
        recommendedQuestionCount = 12;
        recommendedQuizCount = 12;
        break;
      case "dense":
        recommendedConceptCount = 8;
        recommendedRecallTopicCount = 15;
        recommendedQuestionCount = 15;
        recommendedQuizCount = 15;
        break;
    }

    if (estimatedTopics >= 8) {
      recommendedRecallTopicCount = Math.min(
        15,
        recommendedRecallTopicCount + 3,
      );
      recommendedQuestionCount = Math.min(15, recommendedQuestionCount + 2);
    }

    console.log(
      `📊 Document Metrics: ${wordCount} words, ${paragraphCount} paragraphs, ~${estimatedTopics} topics, density: ${contentDensity}`,
    );
    console.log(
      `📊 Dynamic Quantities: ${recommendedConceptCount} concepts, ${recommendedRecallTopicCount} recall topics, ${recommendedQuestionCount} questions`,
    );

    return {
      wordCount,
      charCount,
      pageCount,
      paragraphCount,
      estimatedTopics,
      contentDensity,
      recommendedConceptCount,
      recommendedRecallTopicCount,
      recommendedQuestionCount,
      recommendedQuizCount,
    };
  }

  /**
   * 2. OPTIMIZED: Universal Context Generator
   * Runs ONCE per document, extracts EVERYTHING needed for all 4 features.
   */
  private async getUniversalContext(
    fileId: string,
    documentContent: string,
    language: string,
  ): Promise<CompressedContext> {
    // A. Check Cache First
    if (this.contextCache.has(fileId)) {
      console.log(`⚡ Using cached Layer 1 context for file ${fileId}`);
      return this.contextCache.get(fileId)!;
    }

    console.log("🔄 Layer 1: Generating Universal Context (First time run)...");

    // B. Universal Prompt - Extracts data for Summary, Questions, Quiz, AND Notes at once
    const prompt = `You are an expert content analyzer. Extract a comprehensive structured analysis from this document to support study material generation.

DOCUMENT CONTENT:
${documentContent.substring(0, 15000)}${
      documentContent.length > 15000 ? "\n...[truncated]" : ""
    }

TASK: Extract a MASTER JSON object containing:
1. "mainTopics": List of 8-12 major headings/themes.
2. "keyFacts": List of 15-20 specific, testable facts (dates, names, core truths).
3. "importantConcepts": List of 8-10 complex ideas that require explanation.
4. "relevantExamples": Real-world applications or case studies mentioned.
5. "definitions": Key terms and their direct definitions (crucial for Quizzes).
6. "numericalData": Formulas, statistics, or years (crucial for Questions).

NOTATION RULES:
- Math: Use plain text (e.g., "1/3") NOT LaTeX.
- Language: ${language}

Output ONLY valid JSON (no markdown, no thinking tags):
{
  "mainTopics": ["topic1", "topic2"],
  "keyFacts": ["fact1", "fact2"],
  "importantConcepts": ["concept1"],
  "relevantExamples": [],
  "definitions": [],
  "numericalData": []
}`;

    try {
      // Increased token limit for richer context
      const result = await ollamaChatService.generateWithMaxOutput(
        prompt,
        1500,
      );
      const parsed = this.extractAndParseJSON(
        result.answer,
        false,
      ) as CompressedContext;

      // Validate/Sanitize output structure
      const sanitizedContext: CompressedContext = {
        mainTopics: parsed.mainTopics || [],
        keyFacts: parsed.keyFacts || [],
        importantConcepts: parsed.importantConcepts || [],
        relevantExamples: parsed.relevantExamples || [],
        definitions: parsed.definitions || [],
        numericalData: parsed.numericalData || [],
      };

      console.log(
        `✅ Layer 1 Complete: Extracted ${sanitizedContext.mainTopics.length} topics, ${sanitizedContext.keyFacts.length} facts.`,
      );

      // C. Save to Cache
      this.contextCache.set(fileId, sanitizedContext);

      return sanitizedContext;
    } catch (error) {
      console.error("❌ Layer 1 Failed:", error);
      // Fallback context to prevent app crash
      return {
        mainTopics: ["General Content"],
        keyFacts: ["Content available for analysis"],
        importantConcepts: ["Key concepts"],
        definitions: [],
        numericalData: [],
        relevantExamples: [],
      };
    }
  }

  /**
   * LAYER 2: JSON Generator
   * Uses the Universal Context to generate specific schemas
   */
  private async generateJSONFromContext<T>(
    compressedContext: CompressedContext,
    taskType: LMRTaskType,
    language: string,
    schema: {
      description: string;
      jsonTemplate: string;
      isArray: boolean;
    },
    additionalParams?: { count?: number },
  ): Promise<T> {
    const contextSummary = `
EXTRACTED CONTENT:
Main Topics: ${(compressedContext.mainTopics || []).join(", ")}

Key Facts:
${(compressedContext.keyFacts || []).map((f, i) => `${i + 1}. ${f}`).join("\n")}

Important Concepts:
${(compressedContext.importantConcepts || [])
  .map((c, i) => `${i + 1}. ${c}`)
  .join("\n")}

Definitions:
${(compressedContext.definitions || [])
  .map((d, i) => `${i + 1}. ${d}`)
  .join("\n")}

${
  compressedContext.relevantExamples?.length
    ? `Examples:\n${compressedContext.relevantExamples.join("\n")}`
    : ""
}
`;

    const prompt = `You are a precise JSON generator. Generate EXACTLY ${
      schema.isArray ? "a JSON array" : "a JSON object"
    } based on the provided content.

NOTATION RULES (Enforce Simple Text):
- Math: Use plain text fractions (e.g., "1/3") NOT LaTeX (e.g., "\\frac{1}{3}")
- Chemistry: Use plain text formulas (e.g., "C6H12O6", "H2O") with NO subscripts
- General: Keep notation simple, direct, and readable as plain text.

${contextSummary}

TASK: ${schema.description}

${
  additionalParams?.count
    ? `Generate exactly ${additionalParams.count} items.`
    : ""
}

Language: ${language}

EXACT JSON FORMAT REQUIRED:
${schema.jsonTemplate}

CRITICAL RULES:
1. Output ONLY valid JSON - absolutely NO text before or after the JSON
2. Use double quotes for ALL strings
3. NO trailing commas anywhere (not after arrays, objects, or properties)
4. Use "null" NOT "None", "true" NOT "True", "false" NOT "False"
5. Ensure the JSON is complete and properly closed
6. Each string value must be properly escaped
7. DO NOT include any explanation, markdown, or code blocks
8. Array closing format: ["item1", "item2"] NOT ["item1", "item2"],
9. Object closing format: {"key": "value"} NOT {"key": "value",}

OUTPUT THE JSON NOW:`;

    const maxRetries = 3;
    let lastError: Error | null = null;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const result = await ollamaChatService.generateWithMaxOutput(
          prompt,
          3000,
        );
        const response = result.answer;

        const parsed = this.extractAndParseJSON(response, schema.isArray);
        console.log(
          `✅ Layer 2 complete (attempt ${attempt}): Generated ${
            schema.isArray ? (parsed as any[]).length + " items" : "object"
          }`,
        );
        return parsed as T;
      } catch (error) {
        console.warn(
          `⚠️ Layer 2 attempt ${attempt}/${maxRetries} failed:`,
          error,
        );
        lastError = error instanceof Error ? error : new Error("Unknown error");

        // Wait briefly before retry
        if (attempt < maxRetries) {
          await new Promise((resolve) => setTimeout(resolve, 500));
        }
      }
    }

    throw new Error(
      `Layer 2 (JSON Generation) failed after ${maxRetries} attempts: ${lastError?.message}`,
    );
  }

  /**
   * Helper: Get full document content from fileId
   */
  private async getFullDocumentContent(fileId: string): Promise<{
    fileName: string;
    fullContent: string;
    pages: number;
  }> {
    // Try to get pages
    const pages = await documentService.getAllPages(fileId);

    if (pages && pages.length > 0) {
      const fullContent = pages
        .sort((a, b) => a.pageNumber - b.pageNumber)
        .map((p) => p.content)
        .join("\n\n");

      const fileName = "Document";

      return {
        fileName,
        fullContent,
        pages: pages.length,
      };
    }

    // Fallback to legacy document
    const legacyContent = await documentService.getDocument(fileId);
    if (legacyContent) {
      return {
        fileName: "Document",
        fullContent: legacyContent,
        pages: 1,
      };
    }

    throw new Error("Document not found");
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // UPDATED GENERATION METHODS (USING CACHE/UNIVERSAL CONTEXT)
  // ═══════════════════════════════════════════════════════════════════════════

  async generateSummary(
    fileId: string,
    language: LanguageCode,
    tone: string = "professional",
  ): Promise<LMRSummary> {
    try {
      console.log(
        "📝 Starting Summary Generation (Two-Layer AI with Dynamic Quantities)...",
      );

      const document = await this.getFullDocumentContent(fileId);
      const languageName = SUPPORTED_LANGUAGES[language];
      const metrics = this.calculateDocumentMetrics(
        document.fullContent,
        document.pages,
      );
      const conceptCount = metrics.recommendedConceptCount;

      // Use Universal Context (Cached)
      const compressedContext = await this.getUniversalContext(
        fileId,
        document.fullContent,
        languageName,
      );

      console.log(
        `🔄 Layer 2: Generating structured summary with ${conceptCount} important concepts...`,
      );

      const keyTopicCount = Math.min(11, Math.max(6, metrics.estimatedTopics));

      const summarySchema = {
        description: `You are an NCERT/CBSE educational expert. Generate LAST-MINUTE REVISION content from the document.

MANDATORY COUNTS:
• keyTopics: Generate EXACTLY ${keyTopicCount} topics with UNIQUE names from the document
• importantConcepts: Generate EXACTLY ${conceptCount} concepts with UNIQUE names
• Each concept must have EXACTLY 3 bullet points

STRUCTURE REQUIREMENTS:
• introduction: Write 2-3 SENTENCES (a proper paragraph, not one line!)
• conclusion: Write 2 sentences summarizing key takeaways

CRITICAL: Replace ALL placeholder text with ACTUAL content from the document!
DO NOT use generic names like "Topic 1" or "Concept 1" - use REAL topic names!

LANGUAGE: ${languageName} | TONE: ${tone} | NO EMOJIS`,

        jsonTemplate: `{
  "introduction": "Brief 2-3 sentence introduction paragraph about the main subject.",
  "summaryPoints": [
    "Key fact or concept 1",
    "Key fact or concept 2",
    "Key fact or concept 3",
    "Key fact or concept 4",
    "Key fact or concept 5",
    "Key fact or concept 6"
  ],
  "conclusion": "Brief 1-2 sentence conclusion summarizing key takeaways.",
  "keyTopics": [
    {"name": "Topic Name 1", "description": "Brief description of topic 1"},
    {"name": "Topic Name 2", "description": "Brief description of topic 2"},
    {"name": "Topic Name 3", "description": "Brief description of topic 3"},
    {"name": "Topic Name 4", "description": "Brief description of topic 4"},
    {"name": "Topic Name 5", "description": "Brief description of topic 5"},
    {"name": "Topic Name 6", "description": "Brief description of topic 6"}
  ],
  "importantConcepts": [
    {"name": "Concept 1", "points": ["Point 1", "Point 2", "Point 3"]},
    {"name": "Concept 2", "points": ["Point 1", "Point 2", "Point 3"]},
    {"name": "Concept 3", "points": ["Point 1", "Point 2", "Point 3"]},
    {"name": "Concept 4", "points": ["Point 1", "Point 2", "Point 3"]},
    {"name": "Concept 5", "points": ["Point 1", "Point 2", "Point 3"]},
    {"name": "Concept 6", "points": ["Point 1", "Point 2", "Point 3"]},
    {"name": "Concept 7", "points": ["Point 1", "Point 2", "Point 3"]},
    {"name": "Concept 8", "points": ["Point 1", "Point 2", "Point 3"]}
  ]
}

YOU MUST generate exactly 8 importantConcepts and ${keyTopicCount} keyTopics. Do NOT stop early!`,
        isArray: false,
      };

      const result = await this.generateJSONFromContext<{
        introduction: string;
        summaryPoints: string[];
        conclusion: string;
        keyTopics: { name: string; description: string }[];
        importantConcepts: { name: string; points: string[] }[];
      }>(compressedContext, "summary", languageName, summarySchema);

      console.log("✅ Structured summary generation complete!");

      const legacySummary = `${result.introduction} \n\n${result.summaryPoints
        .map((p) => `• ${p}`)
        .join("\n")} \n\n${result.conclusion} `;

      return {
        introduction: result.introduction || "",
        summaryPoints: result.summaryPoints || [],
        conclusion: result.conclusion || "",
        summary: legacySummary,
        keyTopics: (result.keyTopics || []).map((t: any) => ({
          name: typeof t === "string" ? t : t.name || "Topic",
          description: typeof t === "string" ? "" : t.description || "",
        })),
        importantConcepts: (result.importantConcepts || []).map((c: any) => ({
          name: typeof c === "string" ? c : c.name || "Concept",
          points: typeof c === "string" ? [c] : c.points || [],
        })),
        language: languageName,
      };
    } catch (error) {
      console.error("❌ Summary generation failed:", error);
      throw new Error(
        `Failed to generate summary: ${
          error instanceof Error ? error.message : "Unknown error"
        } `,
      );
    }
  }

  async generateQuestions(
    fileId: string,
    language: LanguageCode,
    count: number = 10,
  ): Promise<LMRQuestion[]> {
    try {
      console.log(
        "❓ Starting Q&A Generation (Two-Layer AI with Dynamic Quantities)...",
      );

      const document = await this.getFullDocumentContent(fileId);
      const languageName = SUPPORTED_LANGUAGES[language];
      const metrics = this.calculateDocumentMetrics(
        document.fullContent,
        document.pages,
      );
      const dynamicCount = metrics.recommendedQuestionCount;
      const actualCount = Math.max(count, dynamicCount);

      // Use Universal Context (Cached)
      const compressedContext = await this.getUniversalContext(
        fileId,
        document.fullContent,
        languageName,
      );

      console.log(
        `🔄 Layer 2: Generating ${actualCount} high - quality questions...`,
      );

      const easyCount = Math.round(actualCount * 0.3);
      const mediumCount = Math.round(actualCount * 0.5);
      const hardCount = actualCount - easyCount - mediumCount;

      const questionsSchema = {
        description: `You are an expert exam question creator.
        GENERATE ${actualCount} QUESTIONS using "numericalData" and "facts" from context.
        DISTRIBUTION:
  • ${easyCount} EASY questions (basic recall)
  • ${mediumCount} MEDIUM questions (application)
  • ${hardCount} HARD questions (analysis)

LANGUAGE: ${languageName}`,
        jsonTemplate: `[
        {
          "question": "Clear question",
          "answer": "Answer with explanation",
          "subject": "Topic",
          "difficulty": "Easy",
          "pageReference": null
        }
      ]`,
        isArray: true,
      };

      const questions = await this.generateJSONFromContext<any[]>(
        compressedContext,
        "questions",
        languageName,
        questionsSchema,
        { count: actualCount },
      );

      console.log(
        `✅ Q & A generation complete! Generated ${questions.length} questions`,
      );

      return questions.map((q: any, index: number) => ({
        id: index + 1,
        question: q.question,
        answer: q.answer,
        subject: q.subject || "General",
        difficulty: q.difficulty || "Medium",
        pageReference: q.pageReference,
      }));
    } catch (error) {
      console.error("❌ Q&A generation failed:", error);
      throw new Error(
        `Failed to generate questions: ${
          error instanceof Error ? error.message : "Unknown error"
        } `,
      );
    }
  }

  async generateQuiz(
    fileId: string,
    language: LanguageCode,
    count: number = 10,
  ): Promise<LMRQuiz[]> {
    try {
      console.log("📋 Starting Quiz Generation (Two-Layer AI)...");

      const document = await this.getFullDocumentContent(fileId);
      const languageName = SUPPORTED_LANGUAGES[language];

      // Use Universal Context (Cached)
      const compressedContext = await this.getUniversalContext(
        fileId,
        document.fullContent,
        languageName,
      );

      console.log("🔄 Layer 2: Generating quiz JSON...");
      const quizSchema = {
        description: `Generate ${count} MCQs. Use "definitions" from context for tricky options.
CRITICAL: correctAnswer must be a NUMBER (0-3).`,
        jsonTemplate: `[
        {
          "question": "Question text",
          "options": ["Option A", "Option B", "Option C", "Option D"],
          "correctAnswer": 0,
          "explanation": "Why correct",
          "difficulty": "Easy",
          "subject": "Subject"
        }
      ]`,
        isArray: true,
      };

      const quizzes = await this.generateJSONFromContext<any[]>(
        compressedContext,
        "quiz",
        languageName,
        quizSchema,
        { count },
      );

      console.log("✅ Quiz generation complete!");

      return quizzes.map((q: any, index: number) => ({
        id: index + 1,
        question: q.question,
        options: q.options || [],
        correctAnswer: q.correctAnswer || 0,
        explanation: q.explanation,
        difficulty: q.difficulty || "Medium",
        subject: q.subject || "General",
      }));
    } catch (error) {
      console.error("❌ Quiz generation failed:", error);
      throw new Error(
        `Failed to generate quiz: ${
          error instanceof Error ? error.message : "Unknown error"
        } `,
      );
    }
  }

  async generateRecallNotes(
    fileId: string,
    language: LanguageCode,
  ): Promise<LMRRecallNote[]> {
    try {
      console.log("🧠 Starting Recall Notes Generation...");

      const document = await this.getFullDocumentContent(fileId);
      const languageName = SUPPORTED_LANGUAGES[language];
      const metrics = this.calculateDocumentMetrics(
        document.fullContent,
        document.pages,
      );
      const topicCount = metrics.recommendedRecallTopicCount;

      // Use Universal Context (Cached)
      const compressedContext = await this.getUniversalContext(
        fileId,
        document.fullContent,
        languageName,
      );

      console.log(`🔄 Layer 2: Generating ${topicCount} recall note topics...`);
      const recallNotesSchema = {
        description: `Generate ${topicCount} revision topics.
        Use "keyFacts" for quickFacts and "importantConcepts" for keyPoints.
        Arrays must contain STRINGS only.`,
        jsonTemplate: `[
  {
    "topic": "Main Topic Name",
    "keyPoints": ["Point 1", "Point 2"],
    "quickFacts": ["Fact 1", "Fact 2"],
    "mnemonics": ["Mnemonic"]
  }
]`,
        isArray: true,
      };

      const notes = await this.generateJSONFromContext<any[]>(
        compressedContext,
        "recallNotes",
        languageName,
        recallNotesSchema,
      );

      console.log("✅ Comprehensive recall notes generation complete!");

      const normalizeToStringArray = (items: any[]): string[] => {
        if (!Array.isArray(items)) return [];
        return items
          .map((item) => {
            if (typeof item === "string") return item;
            if (typeof item === "object" && item !== null) {
              return (
                item.text ||
                item.content ||
                item.point ||
                item.fact ||
                item.value ||
                item.description ||
                JSON.stringify(item)
              );
            }
            return String(item);
          })
          .filter(
            (item) =>
              item && item !== "{}" && item !== "null" && item !== "undefined",
          );
      };

      return notes.map((n: any) => ({
        topic: typeof n.topic === "string" ? n.topic : n.topic?.name || "Topic",
        keyPoints: normalizeToStringArray(n.keyPoints || []),
        quickFacts: normalizeToStringArray(n.quickFacts || []),
        mnemonics: normalizeToStringArray(n.mnemonics || []),
      }));
    } catch (error) {
      console.error("❌ Recall notes generation failed:", error);
      throw new Error(
        `Failed to generate recall notes: ${
          error instanceof Error ? error.message : "Unknown error"
        } `,
      );
    }
  }

  async getAllContent(fileId: string, language: LanguageCode) {
    try {
      // Pre-warm the cache
      const doc = await this.getFullDocumentContent(fileId);
      await this.getUniversalContext(
        fileId,
        doc.fullContent,
        SUPPORTED_LANGUAGES[language],
      );

      const [summary, questions, quiz, recallNotes] = await Promise.all([
        this.generateSummary(fileId, language),
        this.generateQuestions(fileId, language, 10),
        this.generateQuiz(fileId, language, 10),
        this.generateRecallNotes(fileId, language),
      ]);

      return {
        summary,
        questions,
        quiz,
        recallNotes,
      };
    } catch (error) {
      throw new Error(
        `Failed to generate content: ${
          error instanceof Error ? error.message : "Unknown error"
        } `,
      );
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // NLLB TRANSLATION METHODS
  // ═══════════════════════════════════════════════════════════════════════════

  private async translateText(
    text: string,
    targetLang: SupportedLanguageCode,
  ): Promise<string> {
    if (!text || text.trim().length === 0) return text;
    if (targetLang === "en") return text;

    try {
      const nllbTargetCode = languageService.toNLLBCode(targetLang);
      const translated = await nllbService.translate(text, {
        srcLang: "eng_Latn",
        tgtLang: nllbTargetCode,
        useCache: false,
      });

      console.log(
        `📝 Translated: "${text.substring(
          0,
          40,
        )}..." -> "${translated.substring(0, 40)}..."`,
      );

      return translated;
    } catch (error) {
      console.warn(
        `⚠️ Translation failed for text "${text.substring(0, 30)}...":`,
        error,
      );
      return text;
    }
  }

  async translateSummary(
    summary: LMRSummary,
    targetLang: SupportedLanguageCode,
  ): Promise<LMRSummary> {
    if (targetLang === "en") return summary;

    console.log(
      `🌐 Translating summary to ${languageService.getLanguageName(
        targetLang,
      )}...`,
    );

    const [
      translatedIntro,
      translatedConclusion,
      translatedSummary,
      translatedPoints,
      translatedTopics,
      translatedConcepts,
    ] = await Promise.all([
      this.translateText(summary.introduction, targetLang),
      this.translateText(summary.conclusion, targetLang),
      summary.summary
        ? this.translateText(summary.summary, targetLang)
        : Promise.resolve(undefined),
      Promise.all(
        summary.summaryPoints.map((p) => this.translateText(p, targetLang)),
      ),
      Promise.all(
        summary.keyTopics.map(async (t) => ({
          name: await this.translateText(t.name, targetLang),
          description: await this.translateText(t.description, targetLang),
        })),
      ),
      Promise.all(
        summary.importantConcepts.map(async (c) => ({
          name: await this.translateText(c.name, targetLang),
          points: await Promise.all(
            c.points.map((p) => this.translateText(p, targetLang)),
          ),
        })),
      ),
    ]);

    return {
      ...summary,
      introduction: translatedIntro,
      conclusion: translatedConclusion,
      summary: translatedSummary,
      summaryPoints: translatedPoints,
      keyTopics: translatedTopics,
      importantConcepts: translatedConcepts,
      language: languageService.getLanguageName(targetLang),
    };
  }

  async translateQuestions(
    questions: LMRQuestion[],
    targetLang: SupportedLanguageCode,
  ): Promise<LMRQuestion[]> {
    if (targetLang === "en") return questions;

    console.log(
      `🌐 Translating ${
        questions.length
      } questions to ${languageService.getLanguageName(targetLang)}...`,
    );

    return Promise.all(
      questions.map(async (q) => ({
        ...q,
        question: await this.translateText(q.question, targetLang),
        answer: await this.translateText(q.answer, targetLang),
        subject: await this.translateText(q.subject, targetLang),
      })),
    );
  }

  async translateQuiz(
    quiz: LMRQuiz[],
    targetLang: SupportedLanguageCode,
  ): Promise<LMRQuiz[]> {
    if (targetLang === "en") return quiz;

    console.log(
      `🌐 Translating ${
        quiz.length
      } quiz questions to ${languageService.getLanguageName(targetLang)}...`,
    );

    return Promise.all(
      quiz.map(async (q) => ({
        ...q,
        question: await this.translateText(q.question, targetLang),
        options: await Promise.all(
          q.options.map((o) => this.translateText(o, targetLang)),
        ),
        explanation: await this.translateText(q.explanation, targetLang),
        subject: await this.translateText(q.subject, targetLang),
      })),
    );
  }

  async translateRecallNotes(
    notes: LMRRecallNote[],
    targetLang: SupportedLanguageCode,
  ): Promise<LMRRecallNote[]> {
    if (targetLang === "en") return notes;

    console.log(
      `🌐 Translating ${
        notes.length
      } recall note topics to ${languageService.getLanguageName(targetLang)}...`,
    );

    return Promise.all(
      notes.map(async (n) => ({
        topic: await this.translateText(n.topic, targetLang),
        keyPoints: await Promise.all(
          n.keyPoints.map((p) => this.translateText(p, targetLang)),
        ),
        quickFacts: await Promise.all(
          n.quickFacts.map((f) => this.translateText(f, targetLang)),
        ),
        mnemonics: n.mnemonics
          ? await Promise.all(
              n.mnemonics.map((m) => this.translateText(m, targetLang)),
            )
          : undefined,
      })),
    );
  }

  async translateContent(
    content: {
      summary?: LMRSummary;
      questions?: LMRQuestion[];
      quiz?: LMRQuiz[];
      recallNotes?: LMRRecallNote[];
    },
    targetLang: SupportedLanguageCode,
  ): Promise<{
    summary?: LMRSummary;
    questions?: LMRQuestion[];
    quiz?: LMRQuiz[];
    recallNotes?: LMRRecallNote[];
  }> {
    if (!env.NLLB_ENABLED) {
      throw new Error(
        "NLLB translation is not enabled. Set NLLB_ENABLED=true in environment.",
      );
    }

    if (targetLang === "en") {
      return content;
    }

    console.log(
      `🌐 Translating all LMR content to ${languageService.getLanguageName(
        targetLang,
      )}...`,
    );

    const [summary, questions, quiz, recallNotes] = await Promise.all([
      content.summary
        ? this.translateSummary(content.summary, targetLang)
        : Promise.resolve(undefined),
      content.questions
        ? this.translateQuestions(content.questions, targetLang)
        : Promise.resolve(undefined),
      content.quiz
        ? this.translateQuiz(content.quiz, targetLang)
        : Promise.resolve(undefined),
      content.recallNotes
        ? this.translateRecallNotes(content.recallNotes, targetLang)
        : Promise.resolve(undefined),
    ]);

    console.log(
      `✅ Translation complete to ${languageService.getLanguageName(
        targetLang,
      )}`,
    );

    return { summary, questions, quiz, recallNotes };
  }
}

export const lmrService = new LMRService();
