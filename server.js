import express from 'express';
import multer from 'multer';
import { GoogleGenerativeAI } from '@google/generative-ai';
import * as xlsx from 'xlsx';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json());
app.use(express.static(__dirname));

const upload = multer({ storage: multer.memoryStorage() });

let uploaded_context = "";

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.post('/api/upload', upload.array('files'), (req, res) => {
    // Clear context on new upload to prevent hitting the 250k token limit on Free Tier
    uploaded_context = "";

    if (!req.files || req.files.length === 0) {
        return res.status(400).json({ error: "No file part" });
    }

    let extracted_text = "";

    req.files.forEach(file => {
        const filename = file.originalname;
        const ext = filename.split('.').pop().toLowerCase();

        try {
            if (['txt', 'md', 'csv'].includes(ext)) {
                const content = file.buffer.toString('utf-8');
                extracted_text += `\n--- Content of ${filename} ---\n${content}\n`;
            } else if (['xlsx', 'xls'].includes(ext)) {
                // Parse Excel with xlsx
                const workbook = xlsx.read(file.buffer, { type: 'buffer' });
                const firstSheetName = workbook.SheetNames[0];
                const worksheet = workbook.Sheets[firstSheetName];
                const content = xlsx.utils.sheet_to_csv(worksheet);
                // Cap extraction at 80,000 characters (roughly 20,000 tokens) 
                // This fits perfectly within Free Tier limits.
                extracted_text += `\n--- Content of ${filename} ---\n${content.substring(0, 80000)}\n`;
            } else {
                extracted_text += `\n[Unsupported file format: ${filename}. Please use txt, md, csv, or xlsx]\n`;
            }
        } catch (e) {
            console.error(`Error reading ${filename}: ${e.message}`);
        }
    });

    if (extracted_text) {
        uploaded_context += extracted_text;
        console.log(`Successfully parsed file. New database context length: ${uploaded_context.length} characters.`);
        res.json({ message: "Successfully processed." });
    } else {
        res.status(400).json({ error: "No valid data extracted" });
    }
});

app.post('/api/chat', async (req, res) => {
    const { api_key, query, history, temperature = 0.7 } = req.body;
    console.log(`Received chat request. History messages: ${history?.length || 0}. Current context length: ${uploaded_context.length}`);

    if (!api_key) {
        return res.status(400).json({ error: "No API key provided." });
    }

    try {
        const genAI = new GoogleGenerativeAI(api_key);
        
        // Using the 2.5 model confirmed on your account previously. 
        // With the 'memory clearing' update, it will no longer hit the quota limit!
        const modelName = "gemini-2.5-flash"; 
        
        const model = genAI.getGenerativeModel({ 
            model: modelName, 
            systemInstruction: "You are a strict data analysis chatbot. Answer the user's question USING ONLY the provided Document Context below. If the answer is not in the context, do not use outside knowledge. Simply reply: 'I cannot find the answer in the provided documents.'"
        });

        // Map our simple history format to the Gemini SDK format
        // Role "assistant" becomes "model" in the SDK
        const historyParts = (history || []).map(m => ({
            role: m.role === "assistant" ? "model" : "user",
            parts: [{ text: m.text }]
        }));

        // Always put the context in the current prompt for the most accurate document grounding
        const finalPrompt = `Document Context:\n${uploaded_context}\n\nUser Question: ${query}`;
        
        // Add the current prompt to the set of contents
        const contents = [
            ...historyParts,
            { role: "user", parts: [{ text: finalPrompt }] }
        ];

        const result = await model.generateContent({
            contents: contents,
            generationConfig: {
                temperature: parseFloat(temperature),
            }
        });
        
        const response = await result.response;
        res.json({ response: response.text() });

    } catch (e) {
        console.error(e);
        res.status(500).json({ error: e.message });
    }
});

app.listen(5000, () => {
    console.log("Starting Indigo Logic Backend Node Server on http://127.0.0.1:5000");
});
