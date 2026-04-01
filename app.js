// Indigo Logic App Interaction Scripts

document.addEventListener('DOMContentLoaded', () => {
    // 1. DOM Elements and Variables
    const tempSlider = document.getElementById('tempSlider');
    const tempValue = document.getElementById('tempValue');
    const apiKeyInput = document.getElementById('apiKeyInput');
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const fileList = document.getElementById('fileList');
    const processBtn = document.getElementById('processBtn');
    const processIcon = document.getElementById('processIcon');
    const processText = document.getElementById('processText');
    const typingIndicator = document.getElementById('typingIndicator');
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const chatContainer = document.getElementById('chatContainer');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');

    let selectedFiles = []; 
    let chatHistory = []; 

    // 2. Initial Setup & Event Listeners
    if(tempSlider && tempValue) {
        tempSlider.addEventListener('input', (e) => {
            tempValue.textContent = parseFloat(e.target.value).toFixed(1);
        });
    }

    const renderFiles = () => {
        fileList.innerHTML = '';
        if (selectedFiles.length === 0) {
            fileList.innerHTML = `<p class="text-on-surface-variant text-xs p-2 italic text-center">No files uploaded yet.</p>`;
            return;
        }
        selectedFiles.forEach((file, index) => {
            const name = file.name;
            const ext = name.split('.').pop().toLowerCase();
            const icon = ext === 'pdf' ? 'picture_as_pdf' : (ext === 'md' || ext === 'txt') ? 'article' : 'description';
            
            const fileItem = document.createElement('div');
            fileItem.className = 'flex items-center justify-between bg-surface-container-high p-2 rounded text-[11px] group border border-outline-variant/10 animate-[fadeIn_0.3s_ease-out_forwards]';
            fileItem.innerHTML = `
                <div class="flex items-center gap-2">
                    <span class="material-symbols-outlined text-[14px] text-primary">${icon}</span>
                    <span class="truncate w-32 font-body text-on-surface" title="${name}">${name}</span>
                </div>
                <span class="material-symbols-outlined text-[14px] text-error opacity-0 group-hover:opacity-100 cursor-pointer hover:scale-110 transition-all" onclick="removeFile(${index})">close</span>
            `;
            fileList.appendChild(fileItem);
        });
    };

    window.removeFile = (index) => {
        selectedFiles.splice(index, 1);
        renderFiles();
    };

    if(fileInput && dropzone) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                Array.from(e.target.files).forEach(f => {
                    selectedFiles.push(f);
                });
                renderFiles();
                fileInput.value = ''; // Reset
            }
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.classList.add('border-primary', 'bg-primary/10');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropzone.classList.remove('border-primary', 'bg-primary/10');
            }, false);
        });
    }

    if(processBtn) {
        processBtn.addEventListener('click', async () => {
            if (selectedFiles.length === 0) return;
            processText.textContent = "Processing...";
            processIcon.classList.add('animate-spin');
            processIcon.style.fontVariationSettings = "'FILL' 0";

            const formData = new FormData();
            selectedFiles.forEach(file => {
                formData.append('files', file);
            });

            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                if (response.ok) {
                    processText.textContent = "Database Updated";
                    processIcon.classList.remove('animate-spin');
                    processIcon.textContent = "check_circle";
                    processBtn.classList.remove('bg-primary-container');
                    processBtn.classList.add('bg-emerald-600/80', 'text-white');
                    setTimeout(() => {
                        processText.textContent = "Process Data";
                        processIcon.textContent = "sync";
                        processIcon.style.fontVariationSettings = "'FILL' 1";
                        processBtn.classList.add('bg-primary-container');
                        processBtn.classList.remove('bg-emerald-600/80', 'text-white');
                    }, 2000);
                } else {
                    throw new Error("Upload Failed");
                }
            } catch (e) {
                processText.textContent = "Error Uploading";
                processIcon.classList.remove('animate-spin');
                processIcon.textContent = "error";
                setTimeout(() => {
                    processText.textContent = "Process Data";
                    processIcon.textContent = "sync";
                }, 2000);
            }
        });
    }

    chatInput.addEventListener('input', function() {
        this.style.height = '48px';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.value.trim() === '') this.style.height = '48px';
    });

    const createMessage = (text, isUser = false) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message-appear flex gap-4 max-w-3xl ${isUser ? 'ml-auto flex-row-reverse' : ''}`;
        let iconHtml = isUser 
            ? `<div class="w-8 h-8 rounded-full bg-surface-container-highest flex-shrink-0 flex items-center justify-center border border-outline-variant/30"><span class="material-symbols-outlined text-on-surface-variant text-sm">person</span></div>`
            : `<div class="w-8 h-8 rounded bg-primary-container flex-shrink-0 flex items-center justify-center shadow-lg shadow-primary-container/20"><span class="material-symbols-outlined text-on-primary-container text-sm" style="font-variation-settings: 'FILL' 1;">smart_toy</span></div>`;
        let bubbleHtml = isUser
            ? `<div class="bg-primary-container/20 p-5 rounded-xl rounded-tr-none border border-primary/20 shadow-md font-body backdrop-blur-sm"><p class="text-sm text-on-surface whitespace-pre-wrap">${text}</p></div>`
            : `<div class="space-y-3 flex-1 font-body"><div class="bg-surface-container p-6 rounded-xl rounded-tl-none border border-outline-variant/10 shadow-xl space-y-4"><p class="text-sm leading-relaxed text-on-surface whitespace-pre-wrap">${text}</p><div class="flex items-center gap-2 pt-3 border-t border-outline-variant/10"><span class="text-[10px] text-on-surface-variant uppercase tracking-widest font-bold">Source:</span><span class="px-2 py-1 bg-surface-container-high rounded text-[11px] text-primary border border-primary/20 flex items-center gap-1"><span class="material-symbols-outlined text-[12px]">dataset</span> Conversation Memory</span></div></div></div>`;
        msgDiv.innerHTML = iconHtml + bubbleHtml;
        return msgDiv;
    };

    const scrollToBottom = () => { chatContainer.scrollTop = chatContainer.scrollHeight; };

    const handleSend = async () => {
        const text = chatInput.value.trim();
        if (!text) return;
        const apiKey = apiKeyInput ? apiKeyInput.value.trim() : '';
        const userMsg = createMessage(text, true);
        chatContainer.insertBefore(userMsg, typingIndicator);
        chatInput.value = '';
        chatInput.style.height = '48px';
        scrollToBottom();

        if (!apiKey) {
            const aiMsg = createMessage("⚠️ Please enter your Gemini API Key in the left sidebar configuration first.", false);
            chatContainer.insertBefore(aiMsg, typingIndicator);
            scrollToBottom();
            return;
        }

        typingIndicator.classList.remove('hidden');
        scrollToBottom();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ api_key: apiKey, query: text, history: chatHistory, temperature: tempSlider ? parseFloat(tempSlider.value) : 0.7 })
            });
            const data = await response.json();
            typingIndicator.classList.add('hidden');
            if (response.ok) {
                const aiMsg = createMessage(data.response, false);
                chatContainer.insertBefore(aiMsg, typingIndicator);
                chatHistory.push({ role: "user", text: text });
                chatHistory.push({ role: "assistant", text: data.response });
                if (chatHistory.length > 20) chatHistory = chatHistory.slice(-20);
            } else {
                const aiMsg = createMessage("⚠️ Error: " + (data.error || "Unknown error"), false);
                chatContainer.insertBefore(aiMsg, typingIndicator);
            }
            scrollToBottom();
        } catch(e) {
            typingIndicator.classList.add('hidden');
            const aiMsg = createMessage("⚠️ Connection error to backend.", false);
            chatContainer.insertBefore(aiMsg, typingIndicator);
            scrollToBottom();
        }
    };

    sendBtn.addEventListener('click', handleSend);
    chatInput.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } });

    if(clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', () => {
            const messages = chatContainer.querySelectorAll('.message-appear');
            messages.forEach((msg, idx) => { if (idx > 0) msg.remove(); });
            chatHistory = [];
            chatContainer.appendChild(typingIndicator);
        });
    }

    renderFiles();
});
