const API_URL = "http://127.0.0.1:8000";


// ============================================================
// DOM ELEMENTS
// ============================================================

const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");

const dropzone = document.getElementById("dropzone");
const fileNameText = document.getElementById("file-name-text");

const filesList = document.getElementById("filesList");
const fileCountBadge = document.getElementById("fileCountBadge");

const statusTitle = document.getElementById("statusTitle");
const statusSub = document.getElementById("statusSub");
const statusCard = document.getElementById("statusCard");

const messageBox = document.getElementById("messageFormeight");

const questionInput = document.getElementById("question");
const sendButton = document.getElementById("send");
const clearButton = document.getElementById("clearBtn");


// ============================================================
// APPLICATION STATE
// ============================================================

let loadedFiles = [];


// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener("DOMContentLoaded", () => {

    checkBackend();

    setupFileInput();

    setupDragAndDrop();

    setupUpload();

    setupChat();

    setupClearChat();

});


// ============================================================
// CHECK BACKEND
// ============================================================

async function checkBackend() {

    try {

        const response = await fetch(
            `${API_URL}/api/health`
        );

        if (!response.ok) {
            throw new Error("Backend unavailable");
        }

        console.log("FastAPI backend is running.");

    } catch (error) {

        console.error(
            "Backend connection failed:",
            error
        );

        updateStatus(
            "Backend unavailable",
            "Make sure FastAPI is running on port 8000."
        );
    }
}


// ============================================================
// FILE INPUT
// ============================================================

function setupFileInput() {

    if (!fileInput) {
        return;
    }

    fileInput.addEventListener(
        "change",
        () => {

            const files =
                Array.from(fileInput.files);

            updateSelectedFilesDisplay(files);
        }
    );
}


// ============================================================
// DISPLAY SELECTED FILES
// ============================================================

function updateSelectedFilesDisplay(files) {

    if (!fileNameText) {
        return;
    }

    if (files.length === 0) {

        fileNameText.textContent =
            "No file selected";

        return;
    }

    if (files.length === 1) {

        fileNameText.textContent =
            files[0].name;

        return;
    }

    fileNameText.textContent =
        `${files.length} PDF files selected`;
}


// ============================================================
// DRAG & DROP
// ============================================================

function setupDragAndDrop() {

    if (!dropzone) {
        return;
    }


    dropzone.addEventListener(
        "dragover",
        (event) => {

            event.preventDefault();

            dropzone.classList.add(
                "drag-over"
            );
        }
    );


    dropzone.addEventListener(
        "dragleave",
        () => {

            dropzone.classList.remove(
                "drag-over"
            );
        }
    );


    dropzone.addEventListener(
        "drop",
        (event) => {

            event.preventDefault();

            dropzone.classList.remove(
                "drag-over"
            );

            const files =
                Array.from(
                    event.dataTransfer.files
                );


            const pdfFiles =
                files.filter(
                    file =>
                        file.name
                            .toLowerCase()
                            .endsWith(".pdf")
                );


            if (pdfFiles.length === 0) {

                alert(
                    "Please select PDF files only."
                );

                return;
            }


            /*
             * DataTransfer allows us to put the dropped
             * files into the existing file input.
             */

            const dataTransfer =
                new DataTransfer();


            pdfFiles.forEach(
                file =>
                    dataTransfer.items.add(file)
            );


            fileInput.files =
                dataTransfer.files;


            updateSelectedFilesDisplay(
                pdfFiles
            );
        }
    );
}


// ============================================================
// UPLOAD
// ============================================================

function setupUpload() {

    if (!uploadForm) {
        return;
    }


    uploadForm.addEventListener(
        "submit",
        async (event) => {

            event.preventDefault();


            const files =
                Array.from(
                    fileInput.files
                );


            if (files.length === 0) {

                alert(
                    "Please select at least one PDF."
                );

                return;
            }


            // Check that every file is a PDF

            const invalidFile =
                files.find(
                    file =>
                        !file.name
                            .toLowerCase()
                            .endsWith(".pdf")
                );


            if (invalidFile) {

                alert(
                    "Only PDF files are allowed."
                );

                return;
            }


            // ------------------------------------------------
            // UI: uploading
            // ------------------------------------------------

            setUploadLoading(true);


            updateStatus(
                "Processing document...",
                "Extracting text and creating embeddings."
            );


            // ------------------------------------------------
            // FormData
            // ------------------------------------------------

            const formData =
                new FormData();


            files.forEach(
                file => {

                    formData.append(
                        "pdf_file",
                        file
                    );
                }
            );


            try {

                const response =
                    await fetch(
                        `${API_URL}/api/upload`,
                        {
                            method: "POST",
                            body: formData
                        }
                    );


                const data =
                    await response.json();


                console.log(
                    "Upload response:",
                    data
                );


                if (!response.ok) {

                    throw new Error(
                        data.detail ||
                        "Upload failed."
                    );
                }


                // ------------------------------------------------
                // Process backend results
                // ------------------------------------------------

                let successfulFiles = 0;


                if (data.files) {

                    data.files.forEach(
                        result => {

                            if (
                                result.status === "ok"
                            ) {

                                successfulFiles++;


                                addLoadedFile(
                                    result.file,
                                    result.chunks
                                );

                            } else {

                                console.error(
                                    `Error processing ${result.file}:`,
                                    result.message
                                );
                            }
                        }
                    );
                }


                // ------------------------------------------------
                // Successful upload
                // ------------------------------------------------

                if (successfulFiles > 0) {

                    removeEmptyState();


                    updateStatus(
                        "Document ready",
                        `${successfulFiles} PDF(s) loaded successfully.`
                    );
                    
                    statusCard.classList.add("active");

                    uploadBtn.textContent =
                        "✓ Uploaded";


                } else {

                    updateStatus(
                        "Upload failed",
                        "No valid documents were loaded."
                    );

                    statusCard.classList.remove("active");
                    
                    uploadBtn.textContent =
                        "↑  Upload & Analyse";
                }


            } catch (error) {

                console.error(
                    "Upload error:",
                    error
                );


                updateStatus(
                    "Upload error",
                    error.message
                );


                alert(
                    `Upload error:\n${error.message}`
                );


                uploadBtn.textContent =
                    "↑  Upload & Analyse";


            } finally {

                uploadBtn.disabled =
                    false;
            }
        }
    );
}


// ============================================================
// UPLOAD BUTTON LOADING STATE
// ============================================================

function setUploadLoading(isLoading) {

    if (!uploadBtn) {
        return;
    }


    uploadBtn.disabled =
        isLoading;


    if (isLoading) {

        uploadBtn.textContent =
            "Uploading...";

    } else {

        uploadBtn.textContent =
            "↑  Upload & Analyse";
    }
}


// ============================================================
// ADD LOADED FILE
// ============================================================

function addLoadedFile(
    filename,
    chunks = null
) {

    // Prevent duplicate display

    const exists =
        loadedFiles.some(
            file =>
                file.name === filename
        );


    if (exists) {
        return;
    }


    loadedFiles.push({
        name: filename,
        chunks: chunks
    });


    renderLoadedFiles();
}


// ============================================================
// RENDER LOADED FILES
// ============================================================

function renderLoadedFiles() {

    if (!filesList) {
        return;
    }


    filesList.innerHTML = "";


    if (loadedFiles.length === 0) {

        filesList.innerHTML = `
            <div
                class="no-files-hint"
                id="noFilesHint"
            >
                No documents yet
            </div>
        `;


        if (fileCountBadge) {
            fileCountBadge.textContent = "0";
        }


        return;
    }


    if (fileCountBadge) {

        fileCountBadge.textContent =
            loadedFiles.length;
    }


    loadedFiles.forEach(
        file => {

            const fileElement =
                document.createElement("div");


            fileElement.className =
                "loaded-file";


            fileElement.innerHTML = `

                <div class="file-dot"></div>

                <div class="loaded-file-info">

                    <div class="loaded-file-name">
                        ${escapeHtml(file.name)}
                    </div>

                    ${
                        file.chunks !== null
                        ?
                        `
                        <div class="loaded-file-meta">
                            ${file.chunks} chunks
                        </div>
                        `
                        :
                        ""
                    }

                </div>
            `;


            filesList.appendChild(
                fileElement
            );
        }
    );
}


// ============================================================
// CHAT SETUP
// ============================================================

function setupChat() {

    if (!sendButton || !questionInput) {
        return;
    }


    // Send button

    sendButton.addEventListener(
        "click",
        askQuestion
    );


    // Enter = send
    // Shift + Enter = new line

    questionInput.addEventListener(
        "keydown",
        (event) => {

            if (
                event.key === "Enter" &&
                !event.shiftKey
            ) {

                event.preventDefault();

                askQuestion();
            }
        }
    );


    // Auto resize textarea

    questionInput.addEventListener(
        "input",
        () => {

            questionInput.style.height =
                "auto";


            questionInput.style.height =
                `${questionInput.scrollHeight}px`;
        }
    );
}


// ============================================================
// ASK QUESTION
// ============================================================

async function askQuestion() {

    const question =
        questionInput.value.trim();


    if (!question) {
        return;
    }


    // --------------------------------------------------------
    // Display user's message
    // --------------------------------------------------------

    addUserMessage(
        question
    );


    // --------------------------------------------------------
    // Clear input
    // --------------------------------------------------------

    questionInput.value = "";

    questionInput.style.height =
        "auto";


    // --------------------------------------------------------
    // Disable send button
    // --------------------------------------------------------

    sendButton.disabled =
        true;


    // --------------------------------------------------------
    // Show thinking animation
    // --------------------------------------------------------

    addThinkingMessage();


    try {

        const response =
            await fetch(
                `${API_URL}/api/ask`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        question: question
                    })
                }
            );


        const data =
            await response.json();


        // ----------------------------------------------------
        // Remove thinking animation
        // ----------------------------------------------------

        removeThinkingMessage();


        // ----------------------------------------------------
        // Backend error
        // ----------------------------------------------------

        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Failed to get an answer."
            );
        }


        // ----------------------------------------------------
        // Display bot answer
        // ----------------------------------------------------

        addBotMessage(
            data.answer
        );


    } catch (error) {

        console.error(
            "Question error:",
            error
        );


        removeThinkingMessage();


        addBotMessage(
            `Error: ${error.message}`
        );


    } finally {

        sendButton.disabled =
            false;


        questionInput.focus();
    }
}


// ============================================================
// USER MESSAGE
// ============================================================

function addUserMessage(text) {

    removeEmptyState();


    const safeText =
        escapeHtml(text);


    messageBox.insertAdjacentHTML(
        "beforeend",
        `

        <div class="msg-row user">

            <div class="bubble-wrap">

                <div class="bubble usr">
                    ${safeText}
                </div>

                <div class="msg-meta">
                    ${getCurrentTime()}
                </div>

            </div>

            <div class="msg-avatar usr">

                <img
                    src="images/user.png"
                    alt="You"
                >

            </div>

        </div>

        `
    );


    scrollMessages();
}


// ============================================================
// BOT MESSAGE
// ============================================================

function addBotMessage(text) {

    removeEmptyState();


    /*
     * Escape HTML for security.
     *
     * Then preserve line breaks from the LLM response.
     */

    const safeText =
        escapeHtml(text)
            .replace(/\n/g, "<br>");


    messageBox.insertAdjacentHTML(
        "beforeend",
        `

        <div class="msg-row bot">

            <div class="msg-avatar bot">

                <img
                    src="images/Baymax.jpeg"
                    alt="DocBot"
                >

            </div>

            <div class="bubble-wrap">

                <div class="bubble bot">
                    ${safeText}
                </div>

                <div class="msg-meta">
                    ${getCurrentTime()}
                </div>

            </div>

        </div>

        `
    );


    scrollMessages();
}


// ============================================================
// THINKING MESSAGE
// ============================================================

function addThinkingMessage() {

    removeEmptyState();


    messageBox.insertAdjacentHTML(
        "beforeend",
        `

        <div
            class="msg-row bot"
            id="thinkingRow"
        >

            <div class="msg-avatar bot">

                <img
                    src="images/Baymax.jpeg"
                    alt="DocBot"
                >

            </div>

            <div class="bubble-wrap">

                <div class="bubble bot">

                    <div class="thinking-dots">

                        <span></span>
                        <span></span>
                        <span></span>

                    </div>

                </div>

            </div>

        </div>

        `
    );


    scrollMessages();
}


// ============================================================
// REMOVE THINKING MESSAGE
// ============================================================

function removeThinkingMessage() {

    const thinkingRow =
        document.getElementById(
            "thinkingRow"
        );


    if (thinkingRow) {

        thinkingRow.remove();
    }
}


// ============================================================
// CLEAR CHAT
// ============================================================

function setupClearChat() {

    if (!clearButton) {
        return;
    }


    clearButton.addEventListener(
        "click",
        async () => {

            try {

                const response =
                    await fetch(
                        `${API_URL}/api/clear`,
                        {
                            method: "POST"
                        }
                    );


                const data =
                    await response.json();


                if (!response.ok) {

                    throw new Error(
                        data.detail ||
                        "Failed to clear chat."
                    );
                }


                // ------------------------------------------------
                // Clear messages
                // ------------------------------------------------

                messageBox.innerHTML = `

                    <div
                        class="empty-state"
                        id="emptyState"
                    >

                        <img
                            src="images/Baymax.jpeg"
                            alt="DocBot"
                            style="
                                width:64px;
                                height:64px;
                                object-fit:contain;
                                border-radius:14px;
                                margin:0 auto 16px;
                                display:block;
                                opacity:0.25;
                            "
                        >

                        <h3>
                            No document loaded
                        </h3>

                        <p>
                            Upload a PDF using the sidebar
                            to start interrogating its content.
                        </p>

                    </div>

                `;


                // ------------------------------------------------
                // Clear loaded documents
                // ------------------------------------------------

                loadedFiles = [];

                renderLoadedFiles();


                // ------------------------------------------------
                // Update status
                // ------------------------------------------------

                updateStatus(
                    "Waiting for document",
                    "Upload a PDF to begin"
                );


                // Reset upload button

                if (uploadBtn) {

                    uploadBtn.disabled =
                        false;

                    uploadBtn.textContent =
                        "↑  Upload & Analyse";
                }


                // Reset file input

                if (fileInput) {
                    fileInput.value = "";
                }


                if (fileNameText) {

                    fileNameText.textContent =
                        "No file selected";
                }


                console.log(
                    data.message
                );


            } catch (error) {

                console.error(
                    "Clear error:",
                    error
                );


                alert(
                    `Could not clear: ${error.message}`
                );
            }
        }
    );
}


// ============================================================
// STATUS
// ============================================================

function updateStatus(
    title,
    subtitle
) {

    if (statusTitle) {

        statusTitle.textContent =
            title;
    }


    if (statusSub) {

        statusSub.textContent =
            subtitle;
    }
}


// ============================================================
// REMOVE EMPTY STATE
// ============================================================

function removeEmptyState() {

    const emptyState =
        document.getElementById(
            "emptyState"
        );


    if (emptyState) {

        emptyState.remove();
    }
}


// ============================================================
// SCROLL CHAT
// ============================================================

function scrollMessages() {

    if (!messageBox) {
        return;
    }


    messageBox.scrollTop =
        messageBox.scrollHeight;
}


// ============================================================
// CURRENT TIME
// ============================================================

function getCurrentTime() {

    const now =
        new Date();


    return (
        now.getHours()
            .toString()
            .padStart(2, "0")
        +
        ":"
        +
        now.getMinutes()
            .toString()
            .padStart(2, "0")
    );
}


// ============================================================
// ESCAPE HTML
// ============================================================

function escapeHtml(text) {

    const div =
        document.createElement("div");


    div.textContent =
        text;


    return div.innerHTML;
}