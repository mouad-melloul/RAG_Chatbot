document.getElementById("send-btn").addEventListener("click", async () => {
    const userInput = document.getElementById("user-input").value;
    if (!userInput) return;

    const chatBox = document.getElementById("chat-box");
    chatBox.innerHTML += `<div><strong>You:</strong> ${userInput}</div>`;

    const response = await fetch("/ask", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: userInput }),
    });

    const data = await response.json();
    if (data.answer) {
        chatBox.innerHTML += `<div><strong>Bot:</strong> ${data.answer}</div>`;
    } else {
        chatBox.innerHTML += `<div><strong>Error:</strong> ${data.error}</div>`;
    }

    document.getElementById("user-input").value = "";
    chatBox.scrollTop = chatBox.scrollHeight;
});