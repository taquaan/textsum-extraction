document.getElementById("summarizeBtn").addEventListener("click", function () {
  const originalText = document.getElementById("originalText").value;

  // AJAX request to server
  fetch("/summarize", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text: originalText }),
  })
    .then((response) => response.json())
    .then((data) => {
      const summarizedText = data.summarized_text;
      document.getElementById("summarizedText").value = summarizedText;
    })
    .catch((err) => console.log("Error fetching summarized text: ", err));
});

document.getElementById("restartBtn").addEventListener("click", () => {
  document.getElementById("originalText").value = "";
  document.getElementById("summarizedText").value = "";
});
