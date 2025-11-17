async function analyzeNews() {
    const text = document.getElementById("newsInput").value;

    if (!text.trim()) {
        alert("Please paste a news article!");
        return;
    }

    const res = await fetch("https://fake-news-api.onrender.com/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ text })
    });

    const data = await res.json();

    document.getElementById("nb").innerText = data["Naive Bayes"];
    document.getElementById("dt").innerText = data["Decision Tree"];
    document.getElementById("rf").innerText = data["Random Forest"];
    document.getElementById("gb").innerText = data["Gradient Boosting"];
    document.getElementById("stack").innerText = data["Stacking Model"];

    const verdict = data["Final Verdict"];
    const badge = document.getElementById("finalVerdict");

    badge.innerText = verdict;

    if (verdict === "REAL") {
        badge.className = "verdict real";
    } else {
        badge.className = "verdict fake";
    }

    document.getElementById("results").classList.remove("hidden");
}

