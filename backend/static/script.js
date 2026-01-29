async function trainModel() {
    const fileInput = document.getElementById("csvFile");
    const model = document.getElementById("model").value;

    if (!fileInput.files.length) {
        alert("Upload CSV file first!");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("model", model);

    const response = await fetch("http://127.0.0.1:8000/train", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    const metricsDiv = document.getElementById("metrics");
    metricsDiv.innerHTML = "";

    for (let key in data.metrics) {
        metricsDiv.innerHTML += `
            <div class="metric-card">
                <h4>${key.toUpperCase()}</h4>
                <p>${data.metrics[key].toFixed(4)}</p>
            </div>
        `;
    }

    document.getElementById("rocImg").src =
        "data:image/png;base64," + data.roc_curve;

    document.getElementById("cmImg").src =
        "data:image/png;base64," + data.confusion_matrix;
}
