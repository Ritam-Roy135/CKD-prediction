const form = document.getElementById("predictionForm");
const submitButton = document.getElementById("submitButton");
const resultSection = document.getElementById("result");
const resultCard = document.getElementById("resultCard");

const numericFields = [
    "age", "bp", "sg", "al", "su", "bgr", "bu",
    "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"
];

const categoricalFields = [
    "rbc", "pc", "pcc", "ba", "htn", "dm",
    "cad", "appet", "pe", "ane"
];

async function checkAPI() {
    try {
        const response = await fetch("/api/health");
        if (!response.ok) throw new Error("API unavailable");
        const data = await response.json();

        document.querySelector(".status-indicator").innerHTML =
            `<span></span> API connected`;

        document.querySelector(".topbar-right").innerHTML =
            `<span class="live-dot"></span><span>System operational</span>`;

        return data;
    } catch (error) {
        document.querySelector(".status-indicator").innerHTML =
            "API disconnected";
        console.error(error);
    }
}

function getFormData() {
    const payload = {};

    numericFields.forEach(field => {
        const value = form.elements[field].value.trim();
        payload[field] = value === "" ? null : Number(value);
    });

    categoricalFields.forEach(field => {
        const value = form.elements[field].value;
        payload[field] = value === "" ? null : value;
    });

    payload.model = document.getElementById("modelSelect").value;

    return payload;
}

function showResult(data) {
    resultSection.hidden = false;

    const isPositive = data.prediction === 1;

    resultCard.className = `result-card ${isPositive ? "positive" : "negative"}`;

    document.getElementById("resultStatus").textContent =
        isPositive ? "MODEL OUTPUT · CKD CLASS" : "MODEL OUTPUT · NOT CKD CLASS";

    document.getElementById("resultTitle").textContent =
        isPositive ? "CKD predicted" : "Not CKD predicted";

    document.getElementById("resultDescription").textContent =
        isPositive
            ? "The selected model classified this input as belonging to the CKD class."
            : "The selected model classified this input as belonging to the Not CKD class.";

    document.getElementById("resultModel").textContent = data.model;

    document.getElementById("resultProbability").textContent =
        data.ckd_probability === null
            ? "Not available"
            : `${data.ckd_probability}%`;

    document.getElementById("resultDisclaimer").textContent =
        data.disclaimer;

    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    submitButton.disabled = true;
    submitButton.innerHTML = "Processing...";

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(getFormData())
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction request failed");
        }

        showResult(data);

    } catch (error) {
        alert(error.message);
        console.error(error);
    } finally {
        submitButton.disabled = false;
        submitButton.innerHTML = 'Run prediction <span>→</span>';
    }
});

form.addEventListener("reset", () => {
    resultSection.hidden = true;
});

checkAPI();

// =====================================
// MODEL COMPARISON AND ANALYTICS
// =====================================

let performanceChartInstance = null;
let cvChartInstance = null;
let importanceChartInstance = null;
let cvMetricsChartInstance = null;

let analyticsData = null;

const algorithmNames = [
    "Random Forest",
    "Bagging Classifier",
    "AdaBoost",
    "Gradient Boosting"
];

const metricNames = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
];

function percentage(value) {
    return `${(Number(value) * 100).toFixed(2)}%`;
}

function renderPerformanceTable(rows) {
    const table = document.getElementById("performanceTable");

    table.innerHTML = rows.map(row => `
        <tr>
            <td>${row.Algorithm}</td>
            <td class="metric-value">${percentage(row.Accuracy)}</td>
            <td>${percentage(row.Precision)}</td>
            <td>${percentage(row.Recall)}</td>
            <td>${percentage(row["F1 Score"])}</td>
            <td>${percentage(row["ROC-AUC"])}</td>
        </tr>
    `).join("");
}

function renderPerformanceChart(rows) {
    const context = document.getElementById("performanceChart");

    if (performanceChartInstance) {
        performanceChartInstance.destroy();
    }

    performanceChartInstance = new Chart(context, {
        type: "bar",
        data: {
            labels: rows.map(row => row.Algorithm),
            datasets: metricNames.map((metric, index) => ({
                label: metric,
                data: rows.map(row => Number(row[metric])),
                backgroundColor: [
                    "#35bdb0", "#6289c7", "#e6ad62",
                    "#8c78c9", "#5aabbc"
                ][index],
                borderRadius: 5
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 0.9,
                    max: 1.01,
                    ticks: {
                        callback: value => `${(value * 100).toFixed(0)}%`
                    }
                }
            },
            plugins: {
                legend: { position: "bottom" }
            }
        }
    });
}

function renderCVChart(rows) {
    const context = document.getElementById("cvChart");

    if (cvChartInstance) cvChartInstance.destroy();

    cvChartInstance = new Chart(context, {
        type: "bar",
        data: {
            labels: rows.map(row => row.Algorithm),
            datasets: [{
                label: "Cross-validation accuracy",
                data: rows.map(row => Number(row.Accuracy)),
                backgroundColor: "#35bdb0",
                borderRadius: 6
            }, {
                label: "Accuracy standard deviation",
                data: rows.map(row => Number(row["Accuracy Std"])),
                backgroundColor: "#e6ad62",
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 0,
                    max: 1,
                    ticks: {
                        callback: value => `${(value * 100).toFixed(0)}%`
                    }
                }
            },
            plugins: { legend: { position: "bottom" } }
        }
    });
}

function renderImportanceChart(modelName) {
    const rows = analyticsData.feature_importance[modelName] || [];
    const context = document.getElementById("importanceChart");

    if (importanceChartInstance) importanceChartInstance.destroy();

    importanceChartInstance = new Chart(context, {
        type: "bar",
        data: {
            labels: rows.map(row => row.Feature),
            datasets: [{
                label: "Permutation importance",
                data: rows.map(row => Number(row.Importance)),
                backgroundColor: "#35bdb0",
                borderRadius: 5
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } }
        }
    });
}

function renderCVMetricsChart(rows) {
    const context = document.getElementById("cvMetricsChart");

    if (cvMetricsChartInstance) cvMetricsChartInstance.destroy();

    cvMetricsChartInstance = new Chart(context, {
        type: "bar",
        data: {
            labels: rows.map(row => row.Algorithm),
            datasets: metricNames.map((metric, index) => ({
                label: metric,
                data: rows.map(row => Number(row[metric])),
                backgroundColor: [
                    "#35bdb0", "#6289c7", "#e6ad62",
                    "#8c78c9", "#5aabbc"
                ][index],
                borderRadius: 4
            }))
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    min: 0.9,
                    max: 1.01,
                    ticks: {
                        callback: value => `${(value * 100).toFixed(0)}%`
                    }
                }
            },
            plugins: { legend: { position: "bottom" } }
        }
    });
}

async function loadAnalytics() {
    try {
        const response = await fetch("/api/analytics");
        if (!response.ok) throw new Error("Could not load analytics");

        analyticsData = await response.json();

        renderPerformanceTable(analyticsData.performance);
        renderPerformanceChart(analyticsData.performance);
        renderCVChart(analyticsData.cross_validation);
        renderCVMetricsChart(analyticsData.cross_validation);

        renderImportanceChart(
            document.getElementById("importanceModel").value
        );

    } catch (error) {
        console.error("Analytics error:", error);
        document.getElementById("performanceTable").innerHTML =
            `<tr><td colspan="6">Unable to load model results. Check the Flask API.</td></tr>`;
    }
}

document.getElementById("importanceModel").addEventListener("change", event => {
    if (analyticsData) renderImportanceChart(event.target.value);
});

loadAnalytics();

// Sidebar navigation active state
document.querySelectorAll(".nav-link").forEach(link => {
    link.addEventListener("click", () => {
        document.querySelectorAll(".nav-link").forEach(item => {
            item.classList.remove("active");
        });
        link.classList.add("active");
    });
});
