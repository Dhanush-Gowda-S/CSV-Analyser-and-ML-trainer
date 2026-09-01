// ======================================================
//  START TRAINING (AJAX)
// ======================================================
function startTrain(form) {
    event.preventDefault();

    const formData = new FormData(form);

    // Show training modal
    const modal = new bootstrap.Modal(document.getElementById("trainModal"));
    modal.show();

    const bar = document.getElementById("progress-bar");
    bar.style.width = "5%";
    bar.innerText = "Starting...";

    // Start training
    fetch("/start_train", {
        method: "POST",
        body: formData
    })
    .then(r => r.json())
    .then(data => {
        if (!data.task_id) {
            alert("❌ Training could not start. No task ID returned.");
            return;
        }

        pollStatus(data.task_id);  // begin polling
    })
    .catch(err => {
        console.error("Training error:", err);
        alert("❌ Failed to start training.");
    });

    return false;
}



// ======================================================
//  POLL TRAINING STATUS
// ======================================================
let pollInterval = null;

function pollStatus(taskId) {
    const url = `/train_status/${taskId}`;
    const bar = document.getElementById("progress-bar");

    pollInterval = setInterval(() => {
        fetch(url)
            .then(r => r.json())
            .then(data => {

                // -------------------------------------------------
                // 🟥 ERROR from backend — stop polling
                // -------------------------------------------------
                if (data.error) {
                    clearInterval(pollInterval);
                    alert("❌ Training Error: " + data.error);
                    return;
                }

                // -------------------------------------------------
                // 🟦 Update progress bar
                // -------------------------------------------------
                let percent = data.percent || data.progress || 0;

                if (percent < 3) percent = 3;
                bar.style.width = percent + "%";
                bar.innerText = percent + "%";

                // -------------------------------------------------
                // 🟩 Detect finished statuses
                // Accepts: "done", "completed", or percent >= 100
                // -------------------------------------------------
                if (
                    data.status === "done" ||
                    data.status === "completed" ||
                    percent >= 100
                ) {
                    clearInterval(pollInterval);
                    bar.style.width = "100%";
                    bar.innerText = "100%";

                    showResults(data);
                }
            })
            .catch(err => {
                console.log("Polling error:", err);
            });

    }, 1200);
}



// ======================================================
//  SHOW FINAL RESULTS MODAL
// ======================================================
function showResults(data) {

    // Hide training modal
    const trainModal = bootstrap.Modal.getInstance(document.getElementById("trainModal"));
    if (trainModal) trainModal.hide();

    // -------------------------------------------
    // 🟧 Fill Summary
    // -------------------------------------------
    const summaryBox = document.getElementById("modelSummary");

    if (data.metrics) {
        summaryBox.innerText = JSON.stringify(data.metrics, null, 2);
    } else {
        summaryBox.innerText = "No metrics returned.";
    }

    // -------------------------------------------
    // 🟨 Feature Importance Chart
    // -------------------------------------------
    if (data.feature_importance && Array.isArray(data.feature_importance)) {
        const ctx = document.getElementById("featureChart").getContext("2d");

        const labels = data.feature_importance.map(x => x.feature);
        const values = data.feature_importance.map(x => x.value);

        new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "Feature Importance",
                    data: values,
                    backgroundColor: "rgba(0, 180, 80, 0.8)",
                    borderColor: "#00a85a",
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    }

    // -------------------------------------------
    // 🟩 Show completed modal
    // -------------------------------------------
    const resultModal = new bootstrap.Modal(document.getElementById("resultModal"));
    resultModal.show();
}
