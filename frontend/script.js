
const modal = document.getElementById("tryOnModal");

let cameraStream = null;
let capturedPhoto = null;
let countdownTimer = null;
let sessionId = 0;
let isGenerating = false;

function cancelCountdown() {
    if (countdownTimer !== null) {
        clearInterval(countdownTimer);
        countdownTimer = null;
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
}

async function openTryOn() {
    cancelCountdown();
    stopCamera();

    const currentSession = ++sessionId;
    capturedPhoto = null;
    isGenerating = false;

    modal.classList.add("active");

    const preview = document.querySelector(".camera-placeholder");

    preview.innerHTML = `
        <video id="cameraVideo" autoplay playsinline muted
            style="width:100%;height:100%;object-fit:cover;">
        </video>
    `;

    let controls = document.getElementById("tryOnControls");

    if (!controls) {
        controls = document.createElement("div");
        controls.id = "tryOnControls";
        controls.style.marginBottom = "18px";
        preview.insertAdjacentElement("afterend", controls);
    }

    controls.innerHTML = `
        <button class="primary-button" id="captureButton"
            style="display:none;">
            RETAKE PHOTO
        </button>

        <button class="primary-button" id="generateButton"
            style="display:none;margin-left:8px;">
            GENERATE AI TRY-ON
        </button>

        <p id="tryOnStatus" style="margin-top:14px;">
            Starting camera...
        </p>
    `;

    document.getElementById("captureButton").onclick = openTryOn;
    document.getElementById("generateButton").onclick = generateTryOn;

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
        });

        // Prevent camera opening after the modal was closed.
        if (currentSession !== sessionId) {
            stream.getTracks().forEach(track => track.stop());
            return;
        }

        cameraStream = stream;

        const video = document.getElementById("cameraVideo");
        video.srcObject = cameraStream;

        await video.play();

        if (currentSession !== sessionId) return;

        startCountdown(currentSession);

    } catch (error) {
        if (currentSession !== sessionId) return;

        document.getElementById("tryOnStatus").textContent =
            "Camera error: " + error.message;
    }
}

function startCountdown(currentSession) {
    cancelCountdown();

    let seconds = 5;

    const status = document.getElementById("tryOnStatus");

    status.textContent = `Get ready! Capturing in ${seconds}...`;

    countdownTimer = setInterval(() => {
        if (currentSession !== sessionId || !modal.classList.contains("active")) {
            cancelCountdown();
            return;
        }

        seconds--;

        if (seconds > 0) {
            status.textContent =
                `Get ready! Capturing in ${seconds}...`;
        } else {
            cancelCountdown();
            capturePhoto(currentSession);
        }
    }, 1000);
}

function capturePhoto(currentSession = sessionId) {
    if (currentSession !== sessionId) return;

    const video = document.getElementById("cameraVideo");

    if (!video || !video.videoWidth) {
        document.getElementById("tryOnStatus").textContent =
            "Camera not ready. Please retake.";
        document.getElementById("captureButton").style.display =
            "inline-block";
        return;
    }

    const canvas = document.createElement("canvas");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    canvas.getContext("2d").drawImage(video, 0, 0);

    capturedPhoto = canvas.toDataURL("image/png");

    document.querySelector(".camera-placeholder").innerHTML =
        `<img src="${capturedPhoto}"
            style="width:100%;height:100%;object-fit:contain;">`;

    document.getElementById("captureButton").style.display =
        "inline-block";

    document.getElementById("captureButton").textContent =
        "RETAKE PHOTO";

    stopCamera();

    // Automatically start AI generation once.
    generateTryOn();
}

async function generateTryOn() {
    if (!capturedPhoto || isGenerating) return;

    isGenerating = true;

    const currentSession = sessionId;
    const photo = capturedPhoto;

    const button = document.getElementById("generateButton");
    const status = document.getElementById("tryOnStatus");

    button.disabled = true;
    button.textContent = "GENERATING...";
    button.style.display = "none";

    status.textContent =
        "AI is generating your virtual try-on. Please wait...";

    try {
        const response = await fetch("http://localhost:5000/api/tryon", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({photo})
        });

        const data = await response.json();

        // Ignore results from an old or closed session.
        if (currentSession !== sessionId) return;

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Generation failed");
        }

        document.querySelector(".camera-placeholder").innerHTML =
            `<img src="${data.image_url}"
                alt="AI-generated try-on"
                style="width:100%;height:100%;object-fit:contain;">`;

        status.textContent = "AI TRY-ON COMPLETE!";

        button.style.display = "none";

    } catch (error) {
        if (currentSession !== sessionId) return;

        status.textContent = "Error: " + error.message;

        button.disabled = false;
        button.textContent = "TRY AGAIN";
        button.style.display = "inline-block";

    } finally {
        if (currentSession === sessionId) {
            isGenerating = false;
        }
    }
}

function closeTryOn() {
    ++sessionId;
    cancelCountdown();
    stopCamera();

    capturedPhoto = null;
    isGenerating = false;

    modal.classList.remove("active");
}

modal.addEventListener("click", event => {
    if (event.target === modal) closeTryOn();
});

document.addEventListener("keydown", event => {
    if (event.key === "Escape") closeTryOn();
});
