const modal = document.getElementById("tryOnModal");
let cameraStream = null;
let capturedPhoto = null;

async function openTryOn() {
    stopCamera();
    capturedPhoto = null;
    modal.classList.add("active");

    const preview = document.querySelector(".camera-placeholder");
    preview.innerHTML = `
        <video id="cameraVideo" autoplay playsinline muted
               style="width:100%;height:100%;object-fit:cover;"></video>
    `;

    let controls = document.getElementById("tryOnControls");

    if (!controls) {
        controls = document.createElement("div");
        controls.id = "tryOnControls";
        controls.style.marginBottom = "18px";
        preview.insertAdjacentElement("afterend", controls);
    }

    controls.innerHTML = `
        <button class="primary-button" id="captureButton">
            CAPTURE PHOTO
        </button>
        <button class="primary-button" id="generateButton"
                style="display:none;margin-left:8px;">
            GENERATE AI TRY-ON
        </button>
        <p id="tryOnStatus" style="margin-top:14px;"></p>
    `;

    document.getElementById("captureButton").onclick = capturePhoto;
    document.getElementById("generateButton").onclick = generateTryOn;

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
        });

        const video = document.getElementById("cameraVideo");
        video.srcObject = cameraStream;
        await video.play();
    } catch (error) {
        preview.textContent = "Camera error: " + error.message;
    }
}

function capturePhoto() {
    const video = document.getElementById("cameraVideo");

    if (!video || !video.videoWidth) {
        alert("Camera is not ready.");
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

    document.getElementById("captureButton").textContent = "RETAKE PHOTO";
    document.getElementById("captureButton").onclick = openTryOn;
    document.getElementById("generateButton").style.display = "inline-block";

    stopCamera();
}

async function generateTryOn() {
    if (!capturedPhoto) return;

    const button = document.getElementById("generateButton");
    const status = document.getElementById("tryOnStatus");

    button.disabled = true;
    button.textContent = "GENERATING...";
    status.textContent = "AI is processing your photo. Please wait...";

    try {
        const response = await fetch("http://localhost:5000/api/tryon", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({photo: capturedPhoto})
        });

        const data = await response.json();

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
        status.textContent = "Error: " + error.message;
        button.disabled = false;
        button.textContent = "TRY AGAIN";
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
}

function closeTryOn() {
    stopCamera();
    modal.classList.remove("active");
}

modal.addEventListener("click", event => {
    if (event.target === modal) closeTryOn();
});

document.addEventListener("keydown", event => {
    if (event.key === "Escape") closeTryOn();
});
