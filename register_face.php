<?php
session_start();
require_once 'config.php';

if (!isset($_SESSION['user_id'])) {
    header("Location: login.php");
    exit("Unauthorized access. Please log in.");
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Register Face - Facial Pay System</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link rel="stylesheet" href="style.css">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f0f4f8;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        
        .container {
            width: 100%;
            max-width: 600px;
            padding: 20px;
        }
        
        .form-container {
            background-color: #ffffff;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            padding: 40px;
            text-align: center;
            transition: all 0.3s ease;
            animation: fadeIn 0.6s ease-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        h2 {
            color: #2c3e50;
            margin-bottom: 25px;
            font-size: 28px;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }
        
        h2 i {
            color: #3498db;
        }
        
        .webcam-container {
            position: relative;
            margin: 0 auto 25px;
            max-width: 480px;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        video {
            width: 100%;
            height: auto;
            display: block;
            background-color: #f8fafc;
            border-radius: 12px;
        }
        
        .face-guide {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 10;
        }
        
        .face-outline {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 180px;
            height: 240px;
            border: 2px dashed #3498db;
            border-radius: 100px 100px 70px 70px;
            transition: all 0.3s ease;
        }
        
        .recording-indicator {
            position: absolute;
            top: 16px;
            left: 16px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: #e53e3e;
        }
        
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0.4; }
            100% { opacity: 1; }
        }
        
        .recording-indicator.active {
            animation: blink 1s infinite;
        }
        
        .instructions {
            background-color: rgba(52, 152, 219, 0.1);
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 8px;
            text-align: left;
        }
        
        .instructions h3 {
            margin-top: 0;
            color: #2c3e50;
            font-size: 18px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .instructions p {
            margin-bottom: 0;
            color: #4a5568;
            line-height: 1.6;
        }
        
        .instructions ul {
            margin: 10px 0 0 0;
            padding-left: 20px;
            color: #4a5568;
        }
        
        .instructions li {
            margin-bottom: 8px;
        }
        
        .status-message {
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            background-color: #e3f2fd;
            color: #1976d2;
            text-align: center;
            font-size: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        
        .status-message i {
            font-size: 18px;
        }
        
        .status-success {
            background-color: #e8f5e9;
            color: #388e3c;
        }
        
        .status-warning {
            background-color: #fffde7;
            color: #f57c00;
        }
        
        .status-error {
            background-color: #ffebee;
            color: #d32f2f;
        }
        
        .action-buttons {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 25px;
        }
        
        .btn {
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            border: none;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        
        .btn-primary {
            background-color: #3498db;
            color: white;
        }
        
        .btn-primary:hover {
            background-color: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 7px 14px rgba(0, 0, 0, 0.1);
        }
        
        .btn-secondary {
            background-color: #f8fafc;
            color: #64748b;
            border: 1px solid #e2e8f0;
        }
        
        .btn-secondary:hover {
            background-color: #f1f5f9;
            border-color: #cbd5e1;
        }
        
        .btn-warning {
            background-color: #f97316;
            color: white;
        }
        
        .btn-warning:hover {
            background-color: #ea580c;
        }
        
        .progress-container {
            margin: 25px auto;
            max-width: 400px;
        }
        
        .progress-bar {
            height: 8px;
            background-color: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background-color: #3498db;
            width: 0%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        
        .frame-counter {
            font-size: 14px;
            color: #64748b;
            margin-top: 6px;
            text-align: right;
        }
        
        .challenge-container {
            padding: 15px;
            margin: 20px 0;
            border-radius: 8px;
            background-color: #ffe8cc;
            border-left: 4px solid #ed8936;
            display: none;
        }
        
        .challenge-container h3 {
            margin-top: 0;
            color: #c05621;
            font-size: 18px;
        }
        
        .challenge-container p {
            margin-bottom: 0;
            color: #9c4221;
        }
        
        /* Tooltip styles */
        .tooltip {
            position: relative;
            display: inline-block;
            margin-left: 5px;
            cursor: help;
        }
        
        .tooltip i {
            color: #3498db;
            font-size: 16px;
        }
        
        .tooltip .tooltip-content {
            visibility: hidden;
            width: 250px;
            background-color: #2c3e50;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
            font-weight: normal;
            font-size: 14px;
            line-height: 1.6;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .tooltip .tooltip-content::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #2c3e50 transparent transparent transparent;
        }
        
        .tooltip:hover .tooltip-content {
            visibility: visible;
            opacity: 1;
        }
    </style>
</head>
<body>
<div class="container">
    <div class="form-container">
        <h2><i class="fas fa-camera"></i> Register Your Face</h2>
        
        <div class="instructions">
            <h3><i class="fas fa-info-circle"></i> Instructions</h3>
            <p>Follow these steps to register your face:</p>
            <ul>
                <li>Position your face within the outline</li>
                <li>Ensure good lighting on your face</li>
                <li>Remove glasses or accessories that cover your face</li>
                <li>During the process, you may be asked to perform simple actions like blinking</li>
            </ul>
        </div>
        
        <div class="webcam-container">
            <video id="video" autoplay playsinline></video>
            <div class="face-guide">
                <div class="face-outline" id="faceOutline"></div>
                <div class="recording-indicator" id="recordingIndicator"></div>
            </div>
        </div>
        
        <div class="challenge-container" id="challengeContainer">
            <h3><i class="fas fa-exclamation-triangle"></i> <span id="challengeTitle">Liveness Challenge</span></h3>
            <p id="challengeInstruction">Please follow the instructions to verify you're a real person.</p>
        </div>
        
        <div id="status" class="status-message">
            <i class="fas fa-camera"></i> Position your face in the outline above
        </div>
        
        <div class="progress-container" id="progressContainer" style="display: none;">
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <div class="frame-counter" id="frameCounter">0/10 frames captured</div>
        </div>
        
        <div class="action-buttons">
            <button id="registerBtn" class="btn btn-primary" onclick="submitFrames()">
                <i class="fas fa-user-plus"></i> Register Face
            </button>
            <button id="retryBtn" class="btn btn-secondary" onclick="retryLiveness()" style="display: none;">
                <i class="fas fa-redo"></i> Retry
            </button>
            <button id="resetBtn" class="btn btn-warning" onclick="resetProcess()" style="display: none;">
                <i class="fas fa-sync-alt"></i> Reset
            </button>
        </div>
    </div>
</div>

<script>
    const video = document.getElementById("video");
    const statusDiv = document.getElementById("status");
    const retryBtn = document.getElementById("retryBtn");
    const resetBtn = document.getElementById("resetBtn");
    const registerBtn = document.getElementById("registerBtn");
    const progressContainer = document.getElementById("progressContainer");
    const progressFill = document.getElementById("progressFill");
    const frameCounter = document.getElementById("frameCounter");
    const recordingIndicator = document.getElementById("recordingIndicator");
    const faceOutline = document.getElementById("faceOutline");
    const challengeContainer = document.getElementById("challengeContainer");
    const challengeTitle = document.getElementById("challengeTitle");
    const challengeInstruction = document.getElementById("challengeInstruction");
    
    let frames = [];
    let stream;
    let captureInterval;
    let faceDetected = false;
    let faceDetectionInterval;

    function updateStatus(message, type = 'info') {
        statusDiv.textContent = message;
        statusDiv.className = `status-message status-${type}`;
        
        // Add appropriate icon based on status type
        let icon = '';
        switch(type) {
            case 'success':
                icon = '<i class="fas fa-check-circle"></i>';
                break;
            case 'error':
                icon = '<i class="fas fa-exclamation-circle"></i>';
                break;
            case 'warning':
                icon = '<i class="fas fa-exclamation-triangle"></i>';
                break;
            default:
                icon = '<i class="fas fa-info-circle"></i>';
        }
        
        statusDiv.innerHTML = `${icon} ${message}`;
    }
    
    function startCamera() {
        updateStatus("Starting camera...");
        
        navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            } 
        })
        .then(s => {
            stream = s;
            video.srcObject = stream;
            video.play();
            
            // Start face detection
            startFaceDetection();
            updateStatus("Position your face within the outline", "info");
        })
        .catch(err => {
            console.error("Camera error:", err);
            updateStatus("Camera access denied. Please enable camera permissions.", "error");
        });
    }
    
    function checkFaceDetection() {
        if (!video.videoWidth) return;
        
        // Create a canvas to capture the current frame
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0);
        
        // Convert to base64
        const imageData = canvas.toDataURL("image/jpeg").split(',')[1];
        
        // Call the API to detect face
        fetch("http://localhost:5000/recognize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                image_base64: imageData 
            })
        })
        .then(res => {
            if (!res.ok) {
                throw new Error(`HTTP error! status: ${res.status}`);
            }
            return res.json();
        })
        .then(data => {
            if (data.status === "failed" && 
                (data.message.includes("No face detected") || 
                 data.message.includes("Face extraction failed") ||
                 data.message.includes("Detected face too small") ||
                 data.message.includes("Low face detection confidence"))) {
                
                if (faceDetected) {
                    faceDetected = false;
                    faceOutline.style.borderColor = "#e53e3e";
                    faceOutline.style.borderStyle = "dashed";
                    updateStatus("No face detected. Position your face in the outline.", "warning");
                }
            } 
            else {
                if (!faceDetected) {
                    faceDetected = true;
                    faceOutline.style.borderColor = "#4caf50";
                    faceOutline.style.borderStyle = "solid";
                    updateStatus("Face detected! You can now register your face.", "success");
                }
            }
        })
        .catch(error => {
            console.error("Face detection API error:", error);
            faceDetected = false;
            faceOutline.style.borderColor = "#e53e3e";
            faceOutline.style.borderStyle = "dashed";
            updateStatus("Face detection error. Please ensure API is running.", "error");
        });
    }
    
    function startFaceDetection() {
        // Check API once and set up detection
        fetch("http://localhost:5000/health")
            .then(response => response.json())
            .then(data => {
                if (data.status === 'healthy' || data.status === 'degraded') {
                    // API is available, start detection
                    faceDetectionInterval = setInterval(checkFaceDetection, 1000);
                } else {
                    // API is unhealthy
                    updateStatus("Face detection API is not operational.", "error");
                }
            })
            .catch(error => {
                console.error("API health check failed:", error);
                updateStatus("Cannot connect to face detection API.", "error");
            });
    }

    function captureFrames() {
        // Clear any previous capture
        if (captureInterval) {
            clearInterval(captureInterval);
        }
        
        frames = [];
        let captured = 0;
        const totalFrames = 10;
        
        progressContainer.style.display = "block";
        progressFill.style.width = "0%";
        frameCounter.textContent = `0/${totalFrames} frames captured`;
        
        recordingIndicator.classList.add('active');
        updateStatus("Capturing frames... Please keep your face in the outline", "info");
        
        captureInterval = setInterval(() => {
            // Create a canvas to capture the current frame
            const canvas = document.createElement("canvas");
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext("2d");
            ctx.drawImage(video, 0, 0);
            
            // Add timestamp as metadata
            const frameData = {
                image: canvas.toDataURL("image/jpeg"),
                timestamp: new Date().getTime()
            };
            
            frames.push(frameData);
            captured++;
            
            // Update progress UI
            const progress = (captured / totalFrames) * 100;
            progressFill.style.width = `${progress}%`;
            frameCounter.textContent = `${captured}/${totalFrames} frames captured`;
            
            if (captured >= totalFrames) {
                clearInterval(captureInterval);
                recordingIndicator.classList.remove('active');
                updateStatus("✅ Frames captured successfully. Ready for registration.", "success");
                resetBtn.style.display = "inline-block";
            }
        }, 500);
    }

    function submitFrames() {
        if (!faceDetected) {
            updateStatus("Please position your face within the outline first", "warning");
            return;
        }
        
        if (frames.length === 0) {
            // If no frames have been captured yet, start capturing
            captureFrames();
            return;
        }
        
        if (frames.length < 10) {
            updateStatus("Please wait for all frames to be captured", "warning");
            return;
        }

        updateStatus("Initiating liveness challenge...", "info");
        retryBtn.style.display = "none";
        resetBtn.style.display = "none";
        registerBtn.disabled = true;

        // Show challenge container with instructions
        challengeContainer.style.display = "block";
        challengeTitle.textContent = "Liveness Challenge";
        challengeInstruction.textContent = "Please blink naturally a few times while we verify your identity.";

        fetch("http://localhost:5000/liveness_challenge", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ challenge_type: "blink" })
        })
        .then(res => res.json())
        .then(challenge => {
            if (!challenge.session_id) {
                updateStatus("Failed to initiate liveness challenge", "error");
                retryBtn.style.display = "inline-block";
                resetBtn.style.display = "inline-block";
                registerBtn.disabled = false;
                return;
            }

            const session_id = challenge.session_id;
            updateStatus("Analyzing your face... Please wait", "info");

            // Extract base64 data from frames
            const frameImages = frames.map(frame => frame.image.split(',')[1]);
            
            // Create a batch mechanism to send frames in groups of 3
            const batchSize = 3;
            const batches = [];
            
            for (let i = 0; i < frameImages.length; i += batchSize) {
                batches.push(frameImages.slice(i, i + batchSize));
            }
            
            // Send frames in batches with progress updates
            let batchesCompleted = 0;
            const totalBatches = batches.length;
            
            const sendNextBatch = (index) => {
                if (index >= batches.length) {
                    // All batches sent, now verify liveness
                    verifyLiveness(session_id);
                    return;
                }
                
                const batch = batches[index];
                Promise.all(batch.map(frame => {
                    return fetch("http://localhost:5000/submit_liveness_frame", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ session_id, frame })
                    });
                }))
                .then(() => {
                    batchesCompleted++;
                    const progress = (batchesCompleted / totalBatches) * 100;
                    progressFill.style.width = `${progress}%`;
                    frameCounter.textContent = `Processing: ${batchesCompleted}/${totalBatches} batches`;
                    
                    // Send next batch
                    sendNextBatch(index + 1);
                })
                .catch(error => {
                    console.error("Error sending batch:", error);
                    updateStatus("Error processing frames", "error");
                    retryBtn.style.display = "inline-block";
                    resetBtn.style.display = "inline-block";
                    registerBtn.disabled = false;
                });
            };
            
            // Start sending batches
            sendNextBatch(0);
        })
        .catch(error => {
            console.error("API error:", error);
            updateStatus("Error connecting to server", "error");
            retryBtn.style.display = "inline-block";
            resetBtn.style.display = "inline-block";
            registerBtn.disabled = false;
        });
    }
    
    function verifyLiveness(session_id) {
        updateStatus("Verifying liveness...", "info");
        
        fetch("http://localhost:5000/verify_liveness", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id })
        })
        .then(res => res.json())
        .then(result => {
            if (result.status === "success" && result.liveness_verified) {
                updateStatus("Liveness verified! Registering your face...", "success");
                challengeContainer.style.display = "none";
                
                // Select a good frame for registration (middle frame)
                const mainImage = frames[Math.floor(frames.length / 2)].image.split(',')[1];
                
                fetch("http://localhost:5000/register", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        user_id: <?php echo (int)$_SESSION['user_id']; ?>,
                        image_base64: mainImage,
                        liveness_session_id: session_id
                    })
                })
                .then(res => res.json())
                .then(data => {
                    if (data.status === "success") {
                        updateStatus("Face registered successfully!", "success");
                        
                        // Show success message before redirecting
                        setTimeout(() => {
                            window.location.href = "dashboard.php";
                        }, 1500);
                    } else {
                        updateStatus(`Registration failed: ${data.message}`, "error");
                        challengeContainer.style.display = "none";
                        
                        if (data.message.includes("No face")) {
                            updateStatus("No valid face detected in the frame. Please retry with your face clearly visible.", "error");
                        }
                        
                        retryBtn.style.display = "inline-block";
                        resetBtn.style.display = "inline-block";
                        registerBtn.disabled = false;
                    }
                })
                .catch(error => {
                    console.error("Registration error:", error);
                    updateStatus("Server error during registration", "error");
                    challengeContainer.style.display = "none";
                    retryBtn.style.display = "inline-block";
                    resetBtn.style.display = "inline-block";
                    registerBtn.disabled = false;
                });
            } else {
                updateStatus("Liveness check failed. Please try again.", "error");
                challengeContainer.style.display = "none";
                retryBtn.style.display = "inline-block";
                resetBtn.style.display = "inline-block";
                registerBtn.disabled = false;
            }
        })
        .catch(error => {
            console.error("Liveness verification error:", error);
            updateStatus("Server error during liveness verification", "error");
            challengeContainer.style.display = "none";
            retryBtn.style.display = "inline-block";
            resetBtn.style.display = "inline-block";
            registerBtn.disabled = false;
        });
    }

    function retryLiveness() {
        challengeContainer.style.display = "none";
        frames = [];
        updateStatus("Retrying: capturing new frames...", "info");
        captureFrames();
        retryBtn.style.display = "none";
        resetBtn.style.display = "none";
        registerBtn.disabled = false;
    }

    function resetProcess() {
        challengeContainer.style.display = "none";
        frames = [];
        updateStatus("Resetting. Position your face in the outline.", "info");
        captureFrames();
        retryBtn.style.display = "none";
        resetBtn.style.display = "none";
        registerBtn.disabled = false;
    }
    
    // Cleanup function
    function cleanup() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        
        if (faceDetectionInterval) {
            clearInterval(faceDetectionInterval);
        }
        
        if (captureInterval) {
            clearInterval(captureInterval);
        }
    }

    // Initialize
    window.onload = startCamera;
    
    // Clean up on page unload
    window.addEventListener('beforeunload', cleanup);
</script>
</body>
</html>