<?php
session_start();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Facial Scan</title>
    <link rel="stylesheet" href="style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        /* Improved container styling for better fit */
        .container {
            width: 100%;
            max-width: 800px;
            padding: 20px;
            margin: 40px auto;
        }
        
        .form-container {
            background-color: white;
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        h2 {
            font-size: 30px;
            margin-bottom: 30px;
            color: #2c3e50;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }
        
        h2 i {
            color: #2c3e50;
            font-size: 36px;
        }
        
        /* Webcam container */
        .webcam-container {
            position: relative;
            width: 100%;
            max-width: 500px;
            margin: 0 auto 30px;
            border-radius: 10px;
            overflow: hidden;
        }
        
        #video {
            width: 100%;
            height: auto;
            border-radius: 10px;
            display: block;
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
            border: 2px dashed #e53e3e;
            border-radius: 100px 100px 70px 70px;
            transition: all 0.3s ease;
        }
        
        .face-outline.success {
            border: 2px solid #4caf50;
            box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.3);
        }
        
        .recording-indicator {
            position: absolute;
            top: 16px;
            left: 16px;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: #000000;
            display: none;
        }
        
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0.4; }
            100% { opacity: 1; }
        }
        
        .recording-indicator.active {
            display: block;
            animation: blink 1s infinite;
        }
        
        .challenge-container {
            background-color: rgba(52, 152, 219, 0.1);
            border-left: 4px solid #3498db;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            text-align: left;
        }
        
        /* Improved two-sided layout */
        .option-container {
            display: flex;
            margin: 40px 0;
            gap: 30px;
        }
        
        .option-box {
            flex: 1;
            padding: 40px 30px;
            border-radius: 12px;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: space-between;
            min-height: 320px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        
        .option-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.1);
        }
        
        .option-box h3 {
            margin: 0 0 15px 0;
            color: #2c3e50;
            font-size: 24px;
            font-weight: 600;
        }
        
        .option-box p {
            color: #7f8c8d;
            margin-bottom: 20px;
            font-size: 16px;
            line-height: 1.6;
        }
        
        .option-left {
            background-color: #e3f2fd;
            border-left: 5px solid #2196f3;
        }
        
        .option-right {
            background-color: #fffde7;
            border-left: 5px solid #ffc107;
        }
        
        .option-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 15px 30px;
            border-radius: 30px;
            font-size: 16px;
            font-weight: 600;
            text-decoration: none;
            margin-top: 30px;
            transition: all 0.3s ease;
            width: 200px;
        }
        
        .option-btn i {
            margin-right: 10px;
        }
        
        .option-btn.blue {
            background-color: #2196f3;
            color: white;
        }
        
        .option-btn.yellow {
            background-color: #ffc107;
            color: #5f4339;
        }
        
        .option-btn.blue:hover {
            background-color: #1976d2;
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(33, 150, 243, 0.3);
        }
        
        .option-btn.yellow:hover {
            background-color: #f9a825;
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(255, 193, 7, 0.3);
        }
        
        .option-icon {
            font-size: 64px;
            margin-bottom: 25px;
            color: #34495e;
        }
        
        /* Improved status message */
        .status-info {
            padding: 15px;
            background-color: #e3f2fd;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 16px;
            color: #1976d2;
        }
        
        .status-success {
            background-color: #e8f5e9;
            color: #388e3c;
        }
        
        .status-warning {
            background-color: #fff8e1;
            color: #f57c00;
        }
        
        .status-error {
            background-color: #ffebee;
            color: #d32f2f;
        }
        
        /* Added styles for buttons and footer */
        .action-buttons {
            display: flex;
            justify-content: center;
            margin: 25px 0;
            gap: 20px;
        }
        
        .action-buttons button {
            padding: 12px 24px;
            font-size: 16px;
            border-radius: 30px;
            border: none;
            background-color: #3498db;
            color: white;
            cursor: pointer;
            transition: all 0.3s;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 140px;
        }
        
        .action-buttons button i {
            margin-right: 8px;
        }
        
        .action-buttons button:hover {
            background-color: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        
        .form-footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
            color: #718096;
            font-size: 15px;
        }
        
        .form-footer a {
            color: #3498db;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s;
            margin: 0 5px;
        }
        
        .form-footer a:hover {
            color: #2980b9;
            text-decoration: underline;
        }
        
        /* Initial view & verification view */
        #initial-view {
            display: block;
        }
        
        #verification-view {
            display: none;
        }
        
        /* Progress bar styling */
        .progress-container {
            margin: 25px auto;
            width: 100%;
            max-width: 500px;
        }
        
        .progress-bar {
            height: 10px;
            background-color: #000000;
            border-radius: 5px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background-color: #000000;
            width: 0%;
            transition: width 0.3s ease;
        }
        
        /* Verify button styling */
        .verify-btn {
            background-color: #4caf50;
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 30px;
            cursor: pointer;
            font-weight: 600;
            font-size: 16px;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 180px;
            margin: 20px auto;
        }
        
        .verify-btn:hover {
            background-color: #45a049;
            transform: translateY(-3px);
            box-shadow: 0 10px 20px rgba(76, 175, 80, 0.3);
        }
        
        .verify-btn i {
            margin-right: 10px;
        }
        
        .verify-btn:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        /* Manual login form styling */
        #manual-login {
            margin-top: 30px;
            padding: 20px;
            background-color: #f8fafc;
            border-radius: 10px;
        }
        
        #manual-login h3 {
            margin-bottom: 20px;
            color: #2c3e50;
            font-size: 20px;
        }
        
        #manual-login input {
            width: 100%;
            padding: 12px 15px;
            margin-bottom: 15px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            font-size: 16px;
        }
        
        #manual-login input:focus {
            border-color: #3498db;
            outline: none;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.2);
        }
        
        #manual-login button {
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 20px;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
            transition: all 0.3s;
        }
        
        #manual-login button:hover {
            background-color: #2980b9;
        }
    </style>
</head>
<body>
<div class="container">
    <div class="form-container">
        <h2><i class="fas fa-shield-alt"></i> Facial Verification</h2>
        
        <div id="initial-view">
            <div id="status" class="status-info">Choose an option below to begin.</div>
            
            <div class="option-container">
                <!-- Left Side - Already Registered -->
                <div class="option-box option-left">
                    <div>
                        <div class="option-icon">
                            <i class="fas fa-user-shield"></i>
                        </div>
                        <h3>Already Registered?</h3>
                        <p>Begin facial verification to access your account.</p>
                    </div>
                    <a href="#" class="option-btn blue" id="startVerificationBtn">
                        <i class="fas fa-play"></i> Start
                    </a>
                </div>
                
                <!-- Right Side - New User -->
                <div class="option-box option-right">
                    <div>
                        <div class="option-icon">
                            <i class="fas fa-user-plus"></i>
                        </div>
                        <h3>New User?</h3>
                        <p>To use our services. Register Now!</p>
                    </div>
                    <a href="register.php" class="option-btn yellow">
                        <i class="fas fa-user-plus"></i> Register Now
                    </a>
                </div>
            </div>
        </div>
        
        <div id="verification-view">
            <div class="challenge-container" id="challengeContainer" style="display: none;">
                <h3><i class="fas fa-check-circle"></i> <span id="challengeTitle">Liveness Challenge</span></h3>
                <p id="challengeInstruction">Please follow the instructions to verify your identity.</p>
            </div>
            
            <div class="webcam-container" id="webcamContainer" style="display: none;">
                <video id="video" autoplay playsinline></video>
                <div class="face-guide">
                    <div class="face-outline" id="faceOutline"></div>
                    <div class="recording-indicator" id="recordingIndicator"></div>
                </div>
            </div>
            
            <div id="status-verification" class="status-info">Click "Start" to begin facial verification.</div>
            
            <div class="progress-container" id="progressContainer" style="display: none;">
                <div class="progress-bar" id="progressBar"></div>
            </div>
            
            <div class="action-buttons">
                <button id="startBtn" onclick="startVerification()"><i class="fas fa-play"></i> Start</button>
                <button id="retryBtn" style="display:none;" onclick="startVerification()"><i class="fas fa-redo"></i> Retry</button>
                <button id="manualBtn" style="display:none;" onclick="showManualLogin()"><i class="fas fa-keyboard"></i> Manual Verification</button>
                <button id="backBtn" style="display:none;" onclick="goBackToOptions()"><i class="fas fa-arrow-left"></i> Back</button>
            </div>

            <div id="manual-login" style="display: none;">
                <h3><i class="fas fa-user"></i> Manual Verification</h3>
                <form method="POST" action="verify.php">
                    <input type="text" name="username" placeholder="Username" required>
                    <input type="password" name="password" placeholder="Password" required>
                    <button type="submit"><i class="fas fa-sign-in-alt"></i> Verify</button>
                </form>
            </div>
        </div>
        
        <div class="form-footer">
            <p>Having trouble? <a href="help.php">Get Help</a> | <a href="contact.php">Contact Support</a></p>
        </div>
    </div>
</div>

<script>
const video = document.getElementById("video");
const statusVerification = document.getElementById("status-verification");
const retryBtn = document.getElementById("retryBtn");
const startBtn = document.getElementById("startBtn");
const manualBtn = document.getElementById("manualBtn");
const backBtn = document.getElementById("backBtn");
const progressContainer = document.getElementById("progressContainer");
const progressBar = document.getElementById("progressBar");
const challengeContainer = document.getElementById("challengeContainer");
const challengeTitle = document.getElementById("challengeTitle");
const challengeInstruction = document.getElementById("challengeInstruction");
const faceOutline = document.getElementById("faceOutline");
const recordingIndicator = document.getElementById("recordingIndicator");
const webcamContainer = document.getElementById("webcamContainer");
const startVerificationBtn = document.getElementById("startVerificationBtn");

let frames = [];
let stream;
let currentChallenge = '';
let faceDetectionInterval;

// Define a single head movement challenge for better detection
const challenges = [
    { type: 'head_movement', title: 'Head Movement', instruction: 'Please slowly turn your head left and right to verify' }
];

function getRandomChallenge() {
    // Now we always return the single head movement challenge
    return challenges[0];
}

function showManualLogin() {
    // Open login.php in a new page instead of showing the form inline
    window.location.href = "login.php";
}

function updateStatus(message, type = 'info') {
    statusVerification.textContent = message;
    statusVerification.className = `status-${type}`;
}

function goBackToOptions() {
    document.getElementById("initial-view").style.display = "block";
    document.getElementById("verification-view").style.display = "none";
    
    // Reset the verification view
    retryBtn.style.display = "none";
    manualBtn.style.display = "none";
    webcamContainer.style.display = "none";
    challengeContainer.style.display = "none";
    manualLogin.style.display = "none";
    progressContainer.style.display = "none";
    startBtn.style.display = "inline-block";
    
    // Stop any active webcam
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    
    // Clear any intervals
    if (faceDetectionInterval) {
        clearInterval(faceDetectionInterval);
    }
    
    updateStatus("Click \"Start\" to begin facial verification.", "info");
}

function startVerification() {
    // Reset UI
    retryBtn.style.display = "none";
    manualBtn.style.display = "none";
    startBtn.style.display = "none";
    webcamContainer.style.display = "block";
    updateStatus("Starting webcam...");

    // Select a random challenge
    const challenge = getRandomChallenge();
    currentChallenge = challenge.type;
    
    // Display challenge instructions
    challengeContainer.style.display = "block";
    challengeTitle.textContent = challenge.title;
    challengeInstruction.textContent = challenge.instruction;
    
    // Add helper text (only once)
    const instructionDetail = document.createElement('p');
    instructionDetail.innerHTML = '<i class="fas fa-info-circle"></i> This helps us verify you\'re a real person. Keep your face in the outline while moving.';
    instructionDetail.style.marginTop = '8px';
    instructionDetail.style.fontSize = '13px';
    instructionDetail.style.color = '#666';
    
    // Remove any existing helper text before adding new one
    const existingHelpers = challengeContainer.querySelectorAll('p');
    existingHelpers.forEach(helper => {
        if (helper !== challengeInstruction) {
            helper.remove();
        }
    });
    
    challengeContainer.appendChild(instructionDetail);

    // Clear any previous face detection interval
    if (faceDetectionInterval) {
        clearInterval(faceDetectionInterval);
    }

    // Access webcam
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
        
        updateStatus("Position your face in the outline, then click the 'Verify' button when ready.", "info");
        
        // Add a verify button instead of auto-capturing
        const verifyButtonContainer = document.createElement('div');
        verifyButtonContainer.style.textAlign = 'center';
        
        const verifyButton = document.createElement('button');
        verifyButton.innerHTML = '<i class="fas fa-camera"></i> Verify';
        verifyButton.className = 'verify-btn';
        verifyButton.disabled = !faceDetected;
        verifyButton.id = 'verifyButton';
        
        // Add the button after the status message
        verifyButtonContainer.appendChild(verifyButton);
        statusVerification.parentNode.insertBefore(verifyButtonContainer, statusVerification.nextSibling);
        
        // Add click event listener
        verifyButton.addEventListener('click', function() {
            // Only proceed if face is detected
            if (faceDetected) {
                // Remove the verify button
                verifyButton.remove();
                // Start capturing frames
                captureFrames();
            } else {
                updateStatus("⚠️ No face detected. Please position your face in the outline.", "warning");
            }
        });
    })
    .catch(err => {
        console.error("Webcam error:", err);
        updateStatus("❌ Webcam not accessible.", "error");
        manualBtn.style.display = "inline-block";
        backBtn.style.display = "inline-block";
    });
}

// Global variables
let faceDetected = false;

// Check if the face detection API is available
function checkApiAvailability() {
    return fetch("http://localhost:5000/health")
        .then(response => response.json())
        .then(data => data.status === 'healthy' || data.status === 'degraded')
        .catch(() => false);
}

// Use the actual API for face detection - with NO simulation fallback
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
    
    // Log what we're doing (for debugging)
    console.log("Checking for face in current frame...");
    
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
        console.log("Face detection response:", data);
        
        // If status is failed and message contains "No face detected" or "Face extraction failed"
        if (data.status === "failed" && 
            (data.message.includes("No face detected") || 
             data.message.includes("Face extraction failed") ||
             data.message.includes("Detected face too small") ||
             data.message.includes("Low face detection confidence"))) {
            
            if (faceDetected) {
                faceDetected = false;
                faceOutline.classList.remove('success');
                updateStatus("No face detected. Position your face in the outline.", "info");
                
                // Disable verify button if it exists
                const verifyButton = document.getElementById('verifyButton');
                if (verifyButton) {
                    verifyButton.disabled = true;
                }
            }
        } 
        // Otherwise, a face was successfully extracted
        else {
            if (!faceDetected) {
                faceDetected = true;
                faceOutline.classList.add('success');
                updateStatus("Face detected! Click 'Verify' when ready.", "success");
                
                // Enable verify button if it exists
                const verifyButton = document.getElementById('verifyButton');
                if (verifyButton) {
                    verifyButton.disabled = false;
                }
            }
        }
    })
    .catch(error => {
        console.error("Face detection API error:", error);
        faceDetected = false;
        faceOutline.classList.remove('success');
        updateStatus("⚠️ Face detection error. Please ensure API is running.", "error");
        
        // Disable verify button if it exists
        const verifyButton = document.getElementById('verifyButton');
        if (verifyButton) {
            verifyButton.disabled = true;
        }
    });
}

// Start face detection with NO simulation fallback
function startFaceDetection() {
    // Check API once and set up detection
    fetch("http://localhost:5000/health")
        .then(response => response.json())
        .then(data => {
            console.log("API health check:", data);
            
            if (data.status === 'healthy' || data.status === 'degraded') {
                // API is available, start detection
                faceDetectionInterval = setInterval(checkFaceDetection, 1000);
                updateStatus("Position your face in the outline.", "info");
            } else {
                // API is unhealthy
                updateStatus("⚠️ Face detection API is not fully operational.", "error");
                manualBtn.style.display = "inline-block";
                backBtn.style.display = "inline-block";
            }
        })
        .catch(error => {
            console.error("API health check failed:", error);
            updateStatus("❌ Cannot connect to face detection API.", "error");
            manualBtn.style.display = "inline-block";
            backBtn.style.display = "inline-block";
        });
}

// Capture frames with additional anti-replay measures
function captureFrames() {
    frames = [];
    let captured = 0;
    const total = 20; // Increased to 20 frames for better analysis
    progressContainer.style.display = "block";
    progressBar.style.width = "0%";

    updateStatus("📸 Capturing frames... Please turn your head slowly left and right.", "info");
    recordingIndicator.classList.add('active');
    
    // Add a challenge prompt at a random point during capture
    const randomChallengeFrame = Math.floor(Math.random() * (total - 5)) + 3;
    let challengeShown = false;
    let challengeType = Math.random() > 0.5 ? 'blink' : 'look_up';
    
    const interval = setInterval(() => {
        // Create timestamp for this frame to detect timing inconsistencies
        const timestamp = new Date().getTime();
        
        // Create a canvas with timestamp overlay to prevent video replay attacks
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        
        // Draw the video frame
        ctx.drawImage(video, 0, 0);
        
        // Add a barely visible random pattern overlay that changes each frame
        // This makes each frame unique and harder to spoof
        ctx.globalAlpha = 0.03;
        for (let i = 0; i < 5; i++) {
            const x = Math.random() * canvas.width;
            const y = Math.random() * canvas.height;
            const size = Math.random() * 20 + 5;
            ctx.fillStyle = `rgba(${Math.random() * 255},${Math.random() * 255},${Math.random() * 255},0.1)`;
            ctx.fillRect(x, y, size, size);
        }
        ctx.globalAlpha = 1.0;
        
        // Add timestamp as metadata
        const frameData = {
            image: canvas.toDataURL("image/jpeg"),
            timestamp: timestamp,
            challenge: challengeShown ? challengeType : null
        };
        
        frames.push(frameData);
        captured++;
        progressBar.style.width = `${(captured / total) * 100}%`;
        
        // If we're at the random challenge point, show a challenge
        if (captured === randomChallengeFrame && !challengeShown) {
            challengeShown = true;
            if (challengeType === 'blink') {
                updateStatus("Now, please blink quickly!", "warning");
            } else {
                updateStatus("Now, please look up briefly!", "warning");
            }
        }
        
        if (captured >= total) {
            clearInterval(interval);
            verifyWithAPI();
        }
    }, 250); // Faster frame rate for more detailed motion capture
}

function verifyWithAPI() {
    updateStatus("🔍 Analyzing your identity and liveness...");
    recordingIndicator.classList.remove('active');
    
    if (frames.length < 6) {
        updateStatus("❌ Not enough frames captured. Please try again.", "error");
        retryBtn.style.display = "inline-block";
        manualBtn.style.display = "inline-block";
        backBtn.style.display = "inline-block";
        return;
    }
    
    // Extract base64 data from all frames
    const mainImageData = frames[0].image.split(',')[1];
    const additionalFramesData = frames.slice(1).map(frame => frame.image.split(',')[1]);
    
    // Extract timestamps for timing analysis
    const timestamps = frames.map(frame => frame.timestamp);
    
    // Calculate the time differences between frames
    const timeDiffs = [];
    for (let i = 1; i < timestamps.length; i++) {
        timeDiffs.push(timestamps[i] - timestamps[i-1]);
    }
    
    // Analyze timing patterns to detect pre-recorded videos
    const avgTimeDiff = timeDiffs.reduce((sum, diff) => sum + diff, 0) / timeDiffs.length;
    const timingVariance = Math.sqrt(
        timeDiffs.reduce((sum, diff) => sum + Math.pow(diff - avgTimeDiff, 2), 0) / timeDiffs.length
    );
    
    // Add timing data to the API request for server-side analysis
    const authData = {
        image_base64: mainImageData,
        additional_frames: additionalFramesData,
        challenge_type: 'head_movement',
        timing_data: {
            timestamps: timestamps,
            avg_diff: avgTimeDiff,
            variance: timingVariance
        },
        client_random: Math.random().toString().substring(2, 10) // Add randomness to each request
    };
    
    // Add challenge response data if available
    const challengeFrames = frames.filter(frame => frame.challenge !== null);
    if (challengeFrames.length > 0) {
        authData.challenge_data = {
            type: challengeFrames[0].challenge,
            frame_index: frames.findIndex(frame => frame.challenge !== null)
        };
    }
    
    // Call the authentication endpoint
    fetch("http://localhost:5000/authenticate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(authData)
    })
    .then(res => {
        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
    })
    .then(data => {
        if (data.status === "success" && data.liveness_verified) {
            updateStatus("✅ Identity verified successfully!", "success");
            
            // Store session data
            fetch("store_session.php", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    user_id: data.user_id,
                    liveness_score: data.liveness_score,
                    similarity: data.similarity
                })
            })
            .then(response => response.json())
            .then(result => {
                if (result.status === "success") {
                    // Redirect to pin.php as specified
                    window.location.href = "pin.php";
                } else {
                    updateStatus("✅ Verification successful, but session storage failed.", "warning");
                    retryBtn.style.display = "inline-block";
                }
            })
            .catch(error => {
                console.error("Session storage error:", error);
                updateStatus("✅ Verification successful, but session storage failed.", "warning");
                retryBtn.style.display = "inline-block";
            });
        } else {
            // Handle failure with detailed messages
            if (!data.liveness_verified) {
                if (data.message && data.message.includes("spoof")) {
                    updateStatus("🚫 Spoofing attempt detected. Please try again with a real person.", "error");
                } else if (data.message && data.message.includes("static")) {
                    updateStatus("🚫 Video replay detected. Live verification required.", "error");
                } else {
                    updateStatus(`❌ Liveness check failed: ${data.message || "Please follow the instructions carefully."}`, "error");
                }
            } else if (data.message && data.message.includes("not recognized")) {
                updateStatus("⚠️ Face not recognized. Please try again or use manual verification.", "warning");
            } else {
                updateStatus(`❌ Verification failed: ${data.message || "Unknown error"}`, "error");
            }
            
            retryBtn.style.display = "inline-block";
            manualBtn.style.display = "inline-block";
        }
    })
    .catch(error => {
        console.error("API error:", error);
        updateStatus("❌ Server error during verification. Please try again later.", "error");
        retryBtn.style.display = "inline-block";
        manualBtn.style.display = "inline-block";
    });
}

    // Add some styles for the verify button
    const style = document.createElement('style');
    style.textContent = `
        .verify-btn {
            background-color: #4caf50;
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 50px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin: 15px auto;
        }
        .verify-btn:hover {
            background-color: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3);
        }
        .verify-btn i {
            margin-right: 8px;
        }
        .verify-btn:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
    `;
    document.head.appendChild(style);

// Stop webcam when page is unloaded
window.addEventListener('beforeunload', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    if (faceDetectionInterval) {
        clearInterval(faceDetectionInterval);
    }
});

function showVerificationView() {
    document.getElementById("initial-view").style.display = "none";
    document.getElementById("verification-view").style.display = "block";
}

// Initialize on page load
window.onload = function() {
    // Show initial view, hide verification view
    document.getElementById('initial-view').style.display = 'block';
    document.getElementById('verification-view').style.display = 'none';
    
    // Add event listener for the "Start Verification" button
    startVerificationBtn.addEventListener('click', function(e) {
        e.preventDefault();
        showVerificationView();
    });
}
</script>
</body>
</html>